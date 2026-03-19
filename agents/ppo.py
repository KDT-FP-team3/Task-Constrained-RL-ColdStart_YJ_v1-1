import numpy as np
import random
from agents.base import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ActorCriticNetwork(nn.Module):
    """
    PPO용 Actor-Critic 네트워크
    ─────────────────────────────────────────────────────────
    구조:
        공유 레이어 → ┬→ Actor  (정책 π): 각 행동의 확률 분포 출력
                      └→ Critic (가치 V): 현재 상태의 가치 스칼라 출력

    DQN과 차이:
        DQN    → Q(s,a) 출력 (행동별 Q값)
        PPO    → π(a|s) 출력 (행동 확률) + V(s) 출력 (상태 가치)
    """
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()

        # 공유 특징 추출기
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor: 행동 확률 분포
        self.actor  = nn.Linear(hidden, num_actions)

        # Critic: 상태 가치 (스칼라)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)                    # 행동 로짓
        value  = self.critic(shared).squeeze(-1)       # 상태 가치
        return logits, value

    def get_action_and_value(self, state):
        """행동 샘플링 + log_prob + value 한 번에 반환"""
        logits, value = self.forward(state)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, state, action):
        """저장된 행동에 대한 log_prob, entropy, value 재계산 (학습용)"""
        logits, value = self.forward(state)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return log_prob, entropy, value


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO)

    DQN 계열과 근본적으로 다른 점
    ──────────────────────────────
    1. On-policy: 현재 정책으로 수집한 데이터만 학습에 사용
       (DQN은 Off-policy — 과거 경험 재사용 가능)

    2. Policy를 직접 학습: Q값이 아닌 행동 확률 분포를 출력

    3. Clipping: 정책 업데이트 크기를 clip_eps로 제한
       → 한 번에 너무 크게 바뀌면 학습 망가지는 것을 방지

    4. Entropy bonus: 탐색 장려 항
       → 정책이 너무 일찍 하나의 행동에 수렴하는 것 방지

    PPO Loss = Actor Loss + Critic Loss - Entropy Bonus
               ──────────   ───────────   ──────────────
               정책 개선    가치 정확도    탐색 유지
    """

    STATE_DIM = 5

    def __init__(self, env, use_constraints=True, lr=0.0003, gamma=0.85, eps=0.5,
                 hidden=128, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
                 update_epochs=4, rollout_len=64):
        super().__init__(env, use_constraints, lr, gamma, eps)

        if not TORCH_AVAILABLE:
            raise ImportError("PPO는 PyTorch가 필요합니다: pip install torch")

        self.num_actions   = 1 + env.vocab_size
        self.clip_eps      = clip_eps       # 정책 변화 허용 범위
        self.entropy_coef  = entropy_coef   # 탐색 장려 계수
        self.value_coef    = value_coef     # Critic loss 가중치
        self.update_epochs = update_epochs  # 같은 데이터로 몇 번 학습할지
        self.rollout_len   = rollout_len    # 몇 step 모아서 한 번 업데이트할지
        self.device        = torch.device("cpu")

        self.net       = ActorCriticNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # Rollout 버퍼 — On-policy라 매 업데이트마다 초기화
        self._reset_buffer()

    def get_model_name(self):
        return "PPO"

    def _reset_buffer(self):
        """On-policy 버퍼 초기화 — 업데이트 후 반드시 호출"""
        self.buf_states   = []
        self.buf_actions  = []
        self.buf_rewards  = []
        self.buf_logprobs = []
        self.buf_values   = []

    def _get_state_vec(self, step):
        vecs = [self.env.get_state_vector(step, i) for i in range(self.env.vocab_size)]
        return np.mean(vecs, axis=0).astype(np.float32)

    def _to_tensor(self, arr):
        return torch.FloatTensor(arr).to(self.device)

    def select_action(self, current_step):
        state_vec = self._get_state_vec(current_step)
        state_t   = self._to_tensor(state_vec).unsqueeze(0)

        with torch.no_grad():
            logits, value_t = self.net(state_t)

            # Q 마스킹: 제약 위반 종목 logit을 -inf로 설정
            if self.use_constraints:
                mask = self.env.get_constraint_mask(current_step)
                for ticker_idx, violated in enumerate(mask):
                    if violated:
                        logits[0, ticker_idx + 1] = -1e9

            dist     = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)

        action   = int(action_t.item())
        log_prob = float(log_prob_t.item())
        value    = float(value_t.item())

        if action == 0:
            chosen_ticker, ticker_idx = "CASH", 0
        else:
            ticker_idx    = action - 1
            chosen_ticker = self.env.tickers[ticker_idx]

        reward, raw_ret = self.env.get_step_reward(
            chosen_ticker, current_step, self.prev_action
        )

        # Rollout 버퍼에 저장 (On-policy)
        self.buf_states.append(state_vec)
        self.buf_actions.append(action)
        self.buf_rewards.append(reward)
        self.buf_logprobs.append(log_prob)
        self.buf_values.append(value)

        self.prev_action = action
        self._record(reward)
        return chosen_ticker, True, reward, state_vec, action, ticker_idx, raw_ret

    def learn(self, state, action, reward, next_step):
        """
        PPO는 rollout_len step마다 한 번 업데이트
        step마다 호출되지만 버퍼가 찰 때만 실제 학습
        """
        if len(self.buf_states) < self.rollout_len:
            return

        # ── Rollout 데이터 텐서 변환 ────────────────────────────────────
        s_t  = self._to_tensor(np.array(self.buf_states))
        a_t  = torch.LongTensor(self.buf_actions).to(self.device)
        r_t  = self._to_tensor(np.array(self.buf_rewards))
        lp_t = self._to_tensor(np.array(self.buf_logprobs))

        # ── Advantage 계산 (Returns - Values) ───────────────────────────
        # Discounted returns (뒤에서부터 누적)
        returns = []
        G = 0.0
        for r in reversed(self.buf_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        ret_t = self._to_tensor(np.array(returns))

        # Advantage 정규화 — 학습 안정성 향상
        val_t = self._to_tensor(np.array(self.buf_values))
        adv_t = ret_t - val_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── PPO update_epochs 번 반복 학습 ──────────────────────────────
        for _ in range(self.update_epochs):
            # 현재 정책으로 log_prob, entropy, value 재계산
            new_lp, entropy, new_val = self.net.evaluate(s_t, a_t)

            # Probability ratio: π_new / π_old
            ratio = torch.exp(new_lp - lp_t)

            # Clipped surrogate objective
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            actor_loss  = -torch.min(surr1, surr2).mean()

            # Critic loss (MSE)
            critic_loss = nn.MSELoss()(new_val, ret_t)

            # 전체 loss: Actor - Entropy bonus + Critic
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  # 그래디언트 클리핑
            self.optimizer.step()

        # On-policy: 업데이트 후 버퍼 초기화
        self._reset_buffer()

    def decay_epsilon(self, episode, total_episodes):
        """PPO는 entropy로 탐색을 관리하므로 epsilon은 참고용"""
        self.cosine_decay(episode, total_episodes)

    def get_avg_q(self):
        """PPO는 Q값 대신 Value 출력 — 모니터링용"""
        with torch.no_grad():
            _, value = self.net(self._to_tensor(
                np.zeros(self.STATE_DIM, dtype=np.float32)
            ).unsqueeze(0))
        return float(value.abs().item())