import numpy as np
import random
from agents.base import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Prioritized Experience Replay 버퍼 ─────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER)
    ─────────────────────────────────────────────────────────────────
    일반 Replay: 모든 경험을 동일 확률로 샘플링
    PER:         TD 에러가 큰 경험(= 아직 못 배운 상황)을 더 자주 샘플링

    priority = |TD_error| + epsilon_per   ← 0이 되지 않도록 epsilon 추가
    sample_prob ∝ priority ^ alpha        ← alpha=0이면 균등, 1이면 완전 우선순위

    Importance Sampling Weight (IS weight):
        학습 편향 보정 — 자주 뽑히는 샘플일수록 가중치를 낮춰서 보정
        w = (N × P(i))^(-beta)  → beta는 0에서 1로 선형 증가
    ─────────────────────────────────────────────────────────────────
    """
    def __init__(self, capacity=5000, alpha=0.6, beta_start=0.4, beta_end=1.0):
        self.capacity   = capacity
        self.alpha      = alpha       # 우선순위 강도 (0: 균등, 1: 완전 우선순위)
        self.beta       = beta_start  # IS 보정 강도 (학습하면서 1까지 증가)
        self.beta_end   = beta_end
        self.beta_start = beta_start

        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0           # 현재 쓰기 위치 (circular buffer)
        self.size       = 0

    def push(self, state, action, reward, next_state):
        """새 경험 추가 — 초기 우선순위는 현재 최대값으로 설정"""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state))
            self.size += 1
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, total_steps, max_steps):
        """우선순위 기반 샘플링 + IS 가중치 반환"""
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        # beta 선형 증가 (학습이 진행될수록 IS 보정 강화)
        self.beta = min(self.beta_end,
                        self.beta_start + (self.beta_end - self.beta_start)
                        * total_steps / max_steps)

        # IS 가중치 계산
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()   # 정규화

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        """학습 후 TD 에러로 우선순위 업데이트"""
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6   # epsilon_per = 1e-6

    def __len__(self):
        return self.size


# ── Dueling Network (Rainbow 공용) ─────────────────────────────────────────
class DuelingQNetwork(nn.Module):
    """Rainbow에서도 Dueling 구조 사용 (Double + Dueling + PER = Rainbow)"""
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        shared = self.shared(x)
        v = self.value_stream(shared)
        a = self.advantage_stream(shared)
        return v + (a - a.mean(dim=1, keepdim=True))


# ── Rainbow Agent ───────────────────────────────────────────────────────────
class RainbowAgent(BaseAgent):
    """
    Rainbow DQN = Double DQN + Dueling DQN + Prioritized Experience Replay

    기존 DQN 계열 대비 추가된 것:
    ─────────────────────────────────────────────────────────────────
    Double DQN  : TD target에서 행동 선택(Online) / Q 평가(Target) 분리
    Dueling DQN : Q = V(s) + A(s,a) - mean(A) 구조로 상태/행동 가치 분리
    PER         : TD 에러 큰 경험 우선 학습 → 중요한 상황 더 많이 학습
    ─────────────────────────────────────────────────────────────────

    왜 Rainbow가 더 나은가:
    - 하락장처럼 드물지만 중요한 상황(TD 에러 큰 날)을 더 많이 반복
    - "시장 상태" 학습(V)과 "종목 선택" 학습(A)이 명확히 분리
    - Q값 과대추정 없이 안정적 학습
    """

    STATE_DIM = 5

    def __init__(self, env, use_constraints=True, lr=0.001, gamma=0.85, eps=0.5,
                 hidden=128, target_update=20,
                 per_alpha=0.6, per_beta_start=0.4):
        super().__init__(env, use_constraints, lr, gamma, eps)

        if not TORCH_AVAILABLE:
            raise ImportError("Rainbow는 PyTorch가 필요합니다: pip install torch")

        self.num_actions   = 1 + env.vocab_size
        self.target_update = target_update
        self.learn_step    = 0
        self.total_steps   = 0
        self.max_steps     = 5000   # beta 증가 기준 (메모리 크기와 동일)
        self.device        = torch.device("cpu")

        self.online_net = DuelingQNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.target_net = DuelingQNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=lr)

        # PER 버퍼 — 일반 deque 대신 우선순위 버퍼 사용
        self.memory     = PrioritizedReplayBuffer(
            capacity=5000, alpha=per_alpha, beta_start=per_beta_start
        )
        self.batch_size = 64

    def get_model_name(self):
        return "Rainbow"

    def _get_state_vec(self, step):
        vecs = [self.env.get_state_vector(step, i) for i in range(self.env.vocab_size)]
        return np.mean(vecs, axis=0).astype(np.float32)

    def _to_tensor(self, arr):
        return torch.FloatTensor(arr).to(self.device)

    def select_action(self, current_step):
        state_vec = self._get_state_vec(current_step)

        with torch.no_grad():
            q_np = self.online_net(self._to_tensor(state_vec)).squeeze().numpy()

        q_masked = self.apply_constraint_mask(q_np, current_step)

        if np.random.rand() < self.epsilon:
            valid  = [a for a in range(self.num_actions) if q_masked[a] != -np.inf]
            action = int(np.random.choice(valid))
        else:
            action = int(np.argmax(q_masked))

        if action == 0:
            chosen_ticker, ticker_idx = "CASH", 0
        else:
            ticker_idx    = action - 1
            chosen_ticker = self.env.tickers[ticker_idx]

        reward, raw_ret = self.env.get_step_reward(
            chosen_ticker, current_step, self.prev_action
        )
        self.prev_action  = action
        self.total_steps += 1
        self._record(reward)
        return chosen_ticker, True, reward, state_vec, action, ticker_idx, raw_ret

    def learn(self, state, action, reward, next_step):
        next_state = self._get_state_vec(next_step)

        # PER 버퍼에 저장
        self.memory.push(state, action, reward, next_state)

        if len(self.memory) < self.batch_size:
            return

        # 우선순위 기반 샘플링
        samples, indices, weights = self.memory.sample(
            self.batch_size, self.total_steps, self.max_steps
        )

        s_b  = self._to_tensor(np.array([b[0] for b in samples]))
        a_b  = torch.LongTensor([b[1] for b in samples]).to(self.device)
        r_b  = self._to_tensor(np.array([b[2] for b in samples]))
        ns_b = self._to_tensor(np.array([b[3] for b in samples]))
        w_b  = self._to_tensor(weights)   # IS 가중치

        # 현재 Q값
        q_vals = self.online_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

        # TD target: Double DQN 방식
        with torch.no_grad():
            best_actions = self.online_net(ns_b).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(ns_b).gather(1, best_actions).squeeze(1)
            td_target    = r_b + self.gamma * next_q

        # TD 에러 계산 → 우선순위 업데이트용
        td_errors = (td_target - q_vals).detach().cpu().numpy()

        # IS 가중치 적용한 손실 — 자주 뽑히는 샘플 손실을 낮춰서 편향 보정
        loss = (w_b * (q_vals - td_target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # PER 우선순위 업데이트
        self.memory.update_priorities(indices, td_errors)

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self, episode, total_episodes):
        self.cosine_decay(episode, total_episodes)

    def get_avg_q(self):
        with torch.no_grad():
            q_vals = self.online_net(self._to_tensor(
                np.zeros(self.STATE_DIM, dtype=np.float32)
            ))
        return float(q_vals.abs().mean().item())