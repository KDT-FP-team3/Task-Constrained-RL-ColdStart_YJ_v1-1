import numpy as np
import random
from collections import deque
from agents.base import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class QNetwork(nn.Module):
    """DQN과 동일한 신경망 구조 — 5 → 128 → 128 → num_actions"""
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class DoubleDQNAgent(BaseAgent):
    """
    Double Deep Q-Network (Double DQN)

    DQN의 고질적 문제: Q값 과대추정(Overestimation)
    ────────────────────────────────────────────────
    DQN에서 TD target 계산:
        next_q = target_net(s').max()
        → 행동 선택(argmax)과 Q값 평가를 Target 혼자 담당
        → 노이즈가 낀 상태에서 운 좋게 높은 Q값을 가진 행동을 계속 선택
        → Q값이 실제보다 부풀려짐 → 잘못된 학습 방향

    Double DQN 해결책: 역할 분리
    ────────────────────────────
        best_a = online_net(s').argmax()   # Online이 행동 선택
        next_q = target_net(s')[best_a]    # Target이 Q값 평가
        → 두 네트워크가 서로 견제 → 과대추정 억제

    코드상 DQN과의 차이: learn() 내 TD target 계산 2줄
    """

    STATE_DIM = 5

    def __init__(self, env, use_constraints=True, lr=0.001, gamma=0.85, eps=0.5,
                 hidden=128, target_update=20):
        super().__init__(env, use_constraints, lr, gamma, eps)

        if not TORCH_AVAILABLE:
            raise ImportError("Double DQN은 PyTorch가 필요합니다: pip install torch")

        self.num_actions   = 1 + env.vocab_size
        self.target_update = target_update
        self.learn_step    = 0
        self.device        = torch.device("cpu")

        self.online_net = QNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.target_net = QNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()
        self.memory     = deque(maxlen=5000)
        self.batch_size = 64

    def get_model_name(self):
        return "Double DQN"

    def _get_state_vec(self, step):
        vecs = [self.env.get_state_vector(step, i) for i in range(self.env.vocab_size)]
        return np.mean(vecs, axis=0).astype(np.float32)

    def _to_tensor(self, arr):
        return torch.FloatTensor(arr).to(self.device)

    def select_action(self, current_step):
        state_vec = self._get_state_vec(current_step)

        with torch.no_grad():
            q_np = self.online_net(self._to_tensor(state_vec)).squeeze().numpy()

        # Q 마스킹: use_constraints=True면 위험 종목 선택 차단
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
        self.prev_action = action
        self._record(reward)
        return chosen_ticker, True, reward, state_vec, action, ticker_idx, raw_ret

    def learn(self, state, action, reward, next_step):
        next_state = self._get_state_vec(next_step)
        self.memory.append((state, action, reward, next_state))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s_b   = self._to_tensor(np.array([b[0] for b in batch]))
        a_b   = torch.LongTensor([b[1] for b in batch]).to(self.device)
        r_b   = self._to_tensor(np.array([b[2] for b in batch]))
        ns_b  = self._to_tensor(np.array([b[3] for b in batch]))

        # 현재 Q값
        q_vals = self.online_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

        # ── Double DQN 핵심: DQN과 다른 부분 ──────────────────────────
        with torch.no_grad():
            # Step 1: Online Network로 다음 상태에서 최선의 행동 선택
            best_actions = self.online_net(ns_b).argmax(dim=1, keepdim=True)
            # Step 2: Target Network로 그 행동의 Q값만 평가
            next_q       = self.target_net(ns_b).gather(1, best_actions).squeeze(1)
            td_target    = r_b + self.gamma * next_q
        # ────────────────────────────────────────────────────────────────

        loss = self.loss_fn(q_vals, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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