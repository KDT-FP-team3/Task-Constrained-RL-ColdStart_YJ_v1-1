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


class DuelingQNetwork(nn.Module):
    """
    Dueling Network 구조
    ────────────────────────────────────────────────────────
    일반 DQN:
        입력 → [공유 레이어] → Q(s,a) 직접 출력

    Dueling DQN:
        입력 → [공유 레이어] → ┬→ [Value 스트림]     → V(s)      : 스칼라 1개
                               └→ [Advantage 스트림] → A(s,a)    : 행동 수만큼
                                                    ↓
                               Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
                               (mean 차감 → Advantage 값 안정화)
    ────────────────────────────────────────────────────────
    """
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()

        # 공유 레이어 — 두 스트림이 공통으로 사용
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )

        # Value 스트림 — 상태 가치 V(s) 출력 (스칼라)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Advantage 스트림 — 행동별 상대 우위 A(s,a) 출력
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        # 1D 입력(단일 샘플)이면 배치 차원 추가
        if x.dim() == 1:
            x = x.unsqueeze(0)
        shared = self.shared(x)
        v      = self.value_stream(shared)        # (batch, 1)
        a      = self.advantage_stream(shared)    # (batch, num_actions)

        # Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
        # mean 차감으로 Advantage의 기준점을 0으로 정규화
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DuelingDQNAgent(BaseAgent):
    """
    Dueling Deep Q-Network

    DQN / Double DQN 대비 개선점:
    ──────────────────────────────
    - Q값을 V(상태 가치) + A(행동 우위)로 분리 학습
    - CASH처럼 행동 간 차이가 작은 날에도 V(s)가 독립적으로 업데이트
      → 더 많은 샘플에서 학습 신호 확보
    - 주식처럼 "시장 전체가 나쁜 날" vs "특정 종목이 나쁜 날"을
      명확히 구분 가능

    TD target 계산은 Double DQN 방식 사용
    (행동 선택 Online / Q값 평가 Target — 과대추정 억제)
    """

    STATE_DIM = 5

    def __init__(self, env, use_constraints=True, lr=0.001, gamma=0.85, eps=0.5,
                 hidden=128, target_update=20):
        super().__init__(env, use_constraints, lr, gamma, eps)

        if not TORCH_AVAILABLE:
            raise ImportError("Dueling DQN은 PyTorch가 필요합니다: pip install torch")

        self.num_actions   = 1 + env.vocab_size
        self.target_update = target_update
        self.learn_step    = 0
        self.device        = torch.device("cpu")

        # DuelingQNetwork 사용 — 내부적으로 V/A 스트림 분리
        self.online_net = DuelingQNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.target_net = DuelingQNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()
        self.memory     = deque(maxlen=5000)
        self.batch_size = 64

    def get_model_name(self):
        return "Dueling DQN"

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

        # TD target: Double DQN 방식 (Online 선택 + Target 평가)
        with torch.no_grad():
            best_actions = self.online_net(ns_b).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(ns_b).gather(1, best_actions).squeeze(1)
            td_target    = r_b + self.gamma * next_q

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