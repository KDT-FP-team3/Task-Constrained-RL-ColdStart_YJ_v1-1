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
    """
    상태 벡터(5차원) → Q값 벡터(num_actions차원) 매핑 신경망
    구조: 5 → 128 → 128 → num_actions
    """
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


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN)
    - 상태: 연속 벡터 5차원 (get_state_vector) — 이산화 불필요
    - 행동: CASH(0) + 종목 N개
    - 학습: Experience Replay + Target Network (안정적 학습)

    Tabular Q 대비 개선점:
    1. 상태를 숫자 그대로 입력 → 더 세밀한 시장 패턴 학습
    2. Target Network로 학습 안정성 확보
    3. 종목 수가 늘어도 테이블 크기 폭발 없음
    """

    STATE_DIM = 5   # get_state_vector 출력 차원

    def __init__(self, env, use_constraints=True, lr=0.001, gamma=0.85, eps=0.5,
                 hidden=128, target_update=20):
        super().__init__(env, use_constraints, lr, gamma, eps)

        if not TORCH_AVAILABLE:
            raise ImportError("DQN은 PyTorch가 필요합니다: pip install torch")

        self.num_actions   = 1 + env.vocab_size
        self.target_update = target_update   # Target Network 동기화 주기
        self.learn_step    = 0

        self.device = torch.device("cpu")

        # Online Network: 매 step 업데이트
        self.online_net = QNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        # Target Network: target_update 주기마다 online 가중치 복사
        self.target_net = QNetwork(self.STATE_DIM, self.num_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()

        self.memory     = deque(maxlen=5000)
        self.batch_size = 64

    def get_model_name(self):
        return "DQN"

    def _get_state_vec(self, step):
        """전 종목 상태 벡터의 평균 → 시장 대표 상태"""
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
            chosen_ticker = "CASH"
            ticker_idx    = 0
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

        batch      = random.sample(self.memory, self.batch_size)
        s_b        = self._to_tensor(np.array([b[0] for b in batch]))
        a_b        = torch.LongTensor([b[1] for b in batch]).to(self.device)
        r_b        = self._to_tensor(np.array([b[2] for b in batch]))
        ns_b       = self._to_tensor(np.array([b[3] for b in batch]))

        # Online Net으로 현재 Q값
        q_vals     = self.online_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

        # Target Net으로 다음 Q값 (그래디언트 차단)
        with torch.no_grad():
            next_q  = self.target_net(ns_b).max(1)[0]
            td_target = r_b + self.gamma * next_q

        loss = self.loss_fn(q_vals, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target Network 주기적 동기화
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self, episode, total_episodes):
        self.cosine_decay(episode, total_episodes)

    def get_avg_q(self):
        """현재 Online Network의 평균 Q값 (모니터링용)"""
        with torch.no_grad():
            sample_state = self._to_tensor(np.zeros(self.STATE_DIM, dtype=np.float32))
            q_vals = self.online_net(sample_state)
        return float(q_vals.abs().mean().item())