import numpy as np
import random
from collections import deque
from agents.base import BaseAgent
from config import NUM_STATES


class TabularQAgent(BaseAgent):
    """
    Tabular Q-Learning + Experience Replay
    - 상태: 24개 이산 상태 (get_state)
    - 행동: CASH(0) + 종목 N개 → 총 N+1개
    - 학습: TD 업데이트 + 미니배치 Replay
    """

    def __init__(self, env, use_constraints=True, lr=0.005, gamma=0.85, eps=0.5):
        super().__init__(env, use_constraints, lr, gamma, eps)
        self.num_states  = NUM_STATES
        self.num_actions = 1 + env.vocab_size

        self.q_table = np.random.uniform(-0.01, 0.01, (self.num_states, self.num_actions))
        self.q_table[:, 0] = 0.02   # CASH 초기 bias

        self.memory     = deque(maxlen=5000)
        self.batch_size = 64

    def get_model_name(self):
        return "Tabular Q-Learning"

    def _get_market_state(self, step):
        counts = np.zeros(self.num_states)
        for i in range(self.env.vocab_size):
            counts[self.env.get_state(step, i)] += 1
        return int(np.argmax(counts))

    def select_action(self, current_step):
        state = self._get_market_state(current_step)

        # Q 마스킹: use_constraints=True면 위험 종목 선택 차단
        q_masked = self.apply_constraint_mask(self.q_table[state], current_step)

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
        return chosen_ticker, True, reward, state, action, ticker_idx, raw_ret

    def learn(self, state, action, reward, next_step):
        next_state = self._get_market_state(next_step)
        self.memory.append((state, action, reward, next_state))

        if len(self.memory) >= self.batch_size:
            for m_s, m_a, m_r, m_ns in random.sample(self.memory, self.batch_size):
                td_target = m_r + self.gamma * np.max(self.q_table[m_ns])
                self.q_table[m_s, m_a] += self.lr * (td_target - self.q_table[m_s, m_a])

    def decay_epsilon(self, episode, total_episodes):
        self.cosine_decay(episode, total_episodes)

    def get_avg_q(self):
        return float(np.mean(np.abs(self.q_table)))