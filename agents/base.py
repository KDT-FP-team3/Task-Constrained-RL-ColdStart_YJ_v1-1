from abc import ABC, abstractmethod
import numpy as np
import os
import json
from datetime import datetime
from config import DAILY_RISK_FREE_RATE


class BaseAgent(ABC):
    """
    모든 RL 에이전트가 반드시 구현해야 하는 공통 인터페이스
    - select_action / learn / decay_epsilon 은 반드시 구현
    - get_sharpe / get_mdd / get_win_rate 는 공통 로직으로 제공
    - save / load 는 모델별로 오버라이드
    """

    def __init__(self, env, use_constraints=True, lr=0.005, gamma=0.85, eps=0.5):
        self.env             = env
        self.use_constraints = use_constraints
        self.lr              = lr
        self.gamma           = gamma
        self.epsilon         = eps
        self.initial_epsilon = eps
        self.prev_action     = 0
        self.episode_rewards = []
        self.win_count       = 0
        self.total_count     = 0

    # ── 반드시 구현해야 하는 메서드 ────────────────────────────────────────
    @abstractmethod
    def select_action(self, current_step):
        """반환: (chosen_ticker, done, reward, state, action, ticker_idx, raw_ret)"""

    @abstractmethod
    def learn(self, state, action, reward, next_step):
        """경험으로 Q값 / Policy 업데이트"""

    @abstractmethod
    def decay_epsilon(self, episode, total_episodes):
        """탐색률 감소"""

    @abstractmethod
    def get_model_name(self) -> str:
        """UI 표시용 모델 이름"""

    # ── 저장 / 불러오기 ────────────────────────────────────────────────────
    def _meta(self):
        """저장 시 함께 기록할 메타 정보"""
        return {
            "model_name":     self.get_model_name(),
            "use_constraints":self.use_constraints,
            "lr":             self.lr,
            "gamma":          self.gamma,
            "epsilon":        self.epsilon,
            "initial_epsilon":self.initial_epsilon,
            "win_count":      self.win_count,
            "total_count":    self.total_count,
            "tickers":        list(self.env.tickers),
            "saved_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save(self, path: str):
        """
        모델 저장 — 서브클래스에서 오버라이드
        기본 구현: 메타 정보만 JSON으로 저장
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        meta_path = path.replace(".pt", ".json").replace(".npy", ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """모델 불러오기 — 서브클래스에서 오버라이드"""
        pass

    @staticmethod
    def load_meta(path: str) -> dict:
        """저장된 메타 정보 읽기 (에이전트 생성 전에 확인용)"""
        meta_path = path.replace(".pt", ".json").replace(".npy", ".json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── 공통 제공 메서드 ────────────────────────────────────────────────────
    def _record(self, reward):
        self.episode_rewards.append(reward)
        self.total_count += 1
        if reward > 0:
            self.win_count += 1

    def get_win_rate(self):
        return (self.win_count / self.total_count * 100) if self.total_count > 0 else 0.0

    def get_sharpe(self):
        if len(self.episode_rewards) < 2: return 0.0
        excess = np.array(self.episode_rewards) - DAILY_RISK_FREE_RATE
        std    = np.std(excess)
        if std < 1e-10: return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    def get_mdd(self, cumulative_returns):
        if len(cumulative_returns) < 2: return 0.0
        arr  = np.array(cumulative_returns)
        peak = np.maximum.accumulate(arr)
        return float(np.min(arr - peak))

    def cosine_decay(self, episode, total_episodes, min_eps=0.01):
        cos_val      = 0.5 * (1 + np.cos(np.pi * episode / total_episodes))
        self.epsilon = min_eps + (self.initial_epsilon - min_eps) * cos_val

    def apply_constraint_mask(self, q_values_np, step):
        """
        제약 위반 종목의 Q값을 -inf로 마스킹
        → 행동 선택 단계에서만 제약 적용 (학습 보상은 건드리지 않음)
        """
        if not self.use_constraints:
            return q_values_np
        mask   = self.env.get_constraint_mask(step)
        masked = q_values_np.copy()
        for ticker_idx, violated in enumerate(mask):
            if violated:
                masked[ticker_idx + 1] = -np.inf
        return masked