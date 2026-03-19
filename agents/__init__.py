from agents.tabular_q   import TabularQAgent
from agents.dqn         import DQNAgent
from agents.double_dqn  import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.ppo         import PPOAgent
from agents.rainbow     import RainbowAgent

AGENT_REGISTRY = {
    "Tabular Q-Learning": TabularQAgent,
    "DQN":                DQNAgent,
    "Double DQN":         DoubleDQNAgent,
    "Dueling DQN":        DuelingDQNAgent,
    "PPO":                PPOAgent,
    "Rainbow":            RainbowAgent,
}

AGENT_PARAMS = {
    "Tabular Q-Learning": {
        "lr":    "Q값 업데이트 속도 (0.001~0.01 권장)",
        "gamma": "미래 보상 중요도",
        "eps":   "초기 탐색 확률 (Cosine Annealing 감소)",
    },
    "DQN": {
        "lr":            "신경망 학습률 (0.0001~0.001 권장)",
        "gamma":         "미래 보상 중요도",
        "eps":           "초기 탐색 확률 (Cosine Annealing 감소)",
        "target_update": "Target Network 동기화 주기 (step 단위)",
    },
    "Double DQN": {
        "lr":            "신경망 학습률 (0.0001~0.001 권장)",
        "gamma":         "미래 보상 중요도",
        "eps":           "초기 탐색 확률 (Cosine Annealing 감소)",
        "target_update": "Target Network 동기화 주기 — DQN보다 낮게 설정 권장",
    },
    "Dueling DQN": {
        "lr":            "신경망 학습률 (0.0001~0.001 권장)",
        "gamma":         "미래 보상 중요도",
        "eps":           "초기 탐색 확률 (Cosine Annealing 감소)",
        "target_update": "Target Network 동기화 주기",
    },
    "PPO": {
        "lr":    "신경망 학습률 (0.0001~0.001 권장, 기본 0.0003)",
        "gamma": "미래 보상 중요도",
        "eps":   "초기 탐색 확률 (entropy가 탐색 담당, 참고용)",
    },
    "Rainbow": {
        "lr":            "신경망 학습률 (0.0001~0.001 권장)",
        "gamma":         "미래 보상 중요도",
        "eps":           "초기 탐색 확률 (Cosine Annealing 감소)",
        "target_update": "Target Network 동기화 주기",
    },
}

# DQN 계열 (target_update 파라미터 필요한 모델 목록)
DQN_FAMILY = {"DQN", "Double DQN", "Dueling DQN", "Rainbow"}

# PPO 전용 파라미터 필요한 모델 목록
PPO_FAMILY  = {"PPO"}