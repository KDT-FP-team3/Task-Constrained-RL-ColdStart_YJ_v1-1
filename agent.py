import numpy as np
import config

class StaticConstraintEngine:
    """
    논문의 STATIC 알고리즘을 단순화한 마스킹 엔진[cite: 33].
    포인터 추적 대신 고정된 행렬 연산으로 유효 행동을 필터링합니다[cite: 255].
    """
    def __init__(self):
        self.vocab_size = config.VOCAB_SIZE
        # 콜드 스타트 아이템 수 계산 [cite: 104]
        self.num_invalid = int(self.vocab_size * config.COLD_START_RATIO)
        
        # Dense Mask 생성: 유효(1), 무효(0) [cite: 288, 600]
        self.valid_mask = np.ones(self.vocab_size, dtype=bool)
        if self.num_invalid > 0:
            self.valid_mask[-self.num_invalid:] = False 

    def apply_mask(self, logits):
        """
        논문의 Vectorized Node Transition Kernel(VNTK) 개념 적용 [cite: 293, 336]
        위험 종목(Cold-start)의 확률을 -inf로 밀어내어 행렬 연산에서 배제 [cite: 250, 325]
        """
        # == 벡터화된 마스킹으로 0.033ms 수준의 초고속 처리 구현 [cite: 36, 388] ==
        masked_logits = np.where(self.valid_mask, logits, -np.inf)
        return masked_logits

class RecommendationAgent:
    def __init__(self, use_constraints=False):
        self.use_constraints = use_constraints
        self.engine = StaticConstraintEngine()
        
    def select_action(self):
        """
        에이전트가 최적의 이상값을 산출하는 과정
        """
        # 1. 모델의 초기 예측값 (가우시안 분포 가정)
        logits = np.random.randn(config.VOCAB_SIZE)
        
        # 2. STATIC 제약 조건 적용 여부 결정
        if self.use_constraints:
            logits = self.engine.apply_mask(logits)
            
        # 3. 탐욕적(Greedy) 행동 선택
        chosen_action = int(np.argmax(logits))
        
        # 4. 결과에 따른 보상 반환
        is_valid = self.engine.valid_mask[chosen_action]
        reward = config.REWARD_VALID if is_valid else config.REWARD_INVALID
        
        return chosen_action, is_valid, reward