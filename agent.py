# import numpy as np
# import config

# class StaticConstraintEngine:
#     """
#     논문의 STATIC 알고리즘을 단순화한 마스킹 엔진[cite: 33].
#     포인터 추적 대신 고정된 행렬 연산으로 유효 행동을 필터링합니다[cite: 255].
#     """
#     def __init__(self):
#         self.vocab_size = config.VOCAB_SIZE
#         # 콜드 스타트 아이템 수 계산 [cite: 104]
#         self.num_invalid = int(self.vocab_size * config.COLD_START_RATIO)
        
#         # Dense Mask 생성: 유효(1), 무효(0) [cite: 288, 600]
#         self.valid_mask = np.ones(self.vocab_size, dtype=bool)
#         if self.num_invalid > 0:
#             self.valid_mask[-self.num_invalid:] = False 

#     def apply_mask(self, logits):
#         """
#         논문의 Vectorized Node Transition Kernel(VNTK) 개념 적용 [cite: 293, 336]
#         위험 종목(Cold-start)의 확률을 -inf로 밀어내어 행렬 연산에서 배제 [cite: 250, 325]
#         """
#         # == 벡터화된 마스킹으로 0.033ms 수준의 초고속 처리 구현 [cite: 36, 388] ==
#         masked_logits = np.where(self.valid_mask, logits, -np.inf)
#         return masked_logits

# class RecommendationAgent:
#     def __init__(self, use_constraints=False):
#         self.use_constraints = use_constraints
#         self.engine = StaticConstraintEngine()
        
#     def select_action(self):
#         """
#         에이전트가 최적의 이상값을 산출하는 과정
#         """
#         # 1. 모델의 초기 예측값 (가우시안 분포 가정)
#         logits = np.random.randn(config.VOCAB_SIZE)
        
#         # 2. STATIC 제약 조건 적용 여부 결정
#         if self.use_constraints:
#             logits = self.engine.apply_mask(logits)
            
#         # 3. 탐욕적(Greedy) 행동 선택
#         chosen_action = int(np.argmax(logits))
        
#         # 4. 결과에 따른 보상 반환
#         is_valid = self.engine.valid_mask[chosen_action]
#         reward = config.REWARD_VALID if is_valid else config.REWARD_INVALID
        
#         return chosen_action, is_valid, reward



#----------agent.py 전면 개편 (S&P 500 환경 적용) ---------
# 단순 난수가 아닌, **"20일 이동평균선(SMA) 아래에 있는 하락 추세 종목을 제약 조건(STATIC Mask)으로 차단"**하는 실전 투자 로직을 적용합니다.

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

class SP500Environment:
    """ S&P 500 대표 종목 데이터를 관리하는 환경 """
    def __init__(self):
        # API 부하를 막기 위해 S&P 500 핵심 20개 티커로 축소 시뮬레이션
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY", "V",
                        "JPM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "CVX", "MRK"]
        self.vocab_size = len(self.tickers)
        self.data = self._download_data()

    @st.cache_data(ttl=3600)
    def _download_data(_self):
        # 최근 6개월 데이터 다운로드
        data = yf.download(_self.tickers, period="6mo", interval="1d")['Close']
        data.fillna(method='ffill', inplace=True)
        return data

class StaticConstraintEngine:
    def __init__(self, env, current_step):
        self.env = env
        self.vocab_size = env.vocab_size
        self.valid_mask = np.ones(self.vocab_size, dtype=bool)
        
        # == VNTK 마스킹 로직: 20일 이동평균선(SMA) 이탈 종목 차단 ==
        # 현재 스텝(날짜)을 기준으로 과거 20일 데이터를 분석
        if current_step >= 20:
            history = self.env.data.iloc[current_step-20 : current_step]
            sma_20 = history.mean()
            current_prices = self.env.data.iloc[current_step]
            
            # 현재가가 20일 이평선보다 낮으면 무효(False)로 마스킹
            for i, ticker in enumerate(self.env.tickers):
                if current_prices[ticker] < sma_20[ticker]:
                    self.valid_mask[i] = False
        else:
            # 20일치 데이터가 모이기 전(Cold-start)에는 모두 유효 처리
            pass

    def apply_mask(self, logits):
        # 무효화된 추세 하락 종목의 선택 확률을 -inf로 밀어냄 (VNTK 연산)
        return np.where(self.valid_mask, logits, -np.inf)

class RecommendationAgent:
    def __init__(self, env, use_constraints=False):
        self.env = env
        self.use_constraints = use_constraints
        
    def select_action(self, current_step):
        # 1. 모델 예측값 (여기서는 랜덤 정책을 기반으로 함)
        logits = np.random.randn(self.env.vocab_size)
        
        # 2. STATIC 제약 엔진 구동
        engine = StaticConstraintEngine(self.env, current_step)
        
        if self.use_constraints:
            logits = engine.apply_mask(logits)
            
        # 3. 행동 선택
        chosen_action = int(np.argmax(logits))
        
        # 4. 보상 계산 (다음 날의 실제 주가 수익률 %)
        # 데이터의 끝에 도달하면 보상을 0으로 처리
        if current_step + 1 < len(self.env.data):
            current_price = self.env.data.iloc[current_step, chosen_action]
            next_price = self.env.data.iloc[current_step + 1, chosen_action]
            reward = ((next_price - current_price) / current_price) * 100 
        else:
            reward = 0
            
        is_valid = engine.valid_mask[chosen_action]
        chosen_ticker = self.env.tickers[chosen_action]
        
        return chosen_ticker, is_valid, reward