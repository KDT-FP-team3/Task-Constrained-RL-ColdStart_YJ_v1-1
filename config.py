import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EXTERNAL_API_KEY", "default_key")

# == Q-Learning 상태 공간 ==
# 2(SMA위치) × 3(RSI구간) × 2(변동성추세) × 2(5일모멘텀) = 24
NUM_STATES = 24

# == 보조 지표 계산 윈도우 ==
RSI_PERIOD        = 14
SMA_PERIOD        = 20
VOLATILITY_WINDOW = 20

# == 금융 상수 ==
ANNUAL_RISK_FREE_RATE = 0.05
DAILY_RISK_FREE_RATE  = (ANNUAL_RISK_FREE_RATE / 252) * 100   # 일별 환산 (%)

# == 거래 비용 ==
TRANSACTION_COST = 0.10   # 신규 매수 시 (%)

# == 보상 패널티 계수 ==
# [원인5 수정] 0.1 → 0.03: 고수익 고변동성 종목 불이익 완화
VOL_PENALTY_COEFF = 0.03

# [원인2 수정] 제약 위반 소프트 패널티 (하드 필터 대체)
# 위반 종목도 거래 허용하되 보상에서 차감
CONSTRAINT_PENALTY = 0.20