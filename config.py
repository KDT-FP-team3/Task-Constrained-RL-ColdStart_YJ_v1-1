import os
from dotenv import load_dotenv

# == 환경 변수 및 보안 설정 ==
load_dotenv()
API_KEY = os.getenv("EXTERNAL_API_KEY", "default_key")

# == 금융 시뮬레이션 설정 ==
VOCAB_SIZE = 500       # 상장된 전체 종목 수
COLD_START_RATIO = 0.3 # 상장 초기 종목 또는 저유동성 종목 비율 (제약 대상)
TOTAL_EPISODES = 200   # 테스트 반복 횟수

# == Q-Learning 상태 공간 설정 ==
NUM_STATES = 12        # 2(SMA위치) × 3(RSI구간) × 2(변동성추세)
RSI_PERIOD = 14        # RSI 계산 기간
SMA_PERIOD = 20        # 이동평균 계산 기간
VOLATILITY_WINDOW = 20 # 변동성 계산 윈도우

# == 금융 보상 체계 ==
REWARD_VALID = 15      # 우량주/제약 통과 종목 매수 시 수익
REWARD_INVALID = -100  # 상장 폐지 위험/제약 위반 종목 매수 시 손실 (!! 주의)