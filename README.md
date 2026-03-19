# 🏛️ Personalized-RL-Quant: Adaptive Portfolio Management

> **KDT Team Study — 팀 Chainers 🫡**  
> S&P 500 대표 종목을 대상으로 **제약 조건 강화학습(Constrained Reinforcement Learning)** 을 적용한 포트폴리오 관리 시뮬레이션 시스템

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**🌐 웹 데모**: [Personalized-Quant-RL (Streamlit Cloud)](https://test-constrained-rl-coldstart-cgwsjhrq57w4jm48fqzbmm.streamlit.app/)

---

## 📌 프로젝트 개요

Tabular Q-Learning의 **행동 공간을 종목 단위로 확장**하여 종목 선택과 매수/현금 보유 결정을 하나의 Q-테이블로 통합합니다. 시장 국면(Bull/Bear)에 따른 동적 소프트 제약과 리스크 조정 보상 설계를 통해 **어떤 파라미터 세팅에서도 Vanilla RL 대비 안정적인 초과 수익**을 목표로 합니다.

### 핵심 특징

- **멀티 모델 지원**: Tabular Q-Learning, DQN, Double DQN, PPO 등 다양한 RL 알고리즘 비교
- **Adaptive Soft Constraint**: Bull/Bear 시장 국면에 따라 제약 기준 동적 조정
- **Model Race**: 동일 조건에서 전체 모델 실시간 성능 비교
- **HPO (Hyperparameter Optimization)**: Optuna Bayesian 최적화로 최적 파라미터 자동 탐색
- **실시간 대시보드**: Streamlit + Plotly 기반 인터랙티브 시각화

---

## 📁 프로젝트 구조

```
.
├── .devcontainer/        # 개발 컨테이너 설정 (GitHub Codespaces)
├── agents/               # RL 에이전트 모듈 모음
│   ├── base.py           # BaseAgent 추상 클래스
│   └── ...               # Tabular Q, DQN, Double DQN, PPO 등
├── app.py                # Streamlit UI — 실행 진입점 (6탭 구조)
├── environment.py        # SP500Environment — 데이터 및 기술적 지표 관리
├── config.py             # 하이퍼파라미터 및 금융 상수 중앙 관리
├── hpo.py                # Optuna 기반 HPO (하이퍼파라미터 최적화)
├── requirements.txt      # 파이썬 패키지 의존성
└── README.md
```

---

## 🧠 알고리즘 설계

### 상태 공간 (State Space)

종목별 기술적 지표 4개를 이산화하여 **24개의 이산 상태**로 표현합니다.

| 지표                   | 이산화 기준                             | 가짓수 |
| ---------------------- | --------------------------------------- | ------ |
| 가격 vs SMA20          | 위(1) / 아래(0)                         | 2      |
| RSI(14)                | 과매도 <30(0) / 중립(1) / 과매수 >70(2) | 3      |
| 변동성 추세 (5일 diff) | 상승(1) / 하락(0)                       | 2      |
| 5일 모멘텀             | 상승(1) / 하락(0)                       | 2      |

> 2 × 3 × 2 × 2 = **24 상태**  
> 시장 대표 상태: 전체 종목 상태 분포의 **최빈값(argmax)** 사용

### 행동 공간 (Action Space)

기존 BUY/CASH 2개 구조에서 **종목 단위 21개**로 확장했습니다.

```
action = 0       → CASH 보유
action = 1 ~ 20  → 종목 직접 매수 (AAPL=1, MSFT=2, ... MRK=20)
```

| 항목          | 기존 구조     | 개선 구조             |
| ------------- | ------------- | --------------------- |
| Q 행동 수     | 2 (BUY/CASH)  | 21 (CASH + 종목 20개) |
| 종목 선택     | 모멘텀 고정   | Q가 직접 학습         |
| Q 테이블 크기 | 24 × 2 = 48칸 | 24 × 21 = 504칸       |

### 보상 함수 (Reward Function)

```
CASH 선택:  reward = DAILY_RISK_FREE_RATE (연 5% / 252일 ≈ 0.0198%)
BUY 선택:   reward = raw_ret
                   - TRANSACTION_COST      (신규 진입 시 0.10%)
                   - vol × 100 × 0.03     (변동성 패널티)
                   - CONSTRAINT_PENALTY   (제약 위반 시 0.20%, DCA만 적용)
```

### Adaptive Soft Constraint (DCA 전용)

시장 국면(Bull / Bear / Neutral)에 따라 제약 기준을 동적으로 조정합니다.

| 국면 판단 | SPY 종가 vs SPY SMA(500) |
| --------- | ------------------------ |
| Bull      | SPY > SMA500             |
| Bear      | SPY < SMA500             |
| Neutral   | 데이터 500일 미만        |

| 제약 항목     | Bull 기준    | Bear 기준    |
| ------------- | ------------ | ------------ |
| RSI 상한      | 85           | 75           |
| 변동성 배수   | 중앙값 × 3.0 | 중앙값 × 1.5 |
| SMA 하향 필터 | 미적용       | 적용         |

하드 필터(후보 제거) 대신 **소프트 패널티(위반 종목도 허용, 보상 차감)** 방식으로 강한 모멘텀 종목이 제약에 완전히 차단되는 문제를 해결했습니다.

### Q-Learning 학습 설정

| 항목             | 기본값 | 비고                               |
| ---------------- | ------ | ---------------------------------- |
| 학습률 (α)       | 0.005  | 사이드바 조정 가능                 |
| 할인율 (γ)       | 0.85   | 사이드바 조정 가능                 |
| 초기 탐색률 (ε)  | 0.5    | Cosine Annealing으로 0.01까지 감소 |
| 메모리 크기      | 5,000  | Experience Replay 버퍼             |
| 배치 크기        | 64     | 미니배치 학습                      |
| CASH Q 초기 bias | 0.02   | 학습 초반 과도한 매수 방지         |

**ε 감소 전략 — Cosine Annealing:**

```
ε(t) = ε_min + (ε_init - ε_min) × 0.5 × (1 + cos(π × t / T))
```

---

## 🖥️ UI 구성 (6탭)

| 탭               | 설명                                                         |
| ---------------- | ------------------------------------------------------------ |
| 📈 Live Monitor  | 실시간 누적 수익률 차트 + 메트릭 카드 + 학습 모니터          |
| 📊 Analysis      | Sharpe/MDD 카드, 결정 로그, 포트폴리오 배분, 백테스트 리포트 |
| 🏆 Trial History | 다회차 수익률·Sharpe 분포 박스플롯 + 요약 테이블             |
| 🔬 HPO           | Optuna Bayesian 최적화 — 단일/전체 모델 파라미터 탐색        |
| ⚔️ Model Race    | 모든 모델 동시 학습 후 실시간 성능 비교                      |
| 🎯 Recommender   | Race + HPO 결과 종합 기반 최적 모델 추천                     |

---

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install streamlit pandas numpy plotly yfinance python-dotenv
# HPO 사용 시 (선택)
pip install optuna
```

### 2. 로컬 실행

```bash
streamlit run app.py
```

### 3. 환경 변수 (선택)

프로젝트 루트에 `.env` 파일 생성 후 필요한 API 키를 추가합니다.

```env
EXTERNAL_API_KEY=your_key_here
```

---

## ⚙️ 파라미터 가이드

| 파라미터            | 권장값 | 설명                                                      |
| ------------------- | ------ | --------------------------------------------------------- |
| Learning Rate (α)   | 0.005  | 너무 크면 Q값 발산, 너무 작으면 수렴 느림                 |
| Discount Factor (γ) | 0.85   | 높을수록 장기 수익 지향                                   |
| Exploration (ε)     | 0.5    | Cosine Annealing으로 0.01까지 자동 감소                   |
| Market Pre-Train    | 2 ~ 3  | 본 학습 전 과거 데이터 반복 횟수. 높을수록 초반 학습 안정 |
| Execution Speed     | 0.02s  | 0으로 설정 시 최대 속도                                   |

---

## 📊 성과 지표 해석

| 지표              | 의미                              | 해석 기준                                        |
| ----------------- | --------------------------------- | ------------------------------------------------ |
| Cumulative Return | 누적 수익률 (raw_ret 기준)        | 높을수록 좋음                                    |
| Win Rate          | 양수 보상 비율                    | 50% 이상이면 수익 구간이 손실보다 많음           |
| Q-Score           | Q테이블 평균 절댓값               | 학습 진행에 따라 점진적으로 증가                 |
| Sharpe Ratio      | (초과수익 평균 / 표준편차) × √252 | 1.0 이상이면 우수, 음수이면 무위험 자산보다 못함 |
| MDD               | 고점 대비 최대 낙폭               | 0에 가까울수록 방어적 전략                       |

---

## 🔍 설계 결정 및 트레이드오프

**Q 행동공간 확장의 부작용**  
행동이 21개로 늘어나면 Q테이블이 504칸으로 커지고, 500번의 에피소드로는 모든 (상태, 행동) 쌍을 충분히 탐색하기 어렵습니다. ε 초기값을 0.5로 높이고 메모리/배치를 키웠으며, pretrain을 3회 이상 설정하면 학습 안정성이 크게 향상됩니다.

**소프트 제약의 의도**  
하드 필터는 RSI가 높은 강한 상승 종목을 완전히 차단해 오히려 알파를 손실시킵니다. 소프트 패널티는 위반 종목도 선택 가능하게 두되 보상에 불이익을 줌으로써 에이전트가 스스로 위험 종목을 회피하도록 유도합니다.

**시장 대표 상태의 한계**  
전 종목 상태의 최빈값을 사용하는 방식은 개별 종목의 고유한 기술적 상태를 무시하는 단순화입니다. 향후 종목별 독립 Q-테이블 또는 DQN으로 확장할 여지가 있습니다.

---

## 🔭 향후 개선 방향

- **종목별 독립 Q-테이블**: 시장 대표 상태 대신 각 종목이 자신의 상태를 기반으로 독립 학습
- **DQN 전환**: 상태·행동 공간이 커질 경우 신경망 기반 함수 근사로 확장
- **멀티 포지션**: 현재 1종목 집중 투자 → 포트폴리오 비중 분산 지원
- **실거래 연동**: 브로커 API 연결을 통한 실시간 신호 생성

---

## 👥 팀 정보

**KDT Team Study — 팀 Chainers**  
본 프로젝트는 KDT(Korea Digital Training) 파이널 프로젝트로 진행되었습니다.
