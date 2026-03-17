# Personalized-RL-Quant: Adaptive Portfolio Management

## KDT Team Study — 팀 Chainers 🫡

S&P 500 대표 20개 종목을 대상으로 **제약 조건 강화학습(Constrained Reinforcement Learning)** 을 적용한 포트폴리오 관리 시뮬레이션 시스템입니다.

Tabular Q-Learning의 **행동 공간을 종목 단위로 확장**하여 종목 선택과 매수/현금 보유 결정을 Q 하나로 통합하고, 시장 국면(Bull/Bear)에 따른 동적 소프트 제약과 리스크 조정 보상 설계를 통해 **어떤 파라미터 세팅에서도 Vanilla RL 대비 안정적인 초과 수익**을 목표로 합니다.

- **웹 데모**: [Personalized-Quant-RL](https://test-constrained-rl-coldstart-cgwsjhrq57w4jm48fqzbmm.streamlit.app/)
- **핵심 알고리즘**: Tabular Q-Learning + Experience Replay + Adaptive Soft Constraint
- **프레임워크**: Streamlit + Plotly + yfinance

---

## 1. 프로젝트 구조

```
.
├── app.py        # Streamlit UI — 실행 진입점
├── agent.py      # 환경(SP500Environment) + 제약 엔진 + Q-에이전트
├── config.py     # 하이퍼파라미터 및 금융 상수 중앙 관리
└── README.md
```

---

## 2. 알고리즘 설계

### 2-1. 상태 공간 (State Space)

종목별 기술적 지표 4개를 이산화하여 **24개의 이산 상태**로 표현합니다.

| 지표                   | 이산화 기준                             | 가짓수 |
| ---------------------- | --------------------------------------- | ------ |
| 가격 vs SMA20          | 위(1) / 아래(0)                         | 2      |
| RSI(14)                | 과매도 <30(0) / 중립(1) / 과매수 >70(2) | 3      |
| 변동성 추세 (5일 diff) | 상승(1) / 하락(0)                       | 2      |
| 5일 모멘텀             | 상승(1) / 하락(0)                       | 2      |

> 2 × 3 × 2 × 2 = **24 상태**

실제 에이전트가 사용하는 시장 대표 상태는 전체 20개 종목의 상태 분포에서 **최빈값(argmax)** 을 사용합니다.

### 2-2. 행동 공간 (Action Space) — 핵심 설계

기존 구조(BUY/CASH 2개)에서 **종목 단위 확장(21개)** 으로 변경했습니다.

```
action = 0          → CASH 보유
action = 1 ~ 20     → 각 종목 직접 매수 (AAPL=1, MSFT=2, ... MRK=20)
```

이를 통해 종목 선택과 매수/현금 결정이 Q-테이블 하나로 완전히 통합됩니다.

|               | 기존 구조     | 개선 구조             |
| ------------- | ------------- | --------------------- |
| Q 행동 수     | 2 (BUY/CASH)  | 21 (CASH + 종목 20개) |
| 종목 선택     | 모멘텀 고정   | Q가 직접 학습         |
| Q 테이블 크기 | 24 × 2 = 48칸 | 24 × 21 = 504칸       |

### 2-3. 보상 함수 (Reward Function)

```
CASH 선택:   reward = DAILY_RISK_FREE_RATE (연 5% / 252일 ≈ 0.0198%)

BUY 선택:    reward = raw_ret
                    - TRANSACTION_COST (신규 진입 시 0.10%)
                    - vol × 100 × VOL_PENALTY_COEFF (변동성 패널티 0.03)
                    - CONSTRAINT_PENALTY (제약 위반 시 추가 0.20%, DCA만 적용)
```

CASH에 무위험수익률을 부여하여 에이전트가 BUY와 CASH를 **공정하게 비교**하도록 설계했습니다.

### 2-4. Adaptive Soft Constraint (DCA 전용)

시장 국면(Bull / Bear / Neutral)에 따라 제약 기준을 동적으로 조정합니다.

| 국면 판단 기준 | SPY 종가 vs SPY SMA(500) |
| -------------- | ------------------------ |
| Bull           | SPY > SMA500             |
| Bear           | SPY < SMA500             |
| Neutral        | 데이터 500일 미만        |

| 제약 항목     | Bull 기준    | Bear 기준    |
| ------------- | ------------ | ------------ |
| RSI 상한      | 85           | 75           |
| 변동성 배수   | 중앙값 × 3.0 | 중앙값 × 1.5 |
| SMA 하향 필터 | 미적용       | 적용         |

기존 하드 필터(후보 제거)에서 **소프트 패널티(위반 종목도 허용, 보상 차감)** 방식으로 전환하여 강한 모멘텀 종목이 제약에 의해 완전히 차단되는 문제를 해결했습니다.

### 2-5. Q-Learning 학습 설정

| 항목             | 값    | 비고                               |
| ---------------- | ----- | ---------------------------------- |
| 학습률 (α)       | 0.005 | 사이드바 조정 가능                 |
| 할인율 (γ)       | 0.85  | 사이드바 조정 가능                 |
| 초기 탐색률 (ε)  | 0.5   | Cosine Annealing으로 0.01까지 감소 |
| 메모리 크기      | 5,000 | Experience Replay 버퍼             |
| 배치 크기        | 64    | 미니배치 학습                      |
| CASH Q 초기 bias | 0.02  | 학습 초반 과도한 매수 방지         |

**ε 감소 전략 — Cosine Annealing:**

```
ε(t) = ε_min + (ε_init - ε_min) × 0.5 × (1 + cos(π × t / T))
```

선형 감소 대비 초반 탐색을 충분히 유지하면서 후반에 빠르게 수렴합니다.

---

## 3. 파일별 상세 설명

### `config.py`

모든 하이퍼파라미터와 금융 상수를 중앙 관리합니다. `agent.py`가 이 파일에서 직접 import하여 사용하므로, 수치 변경은 이 파일에서만 하면 됩니다.

```python
NUM_STATES            = 24      # 이산 상태 수 (2×3×2×2)
RSI_PERIOD            = 14      # RSI 계산 기간
SMA_PERIOD            = 20      # SMA 계산 기간
VOLATILITY_WINDOW     = 20      # 변동성 롤링 윈도우
ANNUAL_RISK_FREE_RATE = 0.05    # 연 무위험수익률
DAILY_RISK_FREE_RATE  = ...     # 일별 환산 (자동 계산)
TRANSACTION_COST      = 0.10    # 신규 매수 거래비용 (%)
VOL_PENALTY_COEFF     = 0.03    # 변동성 패널티 계수
CONSTRAINT_PENALTY    = 0.20    # 소프트 제약 위반 패널티 (%)
```

### `agent.py`

세 개의 클래스로 구성됩니다.

**`SP500Environment`** — 데이터 및 지표 관리

- Yahoo Finance에서 5년치 일봉 데이터를 다운로드 (`@st.cache_data` 1시간 캐시)
- SMA20, RSI(14), 변동성(20일), 5일 모멘텀을 사전 계산(precompute)
- `get_state(step, ticker_idx)`: 특정 시점·종목의 이산화 상태 반환
- `get_market_regime(step)`: Bull / Bear / Neutral 판단
- `get_constraint_mask(step)`: 제약 위반 종목 boolean 마스크 반환

**`RecommendationAgent`** — Q-Learning 에이전트

- `select_action(step)`: 시장 대표 상태 계산 → ε-greedy → 보상 계산 → (종목명, 보상, 상태, 행동, 인덱스, raw수익률) 반환
- `learn(state, action, reward, next_step)`: Experience Replay 미니배치 학습
- `decay_epsilon(episode, total)`: Cosine Annealing ε 감소
- `get_sharpe()`: 연환산 Sharpe Ratio 계산 (√252 적용)
- `get_mdd(cumulative_returns)`: 최대낙폭(MDD) 계산

**`use_constraints` 플래그로 두 에이전트 분기:**

```python
# Vanilla RL (베이스라인)
agent_raw    = RecommendationAgent(env, use_constraints=False, ...)

# DCA Agent (제안 모델)
agent_static = RecommendationAgent(env, use_constraints=True,  ...)
```

### `app.py`

Streamlit 기반 인터랙티브 대시보드입니다.

**3탭 구조:**

- `📈 Live Monitor`: 실시간 누적 수익률 차트 + 메트릭 카드 + 학습 모니터
- `📊 Analysis`: Sharpe/MDD 카드 + 에이전트 결정 로그 테이블 + 포트폴리오 배분 차트
- `🏆 Trial History`: 다회차 수익률·Sharpe 분포 박스플롯 + 요약 테이블

**주요 UI 기능:**

- 타이틀 옆 Bull / Bear / Neutral 시장 국면 배지 실시간 표시
- 사이드바 ε progress bar — 탐색→수렴 진행률 실시간 시각화
- 차트 갱신 주기 5스텝 — 깜빡임 없는 부드러운 업데이트
- 누적 수익률은 패널티 제외 **raw_ret 기준** (실제 가격 변화율)

---

## 4. 실행 방법

### 4-1. 의존성 설치

```bash
pip install streamlit pandas numpy plotly yfinance python-dotenv
```

### 4-2. 로컬 실행

```bash
streamlit run app.py
```

### 4-3. 환경 변수 (선택)

프로젝트 루트에 `.env` 파일을 생성하고 필요한 경우 외부 API 키를 추가합니다.

```
EXTERNAL_API_KEY=your_key_here
```

---

## 5. 파라미터 가이드

| 파라미터            | 권장값 | 설명                                                      |
| ------------------- | ------ | --------------------------------------------------------- |
| Learning Rate (α)   | 0.005  | 너무 크면 Q값 발산, 너무 작으면 수렴 느림                 |
| Discount Factor (γ) | 0.85   | 높을수록 장기 수익 지향                                   |
| Exploration (ε)     | 0.5    | Cosine Annealing으로 0.01까지 자동 감소                   |
| Market Pre-Train    | 2 ~ 3  | 본 학습 전 과거 데이터 반복 횟수. 높을수록 초반 학습 안정 |
| Execution Speed     | 0.02s  | 0으로 설정 시 최대 속도                                   |

---

## 6. 성과 지표 해석

| 지표              | 의미                              | 해석 기준                                        |
| ----------------- | --------------------------------- | ------------------------------------------------ |
| Cumulative Return | 누적 수익률 (raw_ret 기준)        | 높을수록 좋음                                    |
| Win Rate          | 양수 보상 비율                    | 50% 이상이면 수익 구간이 손실보다 많음           |
| Q-Score           | Q테이블 평균 절댓값               | 학습 진행에 따라 점진적으로 증가                 |
| Sharpe Ratio      | (초과수익 평균 / 표준편차) × √252 | 1.0 이상이면 우수, 음수이면 무위험 자산보다 못함 |
| MDD               | 고점 대비 최대 낙폭               | 0에 가까울수록 방어적 전략                       |

---

## 7. 설계 결정 및 트레이드오프

### Q 행동공간 확장의 부작용

행동이 21개로 늘어나면 Q테이블이 504칸으로 커지고, 500번의 에피소드로는 모든 (상태, 행동) 쌍을 충분히 탐색하기 어렵습니다. 이를 보완하기 위해 ε 초기값을 0.5로 높이고 메모리/배치를 키웠지만, **pretrain을 3회 이상** 설정할 경우 학습 안정성이 크게 향상됩니다.

### 소프트 제약의 의도

하드 필터 방식은 RSI가 높은 강한 상승 종목을 완전히 차단해 오히려 알파를 손실시킵니다. 소프트 패널티는 위반 종목도 선택 가능하게 두되 보상에 불이익을 줌으로써 에이전트가 스스로 위험 종목을 회피하도록 유도합니다.

### 시장 대표 상태의 한계

`_get_market_state()`는 전 종목 상태의 최빈값을 사용합니다. 이는 개별 종목의 고유한 기술적 상태를 무시하는 단순화이며, 향후 종목별 독립 Q-테이블 또는 DQN으로 확장할 여지가 있습니다.

---

## 8. 향후 개선 방향

- **종목별 독립 Q-테이블**: 현재의 시장 대표 상태 대신 각 종목이 자신의 상태를 기반으로 독립 학습
- **DQN 전환**: 상태·행동 공간이 커질 경우 신경망 기반 함수 근사로 확장
- **멀티 포지션**: 현재는 1종목 집중 투자, 포트폴리오 비중 분산 지원
- **실거래 연동**: 브로커 API 연결을 통한 실시간 신호 생성

---

## 9. 면책조항

© 2026. All rights reserved.

본 시스템은 **교육 및 연구 목적**으로 제작되었으며, 실제 투자에 사용할 경우 발생하는 손실에 대해 어떠한 책임도 지지 않습니다. 모든 투자의 책임은 사용자 본인에게 있습니다.

Contact: sjowun@gmail.com
