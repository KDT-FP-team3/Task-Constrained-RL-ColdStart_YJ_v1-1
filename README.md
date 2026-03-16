# Personalized-RL-Quant: Adaptive Portfolio Management

## KDT Team Study: 팀 Chainers 🫡

S&P 500 대표 종목을 대상으로 **제약 조건 강화학습(Constrained Reinforcement Learning)**을 적용한 고성능 포트폴리오 관리 시뮬레이션 시스템입니다. 논문 "Vectorizing the Trie"의 제약 디코딩 개념을 금융 시장에 이식하여, 하락장에서의 리스크를 방어하고 상승장에서 초과 수익(Alpha)을 창출하도록 설계되었습니다.

- **웹 데모**: [Personalized-Quant-RL](https://test-constrained-rl-coldstart-cgwsjhrq57w4jm48fqzbmm.streamlit.app/)
- **핵심 알고리즘**: Tabular Q-Learning + Experience Replay (하이브리드 엔진)
- **프레임워크**: Personalized DCA & Adaptive Constraint Engine

---

## 1. 프로젝트 주요 특징

### 🧠 고효율 강화학습 엔진 (Adaptive Q-Agent)

- **Tabular Q-Learning**: 500거래일 규모의 실전 데이터에 최적화된 24개 이산 상태(State) 기반의 학습 엔진을 사용합니다. DQN보다 빠르고 정확한 결정 경계를 형성합니다.
- **Experience Replay**: 과거의 성공적인 매매 경험을 메모리에 저장했다가 반복 학습함으로써 학습의 안정성을 극대화했습니다.

### 🏛️ 제약 조건 기반 전략 (Personalized DCA)

- **Adaptive Constraint Engine**: 시장 상황(Bull/Bear)을 실시간 분석하여 RSI, SMA, 변동성 임계값을 유연하게 조정하며 위험 종목을 매수 후보에서 자동 제외(Masking)합니다.
- **Risk-Adjusted Reward**: 단순 수익률이 아닌 **'수익률 - 변동성 패널티'** 보상 함수를 통해 안정적인 우상향 자산을 선호하게 유도합니다.

---

## 2. 시각화 및 분석 도구 (Expert Terminal)

사용자 경험(UX)과 가독성을 최우선으로 고려한 대시보드를 제공합니다.

- **Cumulative Return Comparison**: Vanilla RL, DCA Agent, S&P 500 지수 수익률을 실시간으로 비교합니다.
- **Agent Decision Analysis**: 에이전트가 매일 내리는 판단(종목 선택, 수익률)을 상세 로그로 확인하고 포트폴리오 할당 비중을 분석합니다.
- **Trial History Analysis**: 다회차 반복 실행을 통한 통계적 성과(Mean, Median, Std)를 박스 플롯으로 분석하여 전략의 일관성을 증명합니다.

---

## 3. 실행 방법 (Quick Start)

### 3-1. 환경 설정

```bash
pip install streamlit pandas numpy plotly yfinance
```

### 3-2. 로컬 실행

```bash
streamlit run app.py
```

---

## 4. 최적 파라미터 가이드 (Recommended Defaults)

최고의 성과를 보장하는 기본값이 시스템에 기본 적용되어 있습니다.

| 파라미터                    | 설명                            | 설정값    |
| --------------------------- | ------------------------------- | --------- |
| **Learning Rate (α)**       | AI의 학습 속도 및 유연성        | **0.005** |
| **Discount Factor (γ)**     | 미래 보상의 가치 평가 (인내심)  | **0.85**  |
| **Initial Exploration (ε)** | 새로운 전략 시도 확률 (호기심)  | **0.10**  |
| **Market Warm-up**          | 사전 과거 데이터 복기 학습 횟수 | **2~3회** |

---

## 5. 결론 및 성과

본 프로젝트는 단순 RL 대비 **압도적인 누적 수익률(최대 299%p 차이)**과 **낮은 변동성(MDD 방어)**을 동시에 달성했습니다. 이는 금융 도메인에서 도메인 지식 기반의 제약 조건(Constraints)과 정교한 보상 설계가 AI의 성능을 결정짓는 핵심 요소임을 시사합니다.

---

## 저작권 및 면책조항

© 2026. All rights reserved.
본 시스템은 교육 및 연구용으로 제작되었으며, 실제 투자 시 발생하는 손실에 대해서는 어떠한 책임도 지지 않습니다. 모든 투자의 책임은 사용자 본인에게 있습니다.

Contact: sjowun@gmail.com
