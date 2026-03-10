# Test-Constrained-RL-ColdStart

## KDT Team Study : 팀 Chainers 🫡

S&P 500 대표 종목을 대상으로 제약 조건 강화학습(Constrained Reinforcement Learning)을 적용한 포트폴리오 관리 시뮬레이션 시스템입니다. 논문 "Vectorizing the Trie"에서 제안된 STATIC 프레임워크 개념을 주식 시장에 이식하여, 금융 시장의 불확실성 속에서 리스크를 억제하고 안정적인 수익률을 도출하도록 설계되었습니다.

- 논문 제목 : Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators
- 논문 링크 : https://arxiv.org/pdf/2602.22647

- ✍️ 웹 데모: https://test-constrained-rl-coldstart-cgwsjhrq57w4jm48fqzbmm.streamlit.app/

---

## 1. 프로젝트 개요

강화학습 에이전트가 S&P 500 지수 및 개별 종목 데이터를 실시간으로 수집하여 최적의 종목을 선택합니다.

- 단순 수익률 추종 방식(Vanilla RL)과 지수 이동평균(EMA) 필터 및 현금 보유 옵션을 적용한 제약 조건 방식(STATIC RL)의 성과를 비교 분석합니다.
- 라이트 모드 및 다크 모드 모두에서 차트, 텍스트, 지표가 정상적으로 표시됩니다.

**라이트 모드**

<img width="875" height="641" alt="강화학습-그림1" src="https://github.com/user-attachments/assets/f229c66c-d404-4a6b-8112-f57397481416" />

**다크 모드**

<!-- 다크모드 메인 대시보드 스크린샷 삽입 위치 -->
<!-- <img alt="다크모드-메인대시보드" src="" /> -->

## 2. RL과 STATIC의 연결

에이전트가 행동을 선택할 때, 모든 가능한 행동 공간 중에서 비즈니스 로직이나 안전 기준을 충족하는 유효한 행동만을 선택하도록 제한하는 것이 제약 조건 강화학습의 핵심입니다.

- `agent.py`에 구현된 `StaticConstraintEngine`은 논문의 개념을 차용하여, 주가가 이동평균선(EMA) 아래에 있는 위험 종목을 행동 공간에서 제외(Masking)함으로써 에이전트가 안전한 경로를 탐색하도록 강제합니다.

## 3. 사전 준비 및 환경 설정

다음 라이브러리 설치가 필요합니다.

```bash
pip install streamlit pandas numpy plotly yfinance
```

| 라이브러리 | 용도 |
|---|---|
| streamlit | 웹 대시보드 구성 |
| pandas, numpy | 데이터 처리 및 수치 연산 |
| plotly | 인터랙티브 그래프 시각화 |
| yfinance | 실시간 주가 데이터 수집 |

## 4. 실행 방법

터미널에서 프로젝트 폴더로 이동한 후 다음 명령어를 실행합니다.

```bash
streamlit run app.py
```

## 5. 웹페이지 구성 및 사용 방법

실행 후 웹 브라우저에서 출력되는 로컬 주소(기본값 `http://localhost:8501`)로 접속합니다.

**5-1. 좌측 사이드바 설정**

- Episodes (Trading Days): 시뮬레이션 거래일 수 설정 (기본값 100)
- Frame Speed: 애니메이션 프레임 속도 (기본값 0.03초)
- Base Random Seed: 재현성을 위한 랜덤 시드 (기본값 2026)
- Auto Run Count: 자동 반복 실행 횟수 (기본값 30)

**5-2. RL 하이퍼파라미터 제어**

슬라이더를 통해 아래 값을 조절합니다.

- Learning Rate (alpha): 학습률. 클수록 최근 보상에 민감하게 반응하나 노이즈에 취약합니다.
- Discount Factor (gamma, 기본값 0.98): 미래 보상의 현재 가치 감쇠 인자. 1에 가까울수록 장기 수익을 중시합니다.
- Exploration (epsilon): 탐험률. 클수록 다양한 종목을 탐색하나 변동성이 증가합니다.

**5-3. 실행 및 결과 확인**

Run Evaluation 버튼을 클릭하면 실시간 수익률 곡선이 업데이트됩니다.

## 6. 주요 분석 지표

**6-1. Cumulative Return Comparison**

Vanilla RL, STATIC RL, S&P 500(SPY) 세 가지 수익률 곡선을 비교합니다. STATIC RL은 EMA 필터와 현금 보유 옵션을 통해 하방 리스크를 방어하며, Vanilla RL 대비 완만한 곡선과 우수한 손실 방어 능력을 보입니다.

**6-2. Agent Decision Analysis**

각 거래일별 에이전트의 종목 선택 로그와 종목별 매수 빈도를 확인합니다.

**6-3. Trial History: Statistical Analysis**

다회차 반복 실행 결과의 기대 수익률(Mean), 중앙값(Median), 변동성(표준편차) 등을 분석합니다. 아래는 시뮬레이션 30회 기준 주요 통계입니다. (Alpha 기대치: STATIC +4.25%p, Vanilla +4.04%p vs 시장 평균)

| 지표 | Vanilla RL | STATIC RL |
|---|---|---|
| 평균(Mean) | 14.76% | 14.96% |
| 중앙값(Median) | 14.22% | 18.73% |
| 표준편차(σ) | 11.56% | 11.80% |
| 범위 | -3.75% ~ 44.12% | -3.62% ~ 35.17% |

중앙값 기준으로 STATIC RL이 Vanilla RL을 압도합니다. 이는 50% 이상의 확률로 STATIC 모델이 더 안정적이고 높은 수익을 보장함을 의미합니다. 범위 역시 STATIC이 더 좁게 형성되어 극단적 손실 발생 가능성이 억제됩니다.

**라이트 모드**

<img width="877" height="594" alt="강화학습-그림2" src="https://github.com/user-attachments/assets/e42e4735-9855-434b-bdca-8083658c7adf" />

**다크 모드**

<!-- 다크모드 Trial History 스크린샷 삽입 위치 -->
<!-- <img alt="다크모드-Trial History" src="" /> -->

## 7. 결론

STATIC 프레임워크를 강화학습에 이식한 결과, 제약 조건은 단순 수익률 극대화보다 위험 대비 수익률(Risk-adjusted Return)의 최적화와 재현 가능한 안정적인 성과를 도출하는 데 결정적인 역할을 수행함이 확인되었습니다.

---

## 저작권

본 저장소에 포함된 코드 및 모든 출력 이미지 결과물은 저작권법에 의해 보호됩니다.

저작권자의 명시적 허가 없이 본 자료의 전부 또는 일부를 복제, 배포, 수정, 상업적으로 이용하는 행위를 금합니다.

© 2026. All rights reserved.  
Contact : sjowun@gmail.com
