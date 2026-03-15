# Test-Constrained-RL-ColdStart

## KDT Team Study : 팀 Chainers 🫡

S&P 500 대표 종목을 대상으로 제약 조건 강화학습(Constrained Reinforcement Learning)을 적용한 포트폴리오 관리 시뮬레이션 시스템입니다. 논문 "Vectorizing the Trie"에서 제안된 STATIC 프레임워크의 제약 디코딩(Constrained Decoding) 개념을 주식 시장에 이식하여, 금융 시장의 불확실성 속에서 리스크를 억제하고 안정적인 수익률을 도출하도록 설계되었습니다.

- **논문**: Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators
- **논문 링크**: https://arxiv.org/pdf/2602.22647
- **웹 데모**: https://test-constrained-rl-coldstart-cgwsjhrq57w4jm48fqzbmm.streamlit.app/

---

## 1. 프로젝트 개요

강화학습 에이전트가 S&P 500 지수 및 개별 종목 데이터를 실시간으로 수집하여 최적의 종목을 선택합니다.

- 단순 수익률 추종 방식(Vanilla RL)과 지수 이동평균(EMA) 필터 및 현금 보유 옵션을 적용한 제약 조건 방식(STATIC RL)의 성과를 비교 분석합니다.
- 라이트 모드 및 다크 모드 모두에서 차트, 텍스트, 지표가 정상적으로 표시됩니다.

**[이미지 1] 메인 대시보드 - 라이트 모드**

<img width="875" height="641" alt="메인대시보드-라이트모드" src="https://github.com/user-attachments/assets/f229c66c-d404-4a6b-8112-f57397481416" />

**[이미지 2] 메인 대시보드 - 다크 모드**

<img width="866" height="644" alt="강화학습-그림1_어두운배경" src="https://github.com/user-attachments/assets/44f24630-4c33-4a5c-a236-39882d22a876" />

---

## 2. RL과 STATIC의 연결

에이전트가 행동을 선택할 때, 모든 가능한 행동 공간 중에서 비즈니스 로직이나 안전 기준을 충족하는 유효한 행동만을 선택하도록 제한하는 것이 제약 조건 강화학습의 핵심입니다.

`agent.py`에 구현된 `StaticConstraintEngine`은 논문의 제약 디코딩 개념을 차용하여, 주가가 이동평균선(EMA) 아래에 있는 위험 종목을 행동 공간에서 제외(Masking)함으로써 에이전트가 안전한 경로를 탐색하도록 강제합니다.

---

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

---

## 4. 데이터 획득 및 웹 배포 아키텍처

### 4-1. Alpha Vantage API를 통한 실시간 데이터 연동

기존 주가 데이터 수집 방식을 확장하여 [Alpha Vantage](https://www.alphavantage.co/support/#) API를 통합했습니다. 이를 통해 SPY 및 S&P 500 주요 종목의 실시간 주가 정보를 안정적으로 확보합니다.

**API 키 및 환경 변수 관리**

- `config.py`는 `EXTERNAL_API_KEY` 환경 변수를 참조하도록 구성되어 있습니다.
- 보안을 위해 API 키는 소스 코드에 하드코딩하지 않고 `.env` 파일에 환경 변수로 저장하여 관리합니다.
- 배포된 웹 버전(Streamlit Cloud)은 해당 키를 Streamlit Secrets에 등록한 상태이므로, 로컬 실행 시 `.env` 파일을 별도로 생성할 필요가 없습니다.
- 데이터 수집에 사용되는 `yfinance`는 별도의 인증 없이 공개 주가 데이터를 수집합니다.

### 4-2. Streamlit Cloud를 통한 웹 배포

로컬 환경에서 개발된 대시보드를 [Streamlit Cloud](https://share.streamlit.io/)를 통해 웹으로 배포했습니다.

1. **GitHub 연동**: 소스 코드를 GitHub 리포지토리에 업로드합니다.
2. **배포 설정**: Streamlit Cloud에서 해당 리포지토리를 연결하여 실시간 웹 서비스를 가동합니다.
3. **접근성 확보**: 전문가 및 사용자가 별도의 설치 없이 URL을 통해 강화학습 에이전트의 성과를 실시간으로 모니터링할 수 있습니다.
4. **주의할 점**: 실행 tool인 VScode의 파이썬 버전(3.12)과 Streamlit에서의 파이썬 버전을 일치시켜야 한다.
                  -> Streamlit 에서는 파이썬 버전의 기본값이 3.14로 되어있어서, 필요시 본인의 VScode의 파이썬 버전(3.12)으로 바꾸어야 한다.
                  -> 여기서는 Python 3.12 버전이다. 

---

## 5. 실행 방법

터미널에서 프로젝트 폴더로 이동한 후 다음 명령어를 실행합니다.

```bash
streamlit run app.py
```

실행 후 웹 브라우저에서 출력되는 로컬 주소(기본값 `http://localhost:8501`)로 접속합니다.

---

## 6. 웹페이지 구성 및 사용 방법

### 6-1. 좌측 사이드바 설정

| 설정 항목 | 설명 | 기본값 |
|---|---|---|
| Episodes (Trading Days) | 시뮬레이션 거래일 수 | 100 |
| Frame Speed | 애니메이션 프레임 속도 | 0.03초 |
| Base Random Seed | 재현성을 위한 랜덤 시드 | 2026 |
| Auto Run Count | 자동 반복 실행 횟수 | 30 |

### 6-2. RL 하이퍼파라미터 제어

슬라이더를 통해 아래 값을 조절합니다.

- **Learning Rate (alpha, 최적값 0.005)**: 학습률. 클수록 최근 보상에 민감하게 반응하나 노이즈에 취약합니다.
- **Discount Factor (gamma, 최적값 0.85)**: 미래 보상의 현재 가치 감쇠 인자. 주식 시장 특성상 낮은 값이 단기 추세 포착에 유리합니다.
- **Exploration (epsilon, 최적값 0.1)**: 탐험률. 클수록 다양한 종목을 탐색하나 변동성이 증가합니다.
- **Pre-Train Episodes (최적값 3)**: 실전 투입 전 과거 데이터를 통한 영점 조절 횟수입니다.


### 6-3. 실행 및 결과 확인

Run Evaluation 버튼을 클릭하면 실시간 수익률 곡선이 업데이트됩니다.

---

## 7. 주요 분석 지표

### 7-1. Cumulative Return Comparison

Vanilla RL, STATIC RL, S&P 500(SPY) 세 가지 수익률 곡선을 비교합니다. STATIC RL은 EMA 필터와 현금 보유 옵션을 통해 하방 리스크를 방어하며, Vanilla RL 대비 완만한 곡선과 우수한 손실 방어 능력을 보입니다.

### 7-2. Agent Decision Analysis

각 거래일별 에이전트의 종목 선택 로그와 종목별 매수 빈도를 확인합니다.

### 7-3. Trial History: Statistical Analysis

다회차 반복 실행 결과의 기대 수익률(Mean), 중앙값(Median), 변동성(표준편차) 등을 분석합니다. 

**최적화 결과 (1235일 풀 시뮬레이션):**
- **최고 수익률**: STATIC RL **299.29%** 달성 (Alpha: 0.005, Gamma: 0.85, Pre-train: 3)
- **승률(Win Rate)**: STATIC RL 약 50.8% 수준 확보

| 지표 (1235일 기준) | Vanilla RL | STATIC RL |
|---|---|---|
| 최고 수익률(Max) | 226.37% | **299.29%** |
| 평균 승률(Win Rate) | 47.83% | **50.75%** |

중앙값 및 최고 수익률 기준으로 STATIC RL이 Vanilla RL을 압도합니다. 이는 제약 조건이 단순히 손실을 막는 것을 넘어, 유망한 종목에 집중하게 함으로써 장기적으로 더 높은 복리 효과를 창출함을 증명합니다.


**[이미지 3] Trial History 통계 분석 - 라이트 모드**

<img width="877" height="594" alt="Trial History-라이트모드" src="https://github.com/user-attachments/assets/e42e4735-9855-434b-bdca-8083658c7adf" />

**[이미지 4] Trial History 통계 분석 - 다크 모드**

<img width="871" height="590" alt="강화학습-그림2_어두운배경" src="https://github.com/user-attachments/assets/765175d9-d7b5-480f-9f38-ee29368cb755" />

---

---

## 8. 에이전트 고도화 (Advanced Features)

본 프로젝트는 단순 Q-Learning을 넘어 전문가 수준의 트레이딩 로직을 추가로 구현했습니다.

### 8-1. Experience Replay (경험 재현)
에이전트가 매 스텝 학습하는 대신, 과거의 경험을 메모리에 저장했다가 무작위로 추출하여 학습합니다. 이는 데이터 간의 상관성을 제거하고 학습의 안정성을 획기적으로 높입니다.

### 8-2. Risk-Adjusted Reward (변동성 패널티)
단순 수익률이 아닌 **'수익률 - (변동성 * 가중치)'** 형태의 보상 함수를 적용했습니다. 이를 통해 에이전트가 변동성이 큰 위험한 수익보다 안정적인 우상향 종목을 선호하도록 유도합니다.

### 8-3. Expanded State Space (상태 공간 확장)
기존의 SMA, RSI, 변동성 지표에 **'5일 모멘텀(추세)'** 지표를 추가하여 상태 공간을 24개로 확장했습니다. 더욱 세밀한 시장 상황 판단이 가능해졌습니다.

### 8-4. 결과 내보내기 (CSV Export)
시뮬레이션 완료 후 모든 시행 기록을 CSV 파일로 즉시 다운로드할 수 있는 기능을 추가하여 데이터 기반의 사후 분석을 지원합니다.

---

## 9. 결론

STATIC 프레임워크와 에이전트 고도화(Experience Replay, Risk Penalty 등)를 주식 시장에 적용한 결과, 단순 RL 대비 **압도적인 수익률(299.29%)**과 **낮은 변동성**을 동시에 달성할 수 있었습니다. 이는 금융 도메인에서 제약 조건(Constraints)과 정교한 보상 설계가 AI의 성능을 결정짓는 핵심 요소임을 시사합니다.


---

## 저작권

본 저장소에 포함된 코드 및 모든 출력 이미지 결과물은 저작권법에 의해 보호됩니다.

저작권자의 명시적 허가 없이 본 자료의 전부 또는 일부를 복제, 배포, 수정, 상업적으로 이용하는 행위를 금합니다.

© 2026. All rights reserved.
Contact: sjowun@gmail.com
