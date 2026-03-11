import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import config

class SP500Environment:
    """ S&P 500 대표 종목 및 벤치마크(SPY) 데이터를 관리하는 환경 """
    def __init__(self):
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "XOM", "LLY", "V",
                        "JPM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "CVX", "MRK"]
        self.benchmark = "SPY"
        self.all_symbols = self.tickers + [self.benchmark]
        
        self.data, self.tickers = self._download_data()
        self.vocab_size = len(self.tickers)
        # Hold(현금 보유) 옵션 포함: action 수 = 종목 수 + 1
        self.num_actions = self.vocab_size + 1  
        
        # 기술적 지표 사전 계산
        self._precompute_indicators()

    @st.cache_data(ttl=3600)
    def _download_data(_self):
        data = yf.download(_self.all_symbols, period="5y", interval="1d")['Close']
        data = data.ffill().bfill().dropna(axis=1)
        tickers = [t for t in data.columns if t != _self.benchmark]
        return data, tickers

    def _precompute_indicators(self):
        """SMA, RSI, 변동성 지표를 사전 계산하여 캐싱"""
        prices = self.data[self.tickers]
        
        # SMA 20일
        self.sma20 = prices.rolling(window=20).mean()
        
        # RSI 14일
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        self.rsi = 100 - (100 / (1 + rs))
        
        # 변동성 (20일 수익률 표준편차)
        returns = prices.pct_change()
        self.volatility = returns.rolling(window=20).std()
        # 변동성의 5일 변화 (추세)
        self.vol_trend = self.volatility.diff(5)

    def get_state(self, step, ticker_idx):
        """
        주어진 시점과 종목에 대해 이산화된 상태(state index)를 반환합니다.
        
        상태 구성 (총 12가지):
        - 가격 vs SMA20: 위(1) / 아래(0) → 2가지
        - RSI 구간: 과매도 <30(0) / 중립 30~70(1) / 과매수 >70(2) → 3가지
        - 변동성 추세: 하락(0) / 상승(1) → 2가지
        → 2 × 3 × 2 = 12 상태
        """
        ticker = self.tickers[ticker_idx]
        
        # 데이터가 충분하지 않으면 기본 상태 반환
        if step < 20:
            return 0
        
        price = float(self.data[ticker].iloc[step])
        sma = float(self.sma20[ticker].iloc[step])
        rsi = float(self.rsi[ticker].iloc[step])
        vol_t = float(self.vol_trend[ticker].iloc[step]) if not np.isnan(self.vol_trend[ticker].iloc[step]) else 0.0
        
        # 이산화
        sma_state = 1 if price >= sma else 0        # 0 or 1
        
        if rsi < 30:
            rsi_state = 0   # 과매도
        elif rsi > 70:
            rsi_state = 2   # 과매수
        else:
            rsi_state = 1   # 중립
        
        vol_state = 1 if vol_t > 0 else 0            # 0 or 1
        
        state = sma_state * 6 + rsi_state * 2 + vol_state
        return state


class StaticConstraintEngine:
    """
    고도화된 제약 엔진: SMA20 + RSI + 변동성 필터를 복합 적용하여
    위험 종목을 행동 공간에서 제외합니다.
    """
    def __init__(self, env, current_step):
        self.env = env
        self.vocab_size = env.vocab_size
        self.num_actions = env.num_actions
        self.valid_mask = np.ones(self.num_actions, dtype=bool)  # Hold 포함
        
        if current_step >= 20:
            current_prices = self.env.data[self.env.tickers].iloc[current_step]
            sma_vals = self.env.sma20.iloc[current_step]
            rsi_vals = self.env.rsi.iloc[current_step]
            vol_vals = self.env.volatility.iloc[current_step]
            
            # 변동성 중앙값 (상대 비교용)
            vol_median = vol_vals.median()
            
            for i, ticker in enumerate(self.env.tickers):
                price = current_prices[ticker]
                sma = sma_vals[ticker]
                rsi = rsi_vals[ticker]
                vol = vol_vals[ticker]
                
                # 필터 1: 가격이 SMA20 아래 → 하락 추세
                if price < sma:
                    self.valid_mask[i] = False
                
                # 필터 2: RSI > 80 → 극단적 과매수 (하락 반전 위험)
                if rsi > 80:
                    self.valid_mask[i] = False
                
                # 필터 3: 변동성이 중앙값의 2배 이상 → 과도한 리스크
                if not np.isnan(vol) and not np.isnan(vol_median) and vol > vol_median * 2:
                    self.valid_mask[i] = False
            
            # Hold(현금) 옵션은 항상 유효 (마지막 인덱스)
            self.valid_mask[-1] = True
            
            # 모든 종목이 필터링되면 Hold만 남김
            if not np.any(self.valid_mask[:-1]):
                pass  # Hold만 유효한 상태 유지

    def apply_mask(self, q_values):
        """유효하지 않은 행동의 Q값을 -inf로 마스킹"""
        return np.where(self.valid_mask, q_values, -np.inf)


class RecommendationAgent:
    """
    Q-Learning 기반 강화학습 에이전트.
    기술적 지표 기반 상태에서 최적의 행동(종목 선택)을 학습합니다.
    """
    NUM_STATES = 12  # 2(SMA) × 3(RSI) × 2(변동성)
    
    def __init__(self, env, use_constraints=True, lr=0.01, gamma=0.98, eps=0.1):
        self.env = env
        self.use_constraints = use_constraints
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        self.initial_epsilon = eps
        
        # Q-테이블 초기화: (num_states, num_actions)
        # 작은 랜덤 값으로 초기화하여 초기 탐험 촉진
        self.q_table = np.random.uniform(
            low=-0.01, high=0.01, 
            size=(self.NUM_STATES, env.num_actions)
        )
        
        # 학습 기록
        self.episode_rewards = []
        self.q_value_history = []
        self.win_count = 0
        self.total_count = 0
        
    def _get_global_state(self, current_step):
        """
        모든 종목의 상태를 종합하여 시장 전체의 글로벌 상태를 결정합니다.
        각 종목의 상태를 집계(다수결)하여 하나의 대표 상태를 반환합니다.
        """
        state_counts = np.zeros(self.NUM_STATES)
        for idx in range(self.env.vocab_size):
            s = self.env.get_state(current_step, idx)
            state_counts[s] += 1
        return int(np.argmax(state_counts))
        
    def select_action(self, current_step):
        """
        ε-greedy 정책으로 행동(종목)을 선택합니다.
        
        Returns:
            (ticker_name, is_valid, reward, state, action)
        """
        state = self._get_global_state(current_step)
        engine = StaticConstraintEngine(self.env, current_step)
        
        # Q 값 가져오기
        q_values = self.q_table[state].copy()
        
        if self.use_constraints:
            q_values = engine.apply_mask(q_values)
        
        # ε-greedy 행동 선택
        if np.random.rand() < self.epsilon:
            # 탐험: 유효한 행동 중 랜덤 선택
            valid_actions = np.where(q_values > -np.inf)[0]
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.env.num_actions)
            chosen_action = int(np.random.choice(valid_actions))
        else:
            # 활용: Q값이 최대인 행동 선택
            chosen_action = int(np.argmax(q_values))
        
        # 보상 계산
        is_hold = (chosen_action == self.env.vocab_size)  # Hold 액션
        
        if is_hold:
            reward = 0.0  # 현금 보유 시 보상 0
            chosen_ticker = "HOLD"
            is_valid = True
        elif current_step + 1 < len(self.env.data):
            chosen_ticker = self.env.tickers[chosen_action]
            current_price = float(self.env.data[chosen_ticker].iloc[current_step])
            next_price = float(self.env.data[chosen_ticker].iloc[current_step + 1])
            reward = ((next_price - current_price) / current_price) * 100 if current_price > 0 else 0.0
            is_valid = engine.valid_mask[chosen_action]
        else:
            reward = 0.0
            chosen_ticker = self.env.tickers[chosen_action] if chosen_action < self.env.vocab_size else "HOLD"
            is_valid = True
        
        # 통계 업데이트
        self.total_count += 1
        if reward > 0:
            self.win_count += 1
            
        return chosen_ticker, is_valid, reward, state, chosen_action
    
    def learn(self, state, action, reward, next_step):
        """
        Q-Learning 업데이트:
        Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        """
        next_state = self._get_global_state(next_step)
        
        # 제약 조건 적용 시 다음 상태에서의 최대 Q값도 마스킹
        if self.use_constraints:
            engine = StaticConstraintEngine(self.env, next_step)
            next_q = engine.apply_mask(self.q_table[next_state].copy())
            valid_next = next_q[next_q > -np.inf]
            max_next_q = float(np.max(valid_next)) if len(valid_next) > 0 else 0.0
        else:
            max_next_q = float(np.max(self.q_table[next_state]))
        
        # TD 업데이트
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        # Q값 히스토리 기록
        self.q_value_history.append(float(np.mean(np.abs(self.q_table))))
        self.episode_rewards.append(reward)
    
    def decay_epsilon(self, episode, total_episodes):
        """
        탐험률을 에피소드 진행에 따라 선형 감소시킵니다.
        초반에는 많이 탐험하고, 후반에는 학습된 정책을 활용합니다.
        최소 탐험률은 0.01로 유지합니다.
        """
        min_eps = 0.01
        decay_rate = (self.initial_epsilon - min_eps) / max(total_episodes, 1)
        self.epsilon = max(min_eps, self.initial_epsilon - decay_rate * episode)
    
    def get_win_rate(self):
        """승률 (수익 > 0인 거래 비율) 반환"""
        if self.total_count == 0:
            return 0.0
        return (self.win_count / self.total_count) * 100
    
    def get_avg_q(self):
        """현재 Q-테이블의 평균 절대값 반환"""
        return float(np.mean(np.abs(self.q_table)))