import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from collections import deque
import random

class SP500Environment:
    """ S&P 500 대표 종목 및 벤치마크(SPY) 데이터를 관리하는 환경 """
    def __init__(self):
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "XOM", "LLY", "V",
                        "JPM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "CVX", "MRK"]
        self.benchmark = "SPY"
        self.all_symbols = self.tickers + [self.benchmark]
        
        self.data, self.tickers = self._download_data()
        self.vocab_size = len(self.tickers)
        self.num_actions = 2  # 0: CASH, 1: BUY
        
        self._precompute_indicators()

    @st.cache_data(ttl=3600)
    def _download_data(_self):
        data = yf.download(_self.all_symbols, period="5y", interval="1d")['Close']
        data = data.ffill().bfill().dropna(axis=1)
        tickers = [t for t in data.columns if t != _self.benchmark]
        return data, tickers

    def _precompute_indicators(self):
        prices = self.data[self.tickers]
        self.sma20 = prices.rolling(window=20).mean()
        
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        self.rsi = 100 - (100 / (1 + rs))
        
        returns = prices.pct_change()
        self.volatility = returns.rolling(window=20).std()
        self.vol_trend = self.volatility.diff(5)
        self.momentum_5d = prices > prices.shift(5)

        benchmark_prices = self.data[self.benchmark]
        self.benchmark_sma500 = benchmark_prices.rolling(window=500).mean()

    def get_market_regime(self, step):
        if step < 500: return "Neutral"
        spy_price = self.data[self.benchmark].iloc[step]
        spy_sma = self.benchmark_sma500.iloc[step]
        return "Bull" if spy_price > spy_sma else "Bear"

    def get_state(self, step, ticker_idx):
        """
        주어진 시점과 종목에 대해 이산화된 상태(state index)를 반환합니다. (성능 최적화 복구)
        
        상태 구성 (총 24가지):
        - 가격 vs SMA20: 위(1) / 아래(0) -> 2가지
        - RSI 구간: 과매도 <30(0) / 중립 30~70(1) / 과매수 >70(2) -> 3가지
        - 변동성 추세: 하락(0) / 상승(1) -> 2가지
        - 5일 모멘텀: 상승(1) / 하락(0) -> 2가지
        -> 2 * 3 * 2 * 2 = 24 상태
        """
        ticker = self.tickers[ticker_idx]
        if step < 20: return 0
        
        price = float(self.data[ticker].iloc[step])
        sma = float(self.sma20[ticker].iloc[step])
        rsi = float(self.rsi[ticker].iloc[step])
        vol_t = float(self.vol_trend[ticker].iloc[step]) if not np.isnan(self.vol_trend[ticker].iloc[step]) else 0.0
        mom = 1 if self.momentum_5d[ticker].iloc[step] else 0
        
        # 이산화 로직
        sma_state = 1 if price >= sma else 0
        rsi_state = 0 if rsi < 30 else (2 if rsi > 70 else 1)
        vol_state = 1 if vol_t > 0 else 0
        
        # 24진수/가중치 결합 상태
        state = (mom * 12) + (sma_state * 6) + (rsi_state * 2) + vol_state
        return state

class AdaptiveConstraintEngine:
    """ 시장 상황(Regime)에 따라 임계값을 유연하게 조정하는 동적 제약 엔진 """
    def __init__(self, env, current_step):
        self.env = env
        self.num_tickers = len(env.tickers)
        self.valid_mask = np.ones(self.num_tickers, dtype=bool)
        regime = env.get_market_regime(current_step)
        
        if regime == "Bull":
            rsi_upper, vol_mult, sma_filter = 85, 3.0, False
        else:
            rsi_upper, vol_mult, sma_filter = 75, 1.5, True
            
        if current_step >= 20:
            current_prices = self.env.data[self.env.tickers].iloc[current_step]
            sma_vals = self.env.sma20.iloc[current_step]
            rsi_vals = self.env.rsi.iloc[current_step]
            vol_vals = self.env.volatility.iloc[current_step]
            vol_median = vol_vals.median()
            
            for i, ticker in enumerate(self.env.tickers):
                if sma_filter and current_prices[ticker] < sma_vals[ticker]:
                    self.valid_mask[i] = False
                if rsi_vals[ticker] > rsi_upper:
                    self.valid_mask[i] = False
                if not np.isnan(vol_vals[ticker]) and vol_vals[ticker] > vol_median * vol_mult:
                    self.valid_mask[i] = False

class RecommendationAgent:
    """ Tabular Q-Learning + Experience Replay 에이전트 (성능 복구 버전) """
    NUM_STATES = 24
    
    def __init__(self, env, use_constraints=True, lr=0.005, gamma=0.85, eps=0.1):
        self.env = env
        self.use_constraints = use_constraints
        self.lr, self.gamma = lr, gamma
        self.epsilon = self.initial_epsilon = eps
        
        # Q-테이블 하이브리드 초기화
        self.q_table = np.random.uniform(low=-0.01, high=0.01, size=(self.NUM_STATES, 2))
        
        self.prev_action = 0
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        self.episode_rewards, self.win_count, self.total_count = [], 0, 0

    def _get_global_state(self, current_step):
        counts = np.zeros(self.NUM_STATES)
        for i in range(self.env.vocab_size):
            counts[self.env.get_state(current_step, i)] += 1
        return int(np.argmax(counts))
        
    def select_action(self, current_step):
        engine = AdaptiveConstraintEngine(self.env, current_step)
        valid_indices = np.where(engine.valid_mask)[0]
        
        if len(valid_indices) == 0:
            chosen_ticker, chosen_action = "NONE", 0
            state = self._get_global_state(current_step)
        else:
            momentum_scores = []
            for idx in valid_indices:
                ticker = self.env.tickers[idx]
                price, price_5d = self.env.data[ticker].iloc[current_step], self.env.data[ticker].iloc[max(0, current_step - 5)]
                momentum_scores.append((price - price_5d) / price_5d if price_5d > 0 else 0)
            
            target_idx = valid_indices[np.argmax(momentum_scores)]
            chosen_ticker = self.env.tickers[target_idx]
            state = self.env.get_state(current_step, target_idx)
            
            # ε-greedy (Tabular)
            if np.random.rand() < self.epsilon:
                chosen_action = np.random.randint(2)
            else:
                chosen_action = int(np.argmax(self.q_table[state]))

        reward = 0.0
        if chosen_action == 1 and chosen_ticker != "NONE":
            cur, nxt = float(self.env.data[chosen_ticker].iloc[current_step]), float(self.env.data[chosen_ticker].iloc[current_step + 1])
            raw_ret = ((nxt - cur) / cur) * 100
            comm = 0.10 if self.prev_action == 0 else 0.0
            vol = float(self.env.volatility[chosen_ticker].iloc[current_step])
            reward = raw_ret - comm - (vol * 100 * 0.1)
        else:
            chosen_ticker, reward = "CASH", 0.0
            
        self.prev_action = chosen_action
        self.total_count += 1
        if reward > 0: self.win_count += 1
        return chosen_ticker, True, reward, state, chosen_action
    
    def learn(self, state, action, reward, next_step):
        next_state = self._get_global_state(next_step)
        self.memory.append((state, action, reward, next_state))
        
        # Q-Table + Experience Replay (하이브리드 학습)
        if len(self.memory) >= self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for m_state, m_action, m_reward, m_next_state in minibatch:
                max_next_q = np.max(self.q_table[m_next_state])
                td_target = m_reward + self.gamma * max_next_q
                self.q_table[m_state, m_action] += self.lr * (td_target - self.q_table[m_state, m_action])
                
        self.episode_rewards.append(reward)

    def decay_epsilon(self, episode, total_episodes):
        min_eps = 0.01
        self.epsilon = max(min_eps, self.initial_epsilon - (self.initial_epsilon - min_eps) * (episode / total_episodes))
    
    def get_win_rate(self):
        return (self.win_count / self.total_count * 100) if self.total_count > 0 else 0.0
    
    def get_avg_q(self):
        return float(np.mean(np.abs(self.q_table)))