import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from collections import deque
import random
from config import (NUM_STATES, RSI_PERIOD, SMA_PERIOD, VOLATILITY_WINDOW,
                    DAILY_RISK_FREE_RATE, TRANSACTION_COST,
                    VOL_PENALTY_COEFF, CONSTRAINT_PENALTY)


@st.cache_data(ttl=3600)
def download_sp500_data(all_symbols_tuple, benchmark):
    data = yf.download(list(all_symbols_tuple), period="5y", interval="1d")['Close']
    data = data.ffill().bfill().dropna(axis=1)
    tickers = [t for t in data.columns if t != benchmark]
    return data, tickers


class SP500Environment:
    def __init__(self):
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "XOM", "LLY", "V",
                        "JPM", "UNH", "WMT", "MA", "JNJ", "PG", "HD", "ORCL", "CVX", "MRK"]
        self.benchmark = "SPY"
        self.all_symbols = self.tickers + [self.benchmark]
        self.data, self.tickers = download_sp500_data(tuple(self.all_symbols), self.benchmark)
        self.vocab_size = len(self.tickers)
        self._precompute_indicators()

    def _precompute_indicators(self):
        prices = self.data[self.tickers]
        self.sma20 = prices.rolling(window=SMA_PERIOD).mean()

        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(window=RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(window=RSI_PERIOD).mean()
        self.rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

        returns = prices.pct_change()
        self.volatility  = returns.rolling(window=VOLATILITY_WINDOW).std()
        self.vol_trend   = self.volatility.diff(5)
        self.momentum_5d = prices > prices.shift(5)

        self.benchmark_sma500 = self.data[self.benchmark].rolling(window=500).mean()

    def get_market_regime(self, step):
        if step < 500: return "Neutral"
        return "Bull" if self.data[self.benchmark].iloc[step] > self.benchmark_sma500.iloc[step] else "Bear"

    def get_state(self, step, ticker_idx):
        """
        24 이산 상태:
        2(SMA위치) × 3(RSI구간) × 2(변동성추세) × 2(5일모멘텀)
        """
        ticker = self.tickers[ticker_idx]
        if step < 20: return 0

        price = float(self.data[ticker].iloc[step])
        sma   = float(self.sma20[ticker].iloc[step])
        rsi   = float(self.rsi[ticker].iloc[step])
        vol_t = float(self.vol_trend[ticker].iloc[step]) if not np.isnan(self.vol_trend[ticker].iloc[step]) else 0.0
        mom   = 1 if self.momentum_5d[ticker].iloc[step] else 0

        sma_s = 1 if price >= sma else 0
        rsi_s = 0 if rsi < 30 else (2 if rsi > 70 else 1)
        vol_s = 1 if vol_t > 0 else 0

        return (mom * 12) + (sma_s * 6) + (rsi_s * 2) + vol_s

    def get_constraint_mask(self, step):
        """
        [원인2 수정] 하드 필터 → 소프트 패널티를 위한 위반 여부 마스크 반환
        True = 제약 위반(위험), False = 통과(안전)
        """
        mask = np.zeros(self.vocab_size, dtype=bool)
        if step < 20:
            return mask

        regime = self.get_market_regime(step)
        rsi_upper = 85 if regime == "Bull" else 75
        vol_mult  = 3.0 if regime == "Bull" else 1.5
        sma_filter = (regime != "Bull")

        prices_now = self.data[self.tickers].iloc[step]
        sma_vals   = self.sma20.iloc[step]
        rsi_vals   = self.rsi.iloc[step]
        vol_vals   = self.volatility.iloc[step]
        vol_median = vol_vals.median()

        for i, ticker in enumerate(self.tickers):
            if sma_filter and prices_now[ticker] < sma_vals[ticker]:
                mask[i] = True
            if rsi_vals[ticker] > rsi_upper:
                mask[i] = True
            if not np.isnan(vol_vals[ticker]) and vol_vals[ticker] > vol_median * vol_mult:
                mask[i] = True
        return mask


class RecommendationAgent:
    """
    [원인1+4 수정] Q 행동 공간 확장:
    기존: Q가 BUY/CASH(2개)만 결정, 종목은 모멘텀으로 따로 선택
    변경: Q가 {CASH=0, 종목0=1, 종목1=2, ...종목N} 전체를 직접 결정
    → 종목 선택 + BUY/CASH 결정이 Q 하나로 통합
    """
    NUM_STATES = NUM_STATES  # 24

    def __init__(self, env, use_constraints=True, lr=0.005, gamma=0.85, eps=0.5):
        self.env = env
        self.use_constraints = use_constraints
        self.lr, self.gamma  = lr, gamma
        self.epsilon = self.initial_epsilon = eps

        # [원인1+4] 행동 수: 0=CASH, 1~N=각 종목
        self.num_actions = 1 + env.vocab_size  # CASH + 종목 수
        self.q_table = np.random.uniform(
            low=-0.01, high=0.01,
            size=(self.NUM_STATES, self.num_actions)
        )

        # [원인3] Q테이블 초기화: CASH 행동에 약한 양수 bias
        # → 초반 학습 전에 CASH를 기본값으로 선호하도록 유도
        self.q_table[:, 0] = 0.02

        self.prev_action     = 0
        self.memory          = deque(maxlen=5000)   # 메모리 2000 → 5000
        self.batch_size      = 64                   # 배치 32 → 64

        self.episode_rewards, self.win_count, self.total_count = [], 0, 0

    def _get_market_state(self, step):
        """전체 종목의 상태 중 가장 빈도 높은 상태를 시장 대표 상태로 사용"""
        counts = np.zeros(self.NUM_STATES)
        for i in range(self.env.vocab_size):
            counts[self.env.get_state(step, i)] += 1
        return int(np.argmax(counts))

    def select_action(self, current_step):
        # 시장 대표 상태 (전 종목 상태의 최빈값)
        state = self._get_market_state(current_step)

        # ε-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = int(np.argmax(self.q_table[state]))

        # action 해석: 0=CASH, 1~N=종목 인덱스
        raw_ret = 0.0
        reward  = 0.0

        if action == 0:
            # CASH 선택
            chosen_ticker = "CASH"
            ticker_idx    = 0
            reward  = DAILY_RISK_FREE_RATE
            raw_ret = DAILY_RISK_FREE_RATE

        else:
            ticker_idx    = action - 1   # Q action → 종목 인덱스
            chosen_ticker = self.env.tickers[ticker_idx]

            cur     = float(self.env.data[chosen_ticker].iloc[current_step])
            nxt     = float(self.env.data[chosen_ticker].iloc[current_step + 1])
            raw_ret = ((nxt - cur) / cur) * 100

            comm = TRANSACTION_COST if self.prev_action == 0 else 0.0
            vol  = float(self.env.volatility[chosen_ticker].iloc[current_step])
            reward = raw_ret - comm - (vol * 100 * VOL_PENALTY_COEFF)  # [원인5] 패널티 0.1→0.03

            # [원인2] 제약 위반 시 하드 차단 대신 소프트 패널티
            if self.use_constraints:
                constraint_mask = self.env.get_constraint_mask(current_step)
                if constraint_mask[ticker_idx]:
                    reward -= CONSTRAINT_PENALTY  # 위반해도 거래는 허용, 보상만 차감

        self.prev_action = action
        self.total_count += 1
        if reward > 0: self.win_count += 1

        return chosen_ticker, True, reward, state, action, ticker_idx, raw_ret

    def learn(self, state, action, reward, next_step):
        next_state = self._get_market_state(next_step)
        self.memory.append((state, action, reward, next_state))

        if len(self.memory) >= self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for m_s, m_a, m_r, m_ns in minibatch:
                max_next_q = np.max(self.q_table[m_ns])
                td_target  = m_r + self.gamma * max_next_q
                self.q_table[m_s, m_a] += self.lr * (td_target - self.q_table[m_s, m_a])

        self.episode_rewards.append(reward)

    def decay_epsilon(self, episode, total_episodes):
        # Cosine Annealing
        min_eps = 0.01
        cos_val = 0.5 * (1 + np.cos(np.pi * episode / total_episodes))
        self.epsilon = min_eps + (self.initial_epsilon - min_eps) * cos_val

    def get_win_rate(self):
        return (self.win_count / self.total_count * 100) if self.total_count > 0 else 0.0

    def get_avg_q(self):
        return float(np.mean(np.abs(self.q_table)))

    def get_sharpe(self):
        if len(self.episode_rewards) < 2: return 0.0
        excess = np.array(self.episode_rewards) - DAILY_RISK_FREE_RATE
        std = np.std(excess)
        if std < 1e-10: return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    def get_mdd(self, cumulative_returns):
        if len(cumulative_returns) < 2: return 0.0
        arr  = np.array(cumulative_returns)
        peak = np.maximum.accumulate(arr)
        return float(np.min(arr - peak))