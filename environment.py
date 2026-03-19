import numpy as np
import streamlit as st
import yfinance as yf
from config import RSI_PERIOD, SMA_PERIOD, VOLATILITY_WINDOW


@st.cache_data(ttl=3600)
def download_sp500_data(all_symbols_tuple, benchmark):
    data = yf.download(list(all_symbols_tuple), period="5y", interval="1d")['Close']
    data = data.ffill().bfill().dropna(axis=1)
    tickers = [t for t in data.columns if t != benchmark]
    return data, tickers


@st.cache_data(ttl=3600)
def validate_tickers(tickers_tuple):
    """
    입력된 티커 목록을 yfinance로 검증
    반환: (유효 티커 리스트, 무효 티커 리스트)
    - 5년치 데이터가 100일 이상 존재하면 유효로 판단
    """
    valid, invalid = [], []
    for ticker in tickers_tuple:
        try:
            df = yf.download(ticker, period="5y", interval="1d",
                             progress=False, auto_adjust=True)
            if len(df) >= 100:
                valid.append(ticker)
            else:
                invalid.append(ticker)
        except Exception:
            invalid.append(ticker)
    return valid, invalid


class SP500Environment:
    """
    S&P 500 대표 종목 데이터 + 기술적 지표 관리
    모든 에이전트가 공유하는 환경 클래스
    """
    DEFAULT_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "XOM",  "LLY",  "V",
        "JPM",  "UNH",  "WMT",  "MA",   "JNJ",
        "PG",   "HD",   "ORCL", "CVX",  "MRK",
    ]

    def __init__(self, tickers=None):
        self.benchmark  = "SPY"
        raw_tickers     = tickers or self.DEFAULT_TICKERS
        all_symbols     = raw_tickers + [self.benchmark]
        self.data, self.tickers = download_sp500_data(tuple(all_symbols), self.benchmark)
        self.vocab_size = len(self.tickers)
        self._precompute_indicators()

    def _precompute_indicators(self):
        prices = self.data[self.tickers]
        self.sma20 = prices.rolling(window=SMA_PERIOD).mean()

        delta       = prices.diff()
        gain        = delta.clip(lower=0).rolling(window=RSI_PERIOD).mean()
        loss        = (-delta.clip(upper=0)).rolling(window=RSI_PERIOD).mean()
        self.rsi    = 100 - (100 / (1 + gain / (loss + 1e-10)))

        returns          = prices.pct_change()
        self.volatility  = returns.rolling(window=VOLATILITY_WINDOW).std()
        self.vol_trend   = self.volatility.diff(5)
        self.momentum_5d = prices > prices.shift(5)

        self.benchmark_sma500 = self.data[self.benchmark].rolling(window=500).mean()

    def get_market_regime(self, step):
        if step < 500: return "Neutral"
        return "Bull" if self.data[self.benchmark].iloc[step] > self.benchmark_sma500.iloc[step] else "Bear"

    def get_state_vector(self, step, ticker_idx):
        """
        연속 상태 벡터 반환 — DQN / PPO 등 신경망 기반 에이전트용
        [RSI, SMA위치, 변동성, 5일모멘텀, 시장Regime] 5차원
        """
        ticker = self.tickers[ticker_idx]
        if step < 20:
            return np.zeros(5, dtype=np.float32)

        price = float(self.data[ticker].iloc[step])
        sma   = float(self.sma20[ticker].iloc[step])
        rsi   = float(self.rsi[ticker].iloc[step]) / 100.0        # 0~1 정규화
        vol   = float(self.volatility[ticker].iloc[step])
        vol_t = float(self.vol_trend[ticker].iloc[step]) if not np.isnan(self.vol_trend[ticker].iloc[step]) else 0.0
        mom   = 1.0 if self.momentum_5d[ticker].iloc[step] else 0.0
        sma_s = (price - sma) / (sma + 1e-10)                     # 상대적 위치

        regime_map = {"Bull": 1.0, "Neutral": 0.0, "Bear": -1.0}
        regime_val = regime_map[self.get_market_regime(step)]

        return np.array([rsi, sma_s, vol * 100, mom, regime_val], dtype=np.float32)

    def get_state(self, step, ticker_idx):
        """
        이산 상태 인덱스 반환 — Tabular Q 전용
        24가지: 2(SMA) × 3(RSI) × 2(변동성추세) × 2(모멘텀)
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
        """소프트 제약용 위반 마스크 — True=위험 종목"""
        mask = np.zeros(self.vocab_size, dtype=bool)
        if step < 20: return mask

        regime     = self.get_market_regime(step)
        rsi_upper  = 85 if regime == "Bull" else 75
        vol_mult   = 3.0 if regime == "Bull" else 1.5
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

    def get_step_reward(self, ticker, step, prev_action):
        """
        보상 계산 공통 로직 — 모든 에이전트가 재사용
        제약 조건은 보상에 반영하지 않음 → select_action의 Q 마스킹에서 처리
        """
        from config import TRANSACTION_COST, VOL_PENALTY_COEFF, DAILY_RISK_FREE_RATE

        if ticker == "CASH":
            return DAILY_RISK_FREE_RATE, DAILY_RISK_FREE_RATE

        cur     = float(self.data[ticker].iloc[step])
        nxt     = float(self.data[ticker].iloc[step + 1])
        raw_ret = ((nxt - cur) / cur) * 100
        comm    = TRANSACTION_COST if prev_action == 0 else 0.0
        vol     = float(self.volatility[ticker].iloc[step])
        reward  = raw_ret - comm - (vol * 100 * VOL_PENALTY_COEFF)
        return reward, raw_ret