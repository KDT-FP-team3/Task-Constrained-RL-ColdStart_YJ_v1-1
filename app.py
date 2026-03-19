import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go
import plotly.express as px
from environment import SP500Environment, validate_tickers
from agents import AGENT_REGISTRY, AGENT_PARAMS, DQN_FAMILY, PPO_FAMILY
from agents.base import BaseAgent
from hpo import optimize, optimize_all_models, OPTUNA_AVAILABLE, SEARCH_SPACE, HPO_SETTINGS_GUIDE

st.set_page_config(page_title="Personalized-RL Quant", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-weight: 700 !important; font-size: 1.9rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; opacity: 0.85; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
    thead tr th { font-weight: 700 !important; }
    .regime-bull    { background:#1a7a3a; color:#fff; padding:3px 14px;
                      border-radius:12px; font-size:0.85rem; font-weight:600; margin-left:10px; }
    .regime-bear    { background:#b33030; color:#fff; padding:3px 14px;
                      border-radius:12px; font-size:0.85rem; font-weight:600; margin-left:10px; }
    .regime-neutral { background:#555;    color:#fff; padding:3px 14px;
                      border-radius:12px; font-size:0.85rem; font-weight:600; margin-left:10px; }
</style>
""", unsafe_allow_html=True)

# ── 사이드바: 종목 설정 (가장 먼저) ────────────────────────────────────────
st.sidebar.markdown("### 📋 Universe Selection")

DEFAULT_TICKERS = SP500Environment.DEFAULT_TICKERS
use_custom = st.sidebar.toggle("커스텀 종목 사용", value=False)

selected_tickers = DEFAULT_TICKERS
ticker_error     = []

if use_custom:
    st.sidebar.caption("쉼표(,) 또는 띄어쓰기로 구분. 예: AAPL, TSLA, NVDA")
    raw_input = st.sidebar.text_area(
        "종목 티커 입력",
        value=", ".join(DEFAULT_TICKERS),
        height=120,
        placeholder="AAPL, MSFT, GOOGL ..."
    )

    # 입력 파싱 — 쉼표/띄어쓰기 모두 허용, 대문자 변환
    parsed = [t.strip().upper() for t in raw_input.replace(",", " ").split() if t.strip()]
    parsed = list(dict.fromkeys(parsed))   # 중복 제거 (순서 유지)

    if len(parsed) < 2:
        st.sidebar.error("종목을 2개 이상 입력해주세요.")
        selected_tickers = DEFAULT_TICKERS
    elif len(parsed) > 30:
        st.sidebar.warning("종목은 최대 30개까지 지원합니다. 앞 30개만 사용합니다.")
        parsed = parsed[:30]
        selected_tickers = parsed
    else:
        # 유효성 검증 버튼
        if st.sidebar.button("🔍 종목 검증", use_container_width=True):
            with st.sidebar:
                with st.spinner("yfinance로 검증 중..."):
                    valid, invalid = validate_tickers(tuple(parsed))
            if invalid:
                st.sidebar.error(f"❌ 유효하지 않은 티커: {', '.join(invalid)}")
            if valid:
                st.sidebar.success(f"✅ 사용 가능: {len(valid)}개")
                st.session_state['validated_tickers'] = valid
            else:
                st.sidebar.error("유효한 종목이 없습니다. 기본 종목으로 대체합니다.")

        # 검증된 종목이 있으면 사용, 없으면 입력값 그대로 시도
        selected_tickers = st.session_state.get('validated_tickers', parsed)
        st.sidebar.caption(f"현재 적용 종목: **{len(selected_tickers)}개**")
else:
    # 기본 종목 사용 시 검증 캐시 초기화
    if 'validated_tickers' in st.session_state:
        del st.session_state['validated_tickers']
    st.sidebar.caption(f"기본 S&P 500 대표 종목 {len(DEFAULT_TICKERS)}개 사용")

# ── 환경 초기화 (선택된 종목으로) ────────────────────────────────────────────
env          = SP500Environment(tickers=selected_tickers)
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 521 else 100

# ── 사이드바: 모델 선택 ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Selection")
model_name = st.sidebar.selectbox(
    "RL Algorithm",
    options=list(AGENT_REGISTRY.keys()),
    help="비교할 DCA 에이전트 알고리즘을 선택합니다."
)
st.sidebar.caption(f"선택: **{model_name}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Parameters")
st.sidebar.caption(f"데이터 최대 사용 가능 기간: {max_episodes}일 (5년치)")
episodes = st.sidebar.slider(
    "Episodes (Trading Days)",
    min_value=100,
    max_value=min(max_episodes, 1200),
    value=min(500, max_episodes),
    step=50,
    help="학습에 사용할 거래일 수. 많을수록 더 많은 시장 패턴 학습 가능하지만 실행 시간 증가."
)
speed = st.sidebar.slider("Execution Speed (sec)", 0.0, 0.5, 0.02, step=0.01)

# ── 타이틀 + Regime 배지 (episodes 확정 후 출력) ──────────────────────────
regime      = env.get_market_regime(min(20 + episodes - 1, len(env.data) - 1))
badge_class = {"Bull": "regime-bull", "Bear": "regime-bear"}.get(regime, "regime-neutral")
st.markdown(
    f'<h1 style="margin-bottom:4px">🏛️ Personalized S&P 500 Performance Terminal'
    f'<span class="{badge_class}">{regime} Market</span></h1>',
    unsafe_allow_html=True
)
st.caption(f"Multi-Model RL Framework  ·  학습 기간: {episodes}일  ·  종목 수: {env.vocab_size}개  ·  데이터: Yahoo Finance 5년")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 RL Hyperparameters")
param_hints = AGENT_PARAMS.get(model_name, {})

st.sidebar.caption(param_hints.get("lr", "학습 속도"))
lr_default = 0.0003 if model_name in PPO_FAMILY else (0.001 if model_name in DQN_FAMILY else 0.005)
lr = st.sidebar.number_input(
    "Learning Rate (α)",
    value=lr_default,
    format="%.4f", step=0.0001
)

st.sidebar.caption(param_hints.get("gamma", "미래 보상 중요도"))
gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.85, step=0.01)

st.sidebar.caption(param_hints.get("eps", "초기 탐색 확률"))
eps = st.sidebar.slider("Exploration (ε)", 0.0, 1.0, 0.5, step=0.05)

st.sidebar.caption("본 학습 전 과거 데이터 반복 학습 횟수")
pretrain = st.sidebar.slider("Market Pre-Train", 0, 5, 2)

# DQN / Double DQN 전용 파라미터
target_update = 20
if model_name in DQN_FAMILY:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 DQN Parameters")
    st.sidebar.caption(param_hints.get("target_update", "Target Network 동기화 주기"))
    target_update = st.sidebar.slider("Target Update Interval", 5, 100, 20, step=5)

clip_eps     = 0.2
rollout_len  = 64
update_epochs = 4
if model_name in PPO_FAMILY:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 PPO Parameters")
    st.sidebar.caption("정책 변화 허용 범위 — 클수록 업데이트 공격적 (0.1~0.3 권장)")
    clip_eps = st.sidebar.slider("Clip Epsilon", 0.05, 0.5, 0.2, step=0.05)
    st.sidebar.caption("몇 step 모아서 한 번 업데이트할지")
    rollout_len = st.sidebar.slider("Rollout Length", 32, 256, 64, step=32)
    st.sidebar.caption("같은 데이터로 반복 학습 횟수 (3~10 권장)")
    update_epochs = st.sidebar.slider("Update Epochs", 2, 10, 4, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 학습 진행")
eps_bar = st.sidebar.empty()
eps_bar.progress(0.0, text="대기 중...")

# ── 사이드바: 모델 저장/불러오기 ────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Model Save / Load")

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 저장 버튼 — Run Evaluation 완료 후 활성화
if st.session_state.get('trained_agent'):
    if st.sidebar.button("💾 현재 모델 저장", use_container_width=True):
        agent_to_save = st.session_state['trained_agent']
        fname = f"{model_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%m%d_%H%M')}"
        ext   = ".npy" if model_name == "Tabular Q-Learning" else ".pt"
        fpath = os.path.join(SAVE_DIR, fname + ext)
        agent_to_save.save(fpath)
        st.sidebar.success(f"✅ 저장 완료: {fname + ext}")
        st.session_state['last_save_path'] = fpath
else:
    st.sidebar.caption("Run Evaluation 완료 후 저장 가능")

# 불러오기 — 저장된 파일 목록에서 선택
saved_files = []
if os.path.exists(SAVE_DIR):
    saved_files = [f for f in os.listdir(SAVE_DIR)
                   if f.endswith(".pt") or f.endswith(".npy")]
    saved_files.sort(reverse=True)   # 최신순

if saved_files:
    selected_file = st.sidebar.selectbox(
        "저장된 모델 선택", ["(선택 안 함)"] + saved_files
    )
    if selected_file != "(선택 안 함)":
        load_path = os.path.join(SAVE_DIR, selected_file)
        meta      = BaseAgent.load_meta(load_path)
        if meta:
            st.sidebar.caption(
                f"모델: {meta.get('model_name', '?')}  |  "
                f"저장: {meta.get('saved_at', '?')[:16]}  |  "
                f"종목: {len(meta.get('tickers', []))}개"
            )
        if st.sidebar.button("📂 모델 불러오기", use_container_width=True):
            st.session_state['load_path']  = load_path
            st.session_state['load_meta']  = meta
            st.sidebar.success(f"✅ 불러오기 준비: {selected_file}")
else:
    st.sidebar.caption("저장된 모델이 없습니다.")

# ── session state ─────────────────────────────────────────────────────
if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []
if 'has_run' not in st.session_state:
    st.session_state.has_run = False

# ── 탭 ───────────────────────────────────────────────────────────────
tab_live, tab_analysis, tab_history, tab_hpo, tab_race, tab_rec = st.tabs(
    ["📈 Live Monitor", "📊 Analysis", "🏆 Trial History",
     "🔬 HPO", "⚔️ Model Race", "🎯 Recommender"]
)

with tab_live:
    if not st.session_state.has_run:
        st.info("👈 사이드바에서 알고리즘과 파라미터를 설정한 뒤 **Run Evaluation** 을 눌러 시작하세요.")

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines',
        name='<b>Vanilla RL</b>',             line=dict(color='#e05050', width=2.5)))
    fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines',
        name=f'<b>DCA ({model_name})</b>',    line=dict(color='#4a90d9', width=3)))
    fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines',
        name='<b>S&P 500 Index</b>',           line=dict(color='#2ea84a', width=2, dash='dot')))
    fig_main.update_layout(
        title=dict(text=f"<b>Cumulative Return — {model_name}</b>", font=dict(size=22)),
        xaxis=dict(title="<b>Trading Days</b>", showgrid=True),
        yaxis=dict(title="<b>Cumulative Return (%)</b>", showgrid=True),
        legend=dict(font=dict(size=13), x=0.01, y=0.99, bgcolor='rgba(128,128,128,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=460, margin=dict(t=50, b=40)
    )
    fig_main.add_hline(y=0, line_width=1.5, line_color="rgba(150,150,150,0.4)")
    chart_view = st.empty()
    chart_view.plotly_chart(fig_main, use_container_width=True)

    col1, col2, col3    = st.columns(3)
    m_u, m_s, m_b       = col1.empty(), col2.empty(), col3.empty()
    st.markdown("---")
    st.markdown("##### 🔬 Learning Monitor")
    cl1, cl2, cl3, cl4  = st.columns(4)
    m_win_u, m_win_s, m_q_u, m_q_s = cl1.empty(), cl2.empty(), cl3.empty(), cl4.empty()

with tab_analysis:
    if not st.session_state.has_run:
        st.info("Run Evaluation 실행 후 분석 결과가 표시됩니다.")
    sharpe_area      = st.empty()
    analysis_header  = st.empty()
    col_tbl, col_bar = st.columns([1.2, 1])
    tbl_view         = col_tbl.empty()
    bar_view         = col_bar.empty()
    # 백테스팅 리포트 영역
    bt_area          = st.empty()

with tab_history:
    if not st.session_state.trial_history:
        st.info("여러 번 실행하면 시도별 성과 분포가 여기에 누적됩니다.")
    history_area = st.empty()

# ── helpers ──────────────────────────────────────────────────────────
def style_df(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #e05050; font-weight: bold;'
    return 'font-weight: bold;'


def render_backtest_report(log_data, h_u, h_s, h_b, model_name, env, start_step=20):
    """
    백테스팅 리포트 생성
    - 드로우다운 차트
    - 월별 수익률 히트맵
    - 주요 성과 요약 테이블
    """
    with bt_area.container():
        st.markdown("---")
        st.markdown("#### 📋 Backtest Report")

        df = pd.DataFrame(log_data).set_index("Day")

        # 실제 날짜 인덱스 생성
        dates = env.data.index[start_step: start_step + len(df)]
        df.index = pd.to_datetime(dates)

        # ── 1. 드로우다운 차트 ─────────────────────────────────────────
        def calc_drawdown(cum_rets):
            arr  = np.array(cum_rets[1:])   # 첫 0 제외
            peak = np.maximum.accumulate(arr)
            return arr - peak

        dd_u = calc_drawdown(h_u)
        dd_s = calc_drawdown(h_s)
        dd_b = calc_drawdown(h_b)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df.index, y=dd_u, mode='lines', fill='tozeroy',
            name='Vanilla RL', line=dict(color='#e05050', width=1.5),
            fillcolor='rgba(224,80,80,0.15)'
        ))
        fig_dd.add_trace(go.Scatter(
            x=df.index, y=dd_s, mode='lines', fill='tozeroy',
            name=f'DCA ({model_name})', line=dict(color='#4a90d9', width=1.5),
            fillcolor='rgba(74,144,217,0.15)'
        ))
        fig_dd.add_trace(go.Scatter(
            x=df.index, y=dd_b, mode='lines',
            name='S&P 500', line=dict(color='#2ea84a', width=1.5, dash='dot')
        ))
        fig_dd.update_layout(
            title="<b>Drawdown</b>",
            yaxis_title="Drawdown (%)",
            xaxis_title="Date",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.01)
        )
        fig_dd.add_hline(y=0, line_width=1, line_color="rgba(150,150,150,0.3)")
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── 2. 월별 수익률 히트맵 ──────────────────────────────────────
        st.markdown("##### 월별 수익률 — DCA vs Vanilla")

        df['dca_ret']    = df['DCA Ret(%)']
        df['vanilla_ret']= df['Vanilla Ret(%)']
        df['year']       = df.index.year
        df['month']      = df.index.month

        monthly_dca     = df.groupby(['year', 'month'])['dca_ret'].sum().unstack(fill_value=0)
        monthly_vanilla = df.groupby(['year', 'month'])['vanilla_ret'].sum().unstack(fill_value=0)

        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']

        mc1, mc2 = st.columns(2)
        for col, data, title, color in [
            (mc1, monthly_dca,     f"DCA ({model_name})", "RdYlGn"),
            (mc2, monthly_vanilla, "Vanilla RL",          "RdYlGn"),
        ]:
            data.columns = [month_labels[m-1] for m in data.columns]
            fig_hm = go.Figure(go.Heatmap(
                z=data.values,
                x=data.columns.tolist(),
                y=[str(y) for y in data.index.tolist()],
                colorscale=color,
                zmid=0,
                text=[[f"{v:.1f}%" for v in row] for row in data.values],
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=False,
            ))
            fig_hm.update_layout(
                title=f"<b>{title}</b>",
                height=max(180, len(data) * 45 + 60),
                margin=dict(t=40, b=20, l=40, r=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            col.plotly_chart(fig_hm, use_container_width=True)

        # ── 3. 주요 성과 요약 테이블 ───────────────────────────────────
        st.markdown("##### 📐 성과 요약")

        def ann_return(cum_rets):
            total = cum_rets[-1] / 100
            n_days = len(cum_rets) - 1
            if n_days <= 0: return 0.0
            return ((1 + total) ** (252 / n_days) - 1) * 100

        def ann_vol(rets):
            return float(np.std(rets) * np.sqrt(252))

        dca_rets     = np.array([log['DCA Ret(%)']     for log in log_data])
        vanilla_rets = np.array([log['Vanilla Ret(%)'] for log in log_data])
        spy_rets     = (np.diff(h_b))

        summary = {
            "지표": [
                "누적 수익률 (%)",
                "연환산 수익률 (%)",
                "연환산 변동성 (%)",
                "Sharpe Ratio",
                "MDD (%)",
                "Win Rate (%)",
                "최대 일수익 (%)",
                "최대 일손실 (%)",
            ],
            f"DCA ({model_name})": [
                f"{h_s[-1]:.2f}",
                f"{ann_return(h_s):.2f}",
                f"{ann_vol(dca_rets):.2f}",
                f"{(np.mean(dca_rets - 0.0198) / (np.std(dca_rets) + 1e-10) * np.sqrt(252)):.3f}",
                f"{min(dd_s):.2f}",
                f"{(dca_rets > 0).mean() * 100:.1f}",
                f"{dca_rets.max():.2f}",
                f"{dca_rets.min():.2f}",
            ],
            "Vanilla RL": [
                f"{h_u[-1]:.2f}",
                f"{ann_return(h_u):.2f}",
                f"{ann_vol(vanilla_rets):.2f}",
                f"{(np.mean(vanilla_rets - 0.0198) / (np.std(vanilla_rets) + 1e-10) * np.sqrt(252)):.3f}",
                f"{min(dd_u):.2f}",
                f"{(vanilla_rets > 0).mean() * 100:.1f}",
                f"{vanilla_rets.max():.2f}",
                f"{vanilla_rets.min():.2f}",
            ],
            "S&P 500": [
                f"{h_b[-1]:.2f}",
                f"{ann_return(h_b):.2f}",
                f"{ann_vol(spy_rets):.2f}",
                f"{(np.mean(spy_rets - 0.0198) / (np.std(spy_rets) + 1e-10) * np.sqrt(252)):.3f}",
                f"{min(dd_b):.2f}",
                f"{(spy_rets > 0).mean() * 100:.1f}",
                f"{spy_rets.max():.2f}",
                f"{spy_rets.min():.2f}",
            ],
        }

        st.dataframe(
            pd.DataFrame(summary).set_index("지표"),
            use_container_width=True
        )

def render_history():
    if not st.session_state.trial_history: return
    df_h = pd.DataFrame(st.session_state.trial_history)
    with history_area.container():
        st.markdown("#### 📊 Consolidated Performance Benchmarks")

        fig_box = go.Figure()
        for col, color, name in [
            ('Vanilla Final (%)',   '#e05050', 'Vanilla RL'),
            ('ADAPTIVE Final (%)', '#4a90d9', 'DCA Agent'),
        ]:
            if col in df_h.columns:
                fig_box.add_trace(go.Box(y=df_h[col], name=name, line=dict(color=color)))

        avg_spy = df_h['SPY Final (%)'].mean()
        fig_box.add_hline(y=avg_spy, line_dash="dot", line_color="#2ea84a",
                          annotation_text=f"S&P 500 Avg: {avg_spy:.2f}%")
        fig_box.update_layout(title="<b>Final Return Distribution</b>", height=400,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

        if 'DCA Sharpe' in df_h.columns:
            fig_sh = go.Figure()
            fig_sh.add_trace(go.Box(y=df_h['Vanilla Sharpe'], name='Vanilla RL',   line=dict(color='#e05050')))
            fig_sh.add_trace(go.Box(y=df_h['DCA Sharpe'],     name='DCA Agent',    line=dict(color='#4a90d9')))
            fig_sh.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Sharpe = 0")
            fig_sh.update_layout(title="<b>Sharpe Ratio Distribution</b>", height=360,
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sh, use_container_width=True)

        # 모델명 컬럼 추가 표시
        fmt_cols = {c: "{:.2f}" for c in df_h.columns if c not in ("Trial", "Model")}
        st.dataframe(df_h.set_index("Trial").style.format(fmt_cols), use_container_width=True)

render_history()

# ── Model Race 탭 ─────────────────────────────────────────────────────────
with tab_race:
    st.markdown("#### ⚔️ Model Race")
    st.caption("같은 종목·기간·파라미터로 모든 모델을 동시에 학습시키고 성능을 비교합니다.")

    # 설정
    rc1, rc2, rc3 = st.columns(3)
    race_episodes = rc1.slider("Episodes", 100, min(max_episodes, 1200), 300, step=50)
    race_pretrain = rc2.slider("Pretrain", 0, 3, 1)
    race_speed    = rc3.slider("Speed (sec)", 0.0, 0.2, 0.0, step=0.01)

    st.markdown("---")
    st.markdown("**공통 하이퍼파라미터**")
    rh1, rh2, rh3 = st.columns(3)
    race_lr    = rh1.number_input("Learning Rate", value=0.001, format="%.4f", step=0.0001)
    race_gamma = rh2.slider("Gamma (γ)", 0.5, 0.99, 0.85, step=0.01)
    race_eps   = rh3.slider("Epsilon (ε)", 0.1, 1.0, 0.5, step=0.05)

    # 레이스에 참가할 모델 선택
    st.markdown("**참가 모델 선택**")
    all_model_names = list(AGENT_REGISTRY.keys())
    race_models = st.multiselect(
        "모델 선택 (2개 이상)",
        options=all_model_names,
        default=all_model_names,
        help="선택한 모델만 레이스에 참가합니다."
    )

    # 결과 영역
    race_prog    = st.empty()
    race_chart   = st.empty()
    race_result  = st.empty()

    if st.button("🏁 Race 시작", type="primary", use_container_width=True,
                 disabled=len(race_models) < 2):

        # 색상 팔레트
        COLORS = ['#e05050','#4a90d9','#2ea84a','#f5a623',
                  '#9b59b6','#1abc9c','#e74c3c','#3498db']

        # 레이스 차트 초기화
        fig_race = go.Figure()
        fig_race.add_hline(y=0, line_width=1, line_color="rgba(150,150,150,0.3)")
        fig_race.update_layout(
            title="<b>Model Race — 누적 수익률 실시간 비교</b>",
            xaxis_title="Trading Days", yaxis_title="Cumulative Return (%)",
            height=480, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(size=12), x=0.01, y=0.99,
                        bgcolor='rgba(128,128,128,0.1)')
        )
        for idx, mn in enumerate(race_models):
            fig_race.add_trace(go.Scatter(
                x=[], y=[], mode='lines',
                name=f'<b>{mn}</b>',
                line=dict(color=COLORS[idx % len(COLORS)], width=2.5)
            ))
        # SPY 기준선 추가
        fig_race.add_trace(go.Scatter(
            x=[], y=[], mode='lines',
            name='<b>S&P 500</b>',
            line=dict(color='#888888', width=1.5, dash='dot')
        ))

        # 에이전트 초기화
        agents = {}
        for mn in race_models:
            AgentClass = AGENT_REGISTRY[mn]
            kwargs = dict(env=env, use_constraints=True,
                         lr=race_lr, gamma=race_gamma, eps=race_eps)
            if mn in DQN_FAMILY:
                kwargs["target_update"] = 20
            if mn in PPO_FAMILY:
                kwargs.update(clip_eps=0.2, rollout_len=64, update_epochs=4)
            try:
                agents[mn] = AgentClass(**kwargs)
            except ImportError as e:
                st.warning(f"⚠️ {mn} 건너뜀: {e}")

        if len(agents) < 2:
            st.error("실행 가능한 모델이 2개 미만입니다.")
            st.stop()

        # Pretrain
        if race_pretrain > 0:
            with st.spinner(f"Warming-up ({race_pretrain} epochs)..."):
                for _ in range(race_pretrain):
                    for i in range(20, 20 + race_episodes):
                        for mn, agent in agents.items():
                            _, _, r, s, a, _, _ = agent.select_action(i)
                            agent.learn(s, a, r, min(i + 1, len(env.data) - 1))
                            agent.decay_epsilon(i - 20, race_episodes)

        # 누적 수익률 초기화
        cum_rets  = {mn: [0.0] for mn in agents}
        cum_spy   = [0.0]
        steps_log = [0]
        UPDATE_INTERVAL = 5

        for i in range(20, 20 + race_episodes):
            step_num = i - 19

            for mn, agent in agents.items():
                _, _, r, s, a, _, raw = agent.select_action(i)
                agent.learn(s, a, r, min(i + 1, len(env.data) - 1))
                agent.decay_epsilon(i - 20, race_episodes)
                cum_rets[mn].append(cum_rets[mn][-1] + raw)

            sc  = float(env.data['SPY'].iloc[i])
            sn  = float(env.data['SPY'].iloc[i + 1])
            cum_spy.append(cum_spy[-1] + ((sn - sc) / sc) * 100)
            steps_log.append(step_num)

            # 진행 바
            race_prog.progress(
                step_num / race_episodes,
                text=f"Step {step_num}/{race_episodes} — "
                     + "  ".join([f"{mn}: {cum_rets[mn][-1]:.1f}%" for mn in agents])
            )

            # 차트 업데이트
            if step_num % UPDATE_INTERVAL == 0 or step_num == race_episodes:
                for idx, mn in enumerate(agents):
                    fig_race.data[idx].x = steps_log
                    fig_race.data[idx].y = cum_rets[mn]
                fig_race.data[-1].x = steps_log
                fig_race.data[-1].y = cum_spy
                race_chart.plotly_chart(fig_race, use_container_width=True)

            if race_speed > 0:
                time.sleep(race_speed)

        race_prog.progress(1.0, text="✅ Race 완료!")

        # 결과 테이블
        with race_result.container():
            st.markdown("---")
            st.markdown("#### 🏆 Race 결과")

            result_rows = []
            spy_final   = cum_spy[-1]
            best_ret    = max(cum_rets[mn][-1] for mn in agents)

            for rank, (mn, color) in enumerate(
                sorted(agents.items(),
                       key=lambda x: cum_rets[x[0]][-1], reverse=True),
                start=1
            ):
                rets      = np.diff(cum_rets[mn])
                sharpe    = (np.mean(rets - 0.0198) /
                             (np.std(rets) + 1e-10) * np.sqrt(252))
                arr       = np.array(cum_rets[mn])
                peak      = np.maximum.accumulate(arr)
                mdd       = float(np.min(arr - peak))
                win_rate  = agents[mn].get_win_rate()
                final_ret = cum_rets[mn][-1]

                result_rows.append({
                    "순위":         f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else str(rank)}",
                    "모델":         mn,
                    "누적 수익 (%)": round(final_ret, 2),
                    "vs S&P 500":   round(final_ret - spy_final, 2),
                    "Sharpe":       round(sharpe, 3),
                    "MDD (%)":      round(mdd, 2),
                    "Win Rate (%)": round(win_rate, 1),
                })

            df_race = pd.DataFrame(result_rows).set_index("순위")

            # 수익률 기준 색상 스타일
            def color_ret(val):
                try:
                    v = float(val)
                    return 'color: #2ea84a; font-weight:bold' if v > 0 else \
                           'color: #e05050; font-weight:bold' if v < 0 else ''
                except:
                    return ''

            st.dataframe(
                df_race.style
                       .map(color_ret, subset=["누적 수익 (%)","vs S&P 500","MDD (%)"])
                       .format({
                           "누적 수익 (%)": "{:.2f}",
                           "vs S&P 500":   "{:+.2f}",
                           "Sharpe":        "{:.3f}",
                           "MDD (%)":       "{:.2f}",
                           "Win Rate (%)":  "{:.1f}",
                       }),
                use_container_width=True
            )

            best_model = result_rows[0]['모델']
            st.success(
                f"🏆 이 종목·기간에서 가장 좋은 모델: **{best_model}**  "
                f"(누적 수익: {result_rows[0]['누적 수익 (%)']:.2f}%,  "
                f"Sharpe: {result_rows[0]['Sharpe']:.3f})"
            )

            # session_state에 저장
            st.session_state['race_result'] = result_rows

    # 이전 레이스 결과 표시
    elif st.session_state.get('race_result'):
        st.markdown("**이전 Race 결과**")
        st.dataframe(
            pd.DataFrame(st.session_state['race_result']).set_index("순위"),
            use_container_width=True
        )
    st.markdown("#### 🔬 Hyperparameter Optimization")
    st.caption("optuna Bayesian Optimization으로 선택 모델 또는 전체 모델의 최적 파라미터를 자동 탐색합니다.")

    if not OPTUNA_AVAILABLE:
        st.error("HPO를 사용하려면 optuna를 설치하세요: `pip install optuna`")
    else:
        # ── 설정값 가이드 ──────────────────────────────────────────────────
        with st.expander("⚙️ 설정값 가이드 — 처음이라면 여기를 먼저 확인하세요", expanded=False):
            g = HPO_SETTINGS_GUIDE
            st.markdown("**Trial 횟수** — optuna가 파라미터 조합을 몇 번 시도할지")
            gc1, gc2, gc3 = st.columns(3)
            gc1.info(f"🐇 **빠름**: {g['n_trials']['fast']['value']}회\n\n{g['n_trials']['fast']['desc']}")
            gc2.success(f"⚖️ **권장**: {g['n_trials']['balanced']['value']}회\n\n{g['n_trials']['balanced']['desc']}")
            gc3.warning(f"🔬 **정밀**: {g['n_trials']['thorough']['value']}회\n\n{g['n_trials']['thorough']['desc']}")

            st.markdown("**학습 Episode 수** — trial 하나당 에이전트를 몇 일치 데이터로 학습할지")
            ec1, ec2, ec3 = st.columns(3)
            ec1.info(f"🐇 **빠름**: {g['episodes']['fast']['value']}일\n\n{g['episodes']['fast']['desc']}")
            ec2.success(f"⚖️ **권장**: {g['episodes']['balanced']['value']}일\n\n{g['episodes']['balanced']['desc']}")
            ec3.warning(f"🔬 **정밀**: {g['episodes']['thorough']['value']}일\n\n{g['episodes']['thorough']['desc']}")

            st.markdown("**Pretrain 횟수** — 본 학습 전 데이터를 미리 몇 번 돌릴지")
            pc1, pc2, pc3 = st.columns(3)
            pc1.info(f"🐇 **없음**: {g['pretrain']['fast']['value']}회\n\n{g['pretrain']['fast']['desc']}")
            pc2.success(f"⚖️ **권장**: {g['pretrain']['balanced']['value']}회\n\n{g['pretrain']['balanced']['desc']}")
            pc3.warning(f"🔬 **정밀**: {g['pretrain']['thorough']['value']}회\n\n{g['pretrain']['thorough']['desc']}")

            st.markdown("---")
            st.markdown("**예상 소요 시간 (Tabular Q 기준)**")
            time_data = {
                "설정": ["빠름 (10회×100일)", "권장 (20회×200일)", "정밀 (50회×500일)"],
                "단일 모델": ["~1분", "~5분", "~30분"],
                "전체 모델 (5개)": ["~5분", "~25분", "~2.5시간"],
            }
            st.dataframe(pd.DataFrame(time_data).set_index("설정"), use_container_width=True)
            st.caption("신경망 모델(DQN 계열, PPO)은 Tabular Q 대비 약 2~3배 더 소요됩니다.")

        st.markdown("---")

        # ── HPO 설정 입력 ─────────────────────────────────────────────────
        hc1, hc2, hc3 = st.columns(3)
        n_trials = hc1.number_input("Trial 횟수", min_value=5, max_value=100, value=20, step=5)
        hpo_eps  = hc2.number_input("학습 Episode 수", min_value=50, max_value=500, value=200, step=50)
        hpo_pre  = hc3.number_input("Pretrain 횟수", min_value=0, max_value=3, value=1, step=1)

        st.markdown("---")

        # ── 모드 선택: 단일 vs 전체 ──────────────────────────────────────
        hpo_mode = st.radio(
            "최적화 대상",
            ["현재 선택 모델만", "전체 모델 비교"],
            horizontal=True,
            help="'전체 모델 비교'는 시간이 더 걸리지만 어떤 모델이 이 종목에 가장 적합한지 한 번에 파악 가능합니다."
        )

        if hpo_mode == "현재 선택 모델만":
            st.markdown(f"**탐색 대상 파라미터 — {model_name}**")
            space = SEARCH_SPACE.get(model_name, {})
            if space:
                rows = [{"파라미터": name, "타입": ptype,
                         "최솟값": lo, "최댓값": hi, "로그 스케일": "✓" if log else ""}
                        for name, (ptype, lo, hi, log) in space.items()]
                st.dataframe(pd.DataFrame(rows).set_index("파라미터"),
                             use_container_width=True, height=200)

        else:
            st.markdown("**전체 모델 탐색 공간**")
            all_rows = []
            for mn, sp in SEARCH_SPACE.items():
                for pname, (ptype, lo, hi, log) in sp.items():
                    all_rows.append({"모델": mn, "파라미터": pname,
                                     "최솟값": lo, "최댓값": hi})
            st.dataframe(pd.DataFrame(all_rows).set_index("모델"),
                         use_container_width=True, height=260)

        hpo_progress = st.empty()
        hpo_chart    = st.empty()
        hpo_result   = st.empty()

        btn_label = f"🚀 HPO 시작 — {model_name}" if hpo_mode == "현재 선택 모델만" else "🚀 전체 모델 HPO 시작"
        if st.button(btn_label, type="primary", use_container_width=True):

            # ── 단일 모델 HPO ───────────────────────────────────────────
            if hpo_mode == "현재 선택 모델만":
                AgentClass = AGENT_REGISTRY[model_name]
                trial_log  = []
                best_so_far = [-999.0]

                prog_bar = hpo_progress.progress(0.0, text="HPO 준비 중...")
                fig_hpo  = go.Figure()
                fig_hpo.add_trace(go.Scatter(x=[], y=[], mode='lines+markers',
                    name='Trial Sharpe', line=dict(color='#4a90d9', width=2)))
                fig_hpo.add_trace(go.Scatter(x=[], y=[], mode='lines',
                    name='Best Sharpe', line=dict(color='#2ea84a', width=2, dash='dash')))
                fig_hpo.update_layout(
                    title=f"<b>HPO 진행 — {model_name}</b>",
                    xaxis_title="Trial", yaxis_title="Sharpe Ratio",
                    height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                )

                def single_callback(trial_num, best_val, params):
                    trial_log.append((trial_num, best_val))
                    best_so_far[0] = max(best_so_far[0], best_val)
                    prog_bar.progress(min(trial_num / n_trials, 1.0),
                        text=f"Trial {trial_num}/{n_trials}  |  Best Sharpe: {best_so_far[0]:.3f}")
                    xs = [t[0] for t in trial_log]
                    ys = [t[1] for t in trial_log]
                    fig_hpo.data[0].x, fig_hpo.data[0].y = xs, ys
                    fig_hpo.data[1].x, fig_hpo.data[1].y = xs, [max(ys[:i+1]) for i in range(len(ys))]
                    hpo_chart.plotly_chart(fig_hpo, use_container_width=True)

                with st.spinner(f"{model_name} 최적화 중..."):
                    best_params, study = optimize(
                        env, model_name, AgentClass,
                        episodes=int(hpo_eps), n_trials=int(n_trials),
                        pretrain=int(hpo_pre), callback=single_callback,
                    )

                prog_bar.progress(1.0, text=f"✅ 완료!  Best Sharpe: {study.best_value:.3f}")

                with hpo_result.container():
                    st.markdown("---")
                    st.markdown("#### 🏆 최적 파라미터 결과")
                    rc1, rc2 = st.columns(2)
                    rc1.metric("Best Sharpe Ratio", f"{study.best_value:.4f}")
                    rc1.metric("완료 Trial 수", len(study.trials))
                    result_rows = [{"파라미터": k, "최적값": round(v, 6)}
                                   for k, v in best_params.items()]
                    rc2.dataframe(pd.DataFrame(result_rows).set_index("파라미터"),
                                  use_container_width=True)
                    st.info("💡 위 파라미터를 사이드바에 입력한 뒤 Run Evaluation을 실행하세요.")
                    st.session_state['hpo_best_params'] = best_params
                    st.session_state['hpo_model']       = model_name
                    st.session_state['hpo_sharpe']      = study.best_value

            # ── 전체 모델 HPO ───────────────────────────────────────────
            else:
                prog_bar    = hpo_progress.progress(0.0, text="전체 모델 HPO 준비 중...")
                status_area = hpo_chart.empty()
                all_results = {}   # {model_name: best_sharpe}

                fig_compare = go.Figure()
                fig_compare.update_layout(
                    title="<b>모델별 Best Sharpe Ratio 비교</b>",
                    xaxis_title="모델", yaxis_title="Best Sharpe",
                    height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                )

                total_models = len(SEARCH_SPACE)

                def all_callback(model_name, model_idx, total_models,
                                 trial_num, total_trials, best_sharpe, best_params):
                    overall = ((model_idx - 1) * total_trials + trial_num) / (total_models * total_trials)
                    prog_bar.progress(min(overall, 1.0),
                        text=f"[{model_idx}/{total_models}] {model_name}  "
                             f"Trial {trial_num}/{total_trials}  |  Best Sharpe: {best_sharpe:.3f}")

                    # 완료된 모델의 best sharpe 업데이트
                    all_results[model_name] = best_sharpe
                    if len(all_results) > 0:
                        mn_list = list(all_results.keys())
                        sv_list = list(all_results.values())
                        colors  = ['#2ea84a' if s == max(sv_list) else '#4a90d9' for s in sv_list]
                        fig_compare.data = []
                        fig_compare.add_trace(go.Bar(
                            x=mn_list, y=sv_list,
                            marker_color=colors,
                            text=[f"{s:.3f}" for s in sv_list],
                            textposition='outside',
                        ))
                        status_area.plotly_chart(fig_compare, use_container_width=True)

                with st.spinner("전체 모델 최적화 중... (시간이 걸릴 수 있습니다)"):
                    all_hpo_results = optimize_all_models(
                        env, AGENT_REGISTRY,
                        episodes=int(hpo_eps), n_trials=int(n_trials),
                        pretrain=int(hpo_pre), callback=all_callback,
                    )

                prog_bar.progress(1.0, text="✅ 전체 모델 HPO 완료!")

                # 결과 테이블
                with hpo_result.container():
                    st.markdown("---")
                    st.markdown("#### 🏆 전체 모델 HPO 결과 비교")

                    summary_rows = []
                    best_model   = max(all_hpo_results, key=lambda m: all_hpo_results[m]["best_sharpe"])
                    for mn, res in all_hpo_results.items():
                        row = {"모델": mn,
                               "Best Sharpe": round(res["best_sharpe"], 4),
                               "추천": "★ 최적" if mn == best_model else ""}
                        row.update({f"  {k}": round(v, 5)
                                    for k, v in res["best_params"].items()})
                        summary_rows.append(row)

                    df_summary = pd.DataFrame(summary_rows).set_index("모델")
                    st.dataframe(df_summary, use_container_width=True)

                    st.success(f"🏆 이 종목에 가장 적합한 모델: **{best_model}**  "
                               f"(Sharpe: {all_hpo_results[best_model]['best_sharpe']:.4f})")
                    st.info("💡 사이드바에서 추천 모델을 선택하고 위 파라미터를 입력한 뒤 Run Evaluation을 실행하세요.")

                    st.session_state['hpo_all_results'] = all_hpo_results
                    st.session_state['hpo_best_model']  = best_model

        # 이전 결과 표시
        elif 'hpo_best_params' in st.session_state and hpo_mode == "현재 선택 모델만":
            prev_model  = st.session_state.get('hpo_model', '')
            prev_sharpe = st.session_state.get('hpo_sharpe', 0)
            prev_params = st.session_state['hpo_best_params']
            st.markdown(f"**이전 HPO 결과 — {prev_model}**  (Sharpe: {prev_sharpe:.4f})")
            result_rows = [{"파라미터": k, "최적값": round(v, 6)}
                           for k, v in prev_params.items()]
            st.dataframe(pd.DataFrame(result_rows).set_index("파라미터"),
                         use_container_width=True)

        elif 'hpo_all_results' in st.session_state and hpo_mode == "전체 모델 비교":
            all_hpo_results = st.session_state['hpo_all_results']
            best_model      = st.session_state.get('hpo_best_model', '')
            st.markdown("**이전 전체 모델 HPO 결과**")
            summary_rows = []
            for mn, res in all_hpo_results.items():
                row = {"모델": mn, "Best Sharpe": round(res["best_sharpe"], 4),
                       "추천": "★ 최적" if mn == best_model else ""}
                row.update({f"  {k}": round(v, 5) for k, v in res["best_params"].items()})
                summary_rows.append(row)
            st.dataframe(pd.DataFrame(summary_rows).set_index("모델"), use_container_width=True)
            if best_model:
                st.success(f"🏆 추천 모델: **{best_model}**")

# ── Recommender 탭 ────────────────────────────────────────────────────────
with tab_rec:
    st.markdown("#### 🎯 종목별 최적 모델 추천")
    st.caption(
        "Model Race와 HPO 결과를 종합해서 현재 선택된 종목에 가장 적합한 모델을 추천합니다."
    )

    # ── 점수 계산 로직 ──────────────────────────────────────────────────
    def compute_scores():
        """
        Race 결과와 HPO 결과를 종합한 모델 점수 계산
        점수 = 누적수익 × 0.4 + Sharpe × 30 × 0.4 + Win Rate × 0.2
        (HPO 결과가 있으면 Sharpe를 HPO 기준으로 보정)
        """
        scores = {}

        # Race 결과 활용
        race_result = st.session_state.get('race_result', [])
        for row in race_result:
            mn = row['모델']
            ret      = row.get('누적 수익 (%)', 0)
            sharpe   = row.get('Sharpe', 0)
            win_rate = row.get('Win Rate (%)', 0)
            scores[mn] = {
                'race_ret':      ret,
                'race_sharpe':   sharpe,
                'race_win_rate': win_rate,
                'hpo_sharpe':    None,
                'hpo_params':    None,
            }

        # HPO 결과로 보정
        hpo_all = st.session_state.get('hpo_all_results', {})
        for mn, res in hpo_all.items():
            if mn not in scores:
                scores[mn] = {
                    'race_ret': 0, 'race_sharpe': 0, 'race_win_rate': 0,
                    'hpo_sharpe': None, 'hpo_params': None,
                }
            scores[mn]['hpo_sharpe'] = res['best_sharpe']
            scores[mn]['hpo_params'] = res['best_params']

        # 단일 HPO 결과도 반영
        hpo_model  = st.session_state.get('hpo_model')
        hpo_sharpe = st.session_state.get('hpo_sharpe')
        hpo_params = st.session_state.get('hpo_best_params')
        if hpo_model and hpo_sharpe is not None:
            if hpo_model not in scores:
                scores[hpo_model] = {
                    'race_ret': 0, 'race_sharpe': 0, 'race_win_rate': 0,
                    'hpo_sharpe': None, 'hpo_params': None,
                }
            scores[hpo_model]['hpo_sharpe'] = hpo_sharpe
            scores[hpo_model]['hpo_params'] = hpo_params

        if not scores:
            return None

        # 최종 점수 계산
        for mn in scores:
            s = scores[mn]
            # HPO Sharpe가 있으면 우선 사용, 없으면 Race Sharpe 사용
            sharpe_score = s['hpo_sharpe'] if s['hpo_sharpe'] is not None else s['race_sharpe']
            s['total_score'] = (
                s['race_ret']      * 0.40 +
                sharpe_score * 30  * 0.40 +
                s['race_win_rate'] * 0.20
            )

        return scores

    scores = compute_scores()

    if scores is None:
        st.info(
            "추천을 받으려면 아래 중 하나를 먼저 실행하세요.\n\n"
            "- **⚔️ Model Race** 탭에서 Race 실행\n"
            "- **🔬 HPO** 탭에서 전체 모델 HPO 실행"
        )
    else:
        # 점수 기준 정렬
        sorted_models = sorted(scores.items(),
                               key=lambda x: x[1]['total_score'], reverse=True)
        best_mn, best_s = sorted_models[0]

        # ── 1. 추천 결과 카드 ──────────────────────────────────────────
        st.markdown("---")
        ticker_str = ", ".join(env.tickers[:5]) + (
            f" 외 {len(env.tickers)-5}개" if len(env.tickers) > 5 else ""
        )
        st.markdown(f"##### 현재 종목: `{ticker_str}`")

        # 1위 모델 강조 카드
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.success(f"🏆 **추천 모델**\n\n**{best_mn}**")
        cc2.metric("종합 점수",   f"{best_s['total_score']:.2f}")
        cc3.metric("누적 수익",   f"{best_s['race_ret']:.2f}%" if best_s['race_ret'] else "N/A")
        sharpe_disp = best_s['hpo_sharpe'] if best_s['hpo_sharpe'] is not None else best_s['race_sharpe']
        cc4.metric("Sharpe",      f"{sharpe_disp:.3f}")

        # 최적 파라미터 표시
        if best_s['hpo_params']:
            st.markdown(f"**{best_mn} 최적 파라미터** (HPO 기준)")
            param_rows = [{"파라미터": k, "최적값": round(v, 6)}
                          for k, v in best_s['hpo_params'].items()]
            st.dataframe(pd.DataFrame(param_rows).set_index("파라미터"),
                         use_container_width=True)
        st.markdown("---")

        # ── 2. 전체 모델 순위 테이블 ───────────────────────────────────
        st.markdown("##### 전체 모델 종합 순위")

        rank_rows = []
        MEDALS = ['🥇','🥈','🥉']
        for rank, (mn, s) in enumerate(sorted_models, start=1):
            sharpe_val = s['hpo_sharpe'] if s['hpo_sharpe'] is not None else s['race_sharpe']
            rank_rows.append({
                "순위":       MEDALS[rank-1] if rank <= 3 else str(rank),
                "모델":       mn,
                "종합 점수":  round(s['total_score'], 2),
                "누적 수익 (%)": round(s['race_ret'], 2) if s['race_ret'] else "-",
                "Sharpe":     round(sharpe_val, 3) if sharpe_val else "-",
                "Win Rate (%)": round(s['race_win_rate'], 1) if s['race_win_rate'] else "-",
                "HPO 완료":   "✓" if s['hpo_params'] else "",
            })

        st.dataframe(
            pd.DataFrame(rank_rows).set_index("순위"),
            use_container_width=True
        )

        # ── 3. 레이더 차트 — 모델별 강점 시각화 ───────────────────────
        if len(sorted_models) >= 3:
            st.markdown("---")
            st.markdown("##### 모델별 강점 비교")

            categories = ['수익률', 'Sharpe', 'Win Rate', '점수']

            # 각 지표 정규화 (0~1)
            all_rets   = [s['race_ret']      for _, s in sorted_models]
            all_sharpe = [(s['hpo_sharpe'] if s['hpo_sharpe'] is not None
                           else s['race_sharpe']) for _, s in sorted_models]
            all_wr     = [s['race_win_rate'] for _, s in sorted_models]
            all_score  = [s['total_score']   for _, s in sorted_models]

            def norm(vals):
                mn_v, mx_v = min(vals), max(vals)
                if mx_v == mn_v: return [0.5] * len(vals)
                return [(v - mn_v) / (mx_v - mn_v) for v in vals]

            n_rets   = norm(all_rets)
            n_sharpe = norm(all_sharpe)
            n_wr     = norm(all_wr)
            n_score  = norm(all_score)

            RADAR_COLORS = ['#e05050','#4a90d9','#2ea84a','#f5a623',
                            '#9b59b6','#1abc9c']

            fig_radar = go.Figure()
            for idx, (mn, _) in enumerate(sorted_models[:6]):
                vals = [n_rets[idx], n_sharpe[idx], n_wr[idx], n_score[idx]]
                vals += [vals[0]]   # 닫힘
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill='toself', name=mn,
                    line=dict(color=RADAR_COLORS[idx % len(RADAR_COLORS)], width=2),
                    fillcolor=RADAR_COLORS[idx % len(RADAR_COLORS)].replace('#', 'rgba(').rstrip(')') + ',0.1)'
                    if '#' in RADAR_COLORS[idx % len(RADAR_COLORS)] else 'rgba(0,0,0,0.05)',
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(size=11))
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── 4. 추천 가이드 ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 💡 추천 사용 가이드")
        st.info(
            f"1. 사이드바에서 **{best_mn}** 을 선택하세요.\n\n"
            + (f"2. 최적 파라미터를 적용하세요: "
               + ", ".join([f"{k}={round(v,4)}"
                             for k, v in best_s['hpo_params'].items()])
               if best_s['hpo_params'] else
               "2. HPO 탭에서 파라미터 최적화를 먼저 실행하면 더 정확한 결과를 얻을 수 있습니다.")
            + "\n\n3. **Run Evaluation**을 실행하세요."
        )

# ── Run Evaluation ────────────────────────────────────────────────────
if st.button("▶ Run Evaluation", type="primary"):
    st.session_state.has_run = True

    AgentClass = AGENT_REGISTRY[model_name]

    # 에이전트 생성 — DQN은 target_update 추가 인자
    agent_kwargs = dict(env=env, use_constraints=True,  lr=lr, gamma=gamma, eps=eps)
    vanilla_kwargs = dict(env=env, use_constraints=False, lr=lr, gamma=gamma, eps=eps)
    if model_name in DQN_FAMILY:
        agent_kwargs["target_update"]   = target_update
        vanilla_kwargs["target_update"] = target_update

    if model_name in PPO_FAMILY:
        for kw in (agent_kwargs, vanilla_kwargs):
            kw["clip_eps"]      = clip_eps
            kw["rollout_len"]   = rollout_len
            kw["update_epochs"] = update_epochs

    try:
        agent_static = AgentClass(**agent_kwargs)
        agent_raw    = AgentClass(**vanilla_kwargs)
    except ImportError as e:
        st.error(f"❌ {e}")
        st.stop()

    # 불러오기 요청이 있으면 DCA 에이전트에 가중치 로드
    if st.session_state.get('load_path'):
        load_path = st.session_state['load_path']
        load_meta = st.session_state.get('load_meta', {})
        # 같은 모델 타입인지 확인
        if load_meta.get('model_name') == model_name:
            try:
                agent_static.load(load_path)
                st.toast(f"✅ 모델 불러오기 완료: {os.path.basename(load_path)}")
            except Exception as ex:
                st.warning(f"⚠️ 불러오기 실패: {ex}")
        else:
            st.warning(
                f"⚠️ 저장된 모델({load_meta.get('model_name')})과 "
                f"선택 모델({model_name})이 다릅니다. 불러오기를 건너뜁니다."
            )
        # 불러오기 완료 후 초기화
        st.session_state.pop('load_path', None)
        st.session_state.pop('load_meta', None)

    # Pretrain
    if pretrain > 0:
        st.toast(f"🧠 {model_name} Warming-up ({pretrain} epochs)...")
        for _ in range(pretrain):
            for i in range(20, 20 + episodes):
                _, _, r_u, s_u, a_u, _, _ = agent_raw.select_action(i)
                _, _, r_s, s_s, a_s, _, _ = agent_static.select_action(i)
                agent_raw.learn(s_u, a_u, r_u, min(i + 1, len(env.data) - 1))
                agent_static.learn(s_s, a_s, r_s, min(i + 1, len(env.data) - 1))
                agent_raw.decay_epsilon(i - 20, episodes)
                agent_static.decay_epsilon(i - 20, episodes)

    h_u, h_s, h_b, steps = [0.0], [0.0], [0.0], [0]
    log_data      = []
    UPDATE_INTERVAL = 5

    for i in range(20, 20 + episodes):
        t_u, _, r_u, s_u, a_u, _, raw_u = agent_raw.select_action(i)
        t_s, _, r_s, s_s, a_s, _, raw_s = agent_static.select_action(i)

        agent_raw.learn(s_u, a_u, r_u, min(i + 1, len(env.data) - 1))
        agent_static.learn(s_s, a_s, r_s, min(i + 1, len(env.data) - 1))
        agent_raw.decay_epsilon(i - 20, episodes)
        agent_static.decay_epsilon(i - 20, episodes)

        sc  = float(env.data['SPY'].iloc[i])
        sn  = float(env.data['SPY'].iloc[i + 1])
        r_b = ((sn - sc) / sc) * 100

        step_num = i - 19
        h_u.append(h_u[-1] + raw_u)
        h_s.append(h_s[-1] + raw_s)
        h_b.append(h_b[-1] + r_b)
        steps.append(step_num)

        log_data.append({
            "Day": step_num,
            "Vanilla Pick": t_u, "Vanilla Ret(%)": raw_u,
            "DCA Pick":     t_s, "DCA Ret(%)":     raw_s,
        })

        eps_progress = 1.0 - agent_static.epsilon / max(eps, 1e-6)
        eps_bar.progress(min(eps_progress, 1.0),
                         text=f"ε: {agent_static.epsilon:.3f}  (탐색 → 수렴)")

        if step_num % UPDATE_INTERVAL == 0 or step_num == episodes:
            fig_main.data[0].x, fig_main.data[0].y = steps, h_u
            fig_main.data[1].x, fig_main.data[1].y = steps, h_s
            fig_main.data[2].x, fig_main.data[2].y = steps, h_b
            chart_view.plotly_chart(fig_main, use_container_width=True)

        m_u.metric("🔴 Vanilla RL",       f"{h_u[-1]:.2f}%", f"{raw_u:+.2f}%")
        m_s.metric(f"🔵 DCA ({t_s})",     f"{h_s[-1]:.2f}%", f"{raw_s:+.2f}%")
        m_b.metric("🟢 S&P 500 Index",    f"{h_b[-1]:.2f}%", f"{r_b:+.2f}%")

        m_win_u.metric("🎯 Vanilla Win Rate", f"{agent_raw.get_win_rate():.1f}%")
        m_win_s.metric("🎯 DCA Win Rate",     f"{agent_static.get_win_rate():.1f}%")
        m_q_u.metric("📐 Vanilla Q-Score",   f"{agent_raw.get_avg_q():.4f}")
        m_q_s.metric("📐 DCA Q-Score",       f"{agent_static.get_avg_q():.4f}")

        if speed > 0: time.sleep(speed)

    eps_bar.progress(1.0, text="✅ 학습 완료")

    # 학습된 DCA 에이전트를 session_state에 보관 (저장 버튼 활성화용)
    st.session_state['trained_agent'] = agent_static

    sharpe_u = agent_raw.get_sharpe()
    sharpe_s = agent_static.get_sharpe()
    mdd_u    = agent_raw.get_mdd(h_u)
    mdd_s    = agent_static.get_mdd(h_s)

    st.session_state.trial_history.append({
        "Trial":              len(st.session_state.trial_history) + 1,
        "Model":              model_name,
        "ADAPTIVE Final (%)": h_s[-1], "Vanilla Final (%)": h_u[-1], "SPY Final (%)": h_b[-1],
        "DCA Sharpe":         sharpe_s, "Vanilla Sharpe":    sharpe_u,
        "DCA MDD (%)":        mdd_s,    "Vanilla MDD (%)":   mdd_u,
    })

    with sharpe_area.container():
        st.markdown("#### 📐 Risk-Adjusted Performance")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("📊 Vanilla Sharpe", f"{sharpe_u:.3f}")
        rc2.metric("📊 DCA Sharpe",     f"{sharpe_s:.3f}",
                   delta=f"{sharpe_s - sharpe_u:+.3f} vs Vanilla")
        rc3.metric("📉 Vanilla MDD",    f"{mdd_u:.2f}%")
        rc4.metric("📉 DCA MDD",        f"{mdd_s:.2f}%",
                   delta=f"{mdd_s - mdd_u:+.2f}% vs Vanilla")
        st.markdown("---")

    analysis_header.markdown(f"#### 🔍 Agent Decision Analysis — {model_name}")
    df_log    = pd.DataFrame(log_data).set_index("Day")
    styled_df = df_log.style.map(style_df).format("{:.2f}", subset=["Vanilla Ret(%)", "DCA Ret(%)"])
    tbl_view.dataframe(styled_df, height=350, use_container_width=True)

    fig_bar = px.bar(
        df_log['DCA Pick'].value_counts().reset_index(),
        x='DCA Pick', y='count',
        title=f"<b>DCA Portfolio Allocation ({model_name})</b>",
        color='count', color_continuous_scale='Blues'
    )
    bar_view.plotly_chart(
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)', height=350),
        use_container_width=True
    )

    render_history()

    # 백테스팅 리포트 렌더링
    render_backtest_report(log_data, h_u, h_s, h_b, model_name, env, start_step=20)