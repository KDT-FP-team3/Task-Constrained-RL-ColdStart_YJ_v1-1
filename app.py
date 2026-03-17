import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

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

# ── 환경 초기화 ─────────────────────────────────────────────────────────────
env           = SP500Environment()
max_episodes  = len(env.data) - 20 - 1 if len(env.data) > 521 else 100
default_steps = min(500, max_episodes)

# ── Regime 배지 ─────────────────────────────────────────────────────────────
regime      = env.get_market_regime(min(20 + default_steps - 1, len(env.data) - 1))
badge_class = {"Bull": "regime-bull", "Bear": "regime-bear"}.get(regime, "regime-neutral")
st.markdown(
    f'<h1 style="margin-bottom:4px">🏛️ Personalized S&P 500 Performance Terminal'
    f'<span class="{badge_class}">{regime} Market</span></h1>',
    unsafe_allow_html=True
)
st.caption("Tabular Q-Learning + Adaptive Constraint Engine vs Vanilla RL  ·  데이터: Yahoo Finance 5년")

# ── 사이드바 ────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ System Parameters")
episodes = st.sidebar.number_input("Episodes (Trading Days)", value=default_steps, disabled=True)
speed    = st.sidebar.slider("Execution Speed (sec)", 0.0, 0.5, 0.02, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 RL Hyperparameters")

st.sidebar.caption("학습 속도 — 너무 크면 발산, 너무 작으면 수렴 느림")
lr = st.sidebar.number_input("Learning Rate (α)", value=0.005, format="%.3f", step=0.001)

st.sidebar.caption("미래 보상 중요도 — 높을수록 장기 수익 지향")
gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.85, step=0.01)

st.sidebar.caption("초기 탐색 확률 — Cosine Annealing으로 자동 감소")
eps = st.sidebar.slider("Exploration (ε)", 0.0, 1.0, 0.5, step=0.05)

st.sidebar.caption("본 학습 전 과거 데이터 반복 학습 횟수")
pretrain = st.sidebar.slider("Market Pre-Train", 0, 5, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 학습 진행")
eps_bar   = st.sidebar.empty()
eps_label = st.sidebar.empty()
eps_bar.progress(0.0, text="대기 중...")

# ── session state ─────────────────────────────────────────────────────────
if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []
if 'has_run' not in st.session_state:
    st.session_state.has_run = False

# ── 탭 구조 ──────────────────────────────────────────────────────────────
tab_live, tab_analysis, tab_history = st.tabs(["📈 Live Monitor", "📊 Analysis", "🏆 Trial History"])

with tab_live:
    if not st.session_state.has_run:
        st.info("👈 사이드바에서 파라미터를 설정한 뒤 **Run Evaluation** 버튼을 눌러 시뮬레이션을 시작하세요.")
        st.caption("처음 실행 시 Yahoo Finance에서 5년치 데이터를 자동으로 불러옵니다 (이후 1시간 캐시).")

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines',
        name='<b>Vanilla RL</b>',             line=dict(color='#e05050', width=2.5)))
    fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines',
        name='<b>Personalized DCA (Ours)</b>', line=dict(color='#4a90d9', width=3)))
    fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines',
        name='<b>S&P 500 Index</b>',           line=dict(color='#2ea84a', width=2, dash='dot')))
    fig_main.update_layout(
        title=dict(text="<b>Cumulative Return Comparison</b>", font=dict(size=22)),
        xaxis=dict(title="<b>Trading Days</b>", showgrid=True),
        yaxis=dict(title="<b>Cumulative Return (%)</b>", showgrid=True),
        legend=dict(font=dict(size=13), x=0.01, y=0.99, bgcolor='rgba(128,128,128,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=460, margin=dict(t=50, b=40)
    )
    fig_main.add_hline(y=0, line_width=1.5, line_color="rgba(150,150,150,0.4)")
    chart_view = st.empty()
    chart_view.plotly_chart(fig_main, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    m_u, m_s, m_b    = col1.empty(), col2.empty(), col3.empty()

    st.markdown("---")
    st.markdown("##### 🔬 Learning Monitor")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    m_win_u, m_win_s, m_q_u, m_q_s = col_l1.empty(), col_l2.empty(), col_l3.empty(), col_l4.empty()

with tab_analysis:
    if not st.session_state.has_run:
        st.info("Run Evaluation 실행 후 에이전트 결정 분석 결과가 여기에 표시됩니다.")
    sharpe_area     = st.empty()
    analysis_header = st.empty()
    col_tbl, col_bar = st.columns([1.2, 1])
    tbl_view = col_tbl.empty()
    bar_view = col_bar.empty()

with tab_history:
    if not st.session_state.trial_history:
        st.info("Run Evaluation을 여러 번 실행하면 시도별 성과 분포가 여기에 누적됩니다.")
    history_area = st.empty()

# ── helpers ───────────────────────────────────────────────────────────────
def style_df(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #e05050; font-weight: bold;'
    return 'font-weight: bold;'

def render_history():
    if not st.session_state.trial_history:
        return
    df_h = pd.DataFrame(st.session_state.trial_history)
    with history_area.container():
        st.markdown("#### 📊 Consolidated Performance Benchmarks")

        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=df_h['Vanilla Final (%)'],  name='Vanilla RL',       line=dict(color='#e05050')))
        fig_box.add_trace(go.Box(y=df_h['ADAPTIVE Final (%)'], name='DCA Agent (Ours)', line=dict(color='#4a90d9')))
        avg_spy = df_h['SPY Final (%)'].mean()
        fig_box.add_hline(y=avg_spy, line_dash="dot", line_color="#2ea84a",
                          annotation_text=f"S&P 500 Avg: {avg_spy:.2f}%")
        fig_box.update_layout(title="<b>Final Return Distribution</b>", height=400,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

        if 'DCA Sharpe' in df_h.columns:
            fig_sh = go.Figure()
            fig_sh.add_trace(go.Box(y=df_h['Vanilla Sharpe'], name='Vanilla RL',       line=dict(color='#e05050')))
            fig_sh.add_trace(go.Box(y=df_h['DCA Sharpe'],     name='DCA Agent (Ours)', line=dict(color='#4a90d9')))
            fig_sh.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Sharpe = 0")
            fig_sh.update_layout(title="<b>Sharpe Ratio Distribution</b>", height=360,
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sh, use_container_width=True)

        fmt_cols = {c: "{:.2f}" for c in df_h.columns if c != "Trial"}
        st.dataframe(df_h.set_index("Trial").style.format(fmt_cols), use_container_width=True)

render_history()

# ── Run Evaluation ────────────────────────────────────────────────────────
if st.button("▶ Run Evaluation", type="primary"):
    st.session_state.has_run = True

    agent_raw    = RecommendationAgent(env, use_constraints=False, lr=lr, gamma=gamma, eps=eps)
    agent_static = RecommendationAgent(env, use_constraints=True,  lr=lr, gamma=gamma, eps=eps)

    # Pretrain
    if pretrain > 0:
        st.toast(f"🧠 Market Warming-up ({pretrain} epochs)...")
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
            "DCA Pick (Ours)": t_s, "DCA Ret(%)": raw_s,
        })

        # ε progress bar
        eps_progress = 1.0 - agent_static.epsilon / max(eps, 1e-6)
        eps_bar.progress(min(eps_progress, 1.0),
                         text=f"ε: {agent_static.epsilon:.3f}  (탐색 → 수렴)")

        # 차트 갱신 (N스텝마다)
        if step_num % UPDATE_INTERVAL == 0 or step_num == episodes:
            fig_main.data[0].x, fig_main.data[0].y = steps, h_u
            fig_main.data[1].x, fig_main.data[1].y = steps, h_s
            fig_main.data[2].x, fig_main.data[2].y = steps, h_b
            chart_view.plotly_chart(fig_main, use_container_width=True)

        m_u.metric("🔴 Vanilla RL",      f"{h_u[-1]:.2f}%", f"{raw_u:+.2f}%")
        m_s.metric(f"🔵 DCA  ({t_s})",   f"{h_s[-1]:.2f}%", f"{raw_s:+.2f}%")
        m_b.metric("🟢 S&P 500 Index",   f"{h_b[-1]:.2f}%", f"{r_b:+.2f}%")

        m_win_u.metric("🎯 Vanilla Win Rate", f"{agent_raw.get_win_rate():.1f}%")
        m_win_s.metric("🎯 DCA Win Rate",     f"{agent_static.get_win_rate():.1f}%")
        m_q_u.metric("📐 Vanilla Q-Score",   f"{agent_raw.get_avg_q():.4f}")
        m_q_s.metric("📐 DCA Q-Score",       f"{agent_static.get_avg_q():.4f}")

        if speed > 0:
            time.sleep(speed)

    eps_bar.progress(1.0, text="✅ 학습 완료")

    # Sharpe / MDD
    sharpe_u = agent_raw.get_sharpe()
    sharpe_s = agent_static.get_sharpe()
    mdd_u    = agent_raw.get_mdd(h_u)
    mdd_s    = agent_static.get_mdd(h_s)

    st.session_state.trial_history.append({
        "Trial":              len(st.session_state.trial_history) + 1,
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

    analysis_header.markdown("#### 🔍 Agent Decision Analysis")
    df_log    = pd.DataFrame(log_data).set_index("Day")
    styled_df = df_log.style.map(style_df).format("{:.2f}", subset=["Vanilla Ret(%)", "DCA Ret(%)"])
    tbl_view.dataframe(styled_df, height=350, use_container_width=True)

    fig_bar = px.bar(
        df_log['DCA Pick (Ours)'].value_counts().reset_index(),
        x='DCA Pick (Ours)', y='count',
        title="<b>DCA Portfolio Allocation</b>",
        color='count', color_continuous_scale='Blues'
    )
    bar_view.plotly_chart(
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)', height=350),
        use_container_width=True
    )

    render_history()