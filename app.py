import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from agent import SP500Environment, RecommendationAgent

# st.set_page_config은 반드시 첫 번째 Streamlit 명령이어야 함
st.set_page_config(page_title="Personalized-RL", layout="wide")

# == [UI 개선] CSS: 지표 카드 및 가독성 강화 ==
st.markdown("""
<style>
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricLabel"] * { color: #e05050 !important; font-weight: 900 !important; font-size: 1.4rem !important; }
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricLabel"] * { color: #4a90d9 !important; font-weight: 900 !important; font-size: 1.4rem !important; }
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricLabel"] * { color: #2ea84a !important; font-weight: 900 !important; font-size: 1.4rem !important; }
    div[data-testid="stMetricValue"] { font-weight: 900 !important; font-size: 2.2rem !important; }
    thead tr th { font-size: 18px !important; color: var(--text-color) !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Personalized S&P 500 Performance Terminal")

# == 🛠 사이드바: 파라미터 제어 ==
st.sidebar.markdown("### ⚙️ System Parameters")
env = SP500Environment()
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 521 else 100
default_eps = 500 if max_episodes >= 500 else max_episodes
episodes = st.sidebar.number_input("Episodes (Trading Days)", value=default_eps, disabled=True)
speed = st.sidebar.slider("Execution Speed (sec)", 0.0, 0.5, 0.02, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 RL Hyperparameters")
lr = st.sidebar.number_input("Learning Rate (α)", value=0.005, format="%.3f", step=0.001)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.85, step=0.01)
eps = st.sidebar.slider("Exploration (ε)", 0.0, 1.0, 0.1, step=0.05)
pretrain = st.sidebar.slider("Market Pre-Train", 0, 5, 2)

if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

# == 📈 메인 수익률 비교 차트 ==
fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>Vanilla RL</b>', line=dict(color='#e05050', width=2.5)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>Personalized DCA (Ours)</b>', line=dict(color='#4a90d9', width=3)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>S&P 500 Index</b>', line=dict(color='green', width=2, dash='dot')))

fig_main.update_layout(
    title=dict(text="<b>Cumulative Return Comparison</b>", font=dict(size=24)),
    xaxis=dict(title="<b>Trading Days</b>", showgrid=True),
    yaxis=dict(title="<b>Total Return (%)</b>", showgrid=True),
    legend=dict(font=dict(size=14), x=0.01, y=0.99, bgcolor='rgba(128,128,128,0.1)'),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500
)
fig_main.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.5)")

chart_view = st.empty()
chart_view.plotly_chart(fig_main, use_container_width=True)

col1, col2, col3 = st.columns(3)
m_u, m_s, m_b = col1.empty(), col2.empty(), col3.empty()

# == 학습 모니터링 영역 ==
st.markdown("---")
col_lr1, col_lr2, col_lr3, col_lr4 = st.columns(4)
m_win_u, m_win_s, m_q_u, m_q_s = col_lr1.empty(), col_lr2.empty(), col_lr3.empty(), col_lr4.empty()

st.markdown("---")
analysis_header = st.empty()
col_tbl, col_bar = st.columns([1.2, 1])
tbl_view = col_tbl.empty()
bar_view = col_bar.empty()

def style_df(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #e05050; font-weight: bold;'
    return 'font-weight: bold;'

if st.button("Run Evaluation"):
    agent_raw = RecommendationAgent(env, use_constraints=False, lr=lr, gamma=gamma, eps=eps)
    agent_static = RecommendationAgent(env, use_constraints=True, lr=lr, gamma=gamma, eps=eps)
    
    if pretrain > 0:
        st.toast(f"🧠 Market Warming-up ({pretrain} epochs)...")
        for _ in range(pretrain):
            for i in range(20, 20 + episodes):
                _, _, r_u, s_u, a_u = agent_raw.select_action(i)
                _, _, r_s, s_s, a_s = agent_static.select_action(i)
                agent_raw.learn(s_u, a_u, r_u, min(i+1, len(env.data)-1))
                agent_static.learn(s_s, a_s, r_s, min(i+1, len(env.data)-1))

    h_u, h_s, h_b, steps = [0], [0], [0], [0]
    log_data = []

    for i in range(20, 20 + episodes):
        t_u, _, r_u, s_u, a_u = agent_raw.select_action(i)
        t_s, _, r_s, s_s, a_s = agent_static.select_action(i)
        
        agent_raw.learn(s_u, a_u, r_u, min(i+1, len(env.data)-1))
        agent_static.learn(s_s, a_s, r_s, min(i+1, len(env.data)-1))
        agent_raw.decay_epsilon(i-20, episodes)
        agent_static.decay_epsilon(i-20, episodes)
        
        sc, sn = float(env.data['SPY'].iloc[i]), float(env.data['SPY'].iloc[i+1])
        r_b = ((sn - sc) / sc) * 100
        
        h_u.append(h_u[-1] + r_u); h_s.append(h_s[-1] + r_s); h_b.append(h_b[-1] + r_b)
        steps.append(i-19)
        
        log_data.append({"Day": i-19, "Vanilla Pick": t_u, "Vanilla Ret(%)": r_u, "DCA Pick (Ours)": t_s, "DCA Ret(%)": r_s})
        
        # UI 업데이트
        fig_main.data[0].x, fig_main.data[0].y = steps, h_u
        fig_main.data[1].x, fig_main.data[1].y = steps, h_s
        fig_main.data[2].x, fig_main.data[2].y = steps, h_b
        chart_view.plotly_chart(fig_main, use_container_width=True)
        
        m_u.metric("Vanilla RL Return", f"{h_u[-1]:.2f}%", f"{r_u:.2f}%")
        m_s.metric(f"DCA Agent (Hold: {t_s})", f"{h_s[-1]:.2f}%", f"{r_s:.2f}%")
        m_b.metric("S&P 500 Index", f"{h_b[-1]:.2f}%", f"{r_b:.2f}%")
        
        m_win_u.metric("🎯 Vanilla Win Rate", f"{agent_raw.get_win_rate():.1f}%")
        m_win_s.metric("🎯 DCA Win Rate", f"{agent_static.get_win_rate():.1f}%")
        m_q_u.metric("📈 Vanilla Q-Score", f"{agent_raw.get_avg_q():.4f}")
        m_q_s.metric("📈 DCA Q-Score", f"{agent_static.get_avg_q():.4f}")
        
        if speed > 0: time.sleep(speed)

    st.session_state.trial_history.append({
        "Trial": len(st.session_state.trial_history) + 1,
        "ADAPTIVE Final (%)": h_s[-1], "Vanilla Final (%)": h_u[-1], "SPY Final (%)": h_b[-1]
    })

    analysis_header.markdown("#### 🔍 Agent Decision Analysis")
    df_log = pd.DataFrame(log_data).set_index("Day")
    styled_df = df_log.style.map(style_df).format("{:.2f}", subset=["Vanilla Ret(%)", "DCA Ret(%)"])
    tbl_view.dataframe(styled_df, height=350, use_container_width=True)
    
    fig_bar = px.bar(df_log['DCA Pick (Ours)'].value_counts().reset_index(), x='DCA Pick (Ours)', y='count',
                     title="<b>DCA Portfolio Allocation</b>", color='count', color_continuous_scale='Blues')
    bar_view.plotly_chart(fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350), use_container_width=True)

# == 📊 하단: 통계 분석 (박스 플롯) ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### 📊 Consolidated Performance Benchmarks")
    df_h = pd.DataFrame(st.session_state.trial_history)
    
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=df_h['Vanilla Final (%)'], name='Vanilla RL', line=dict(color='#e05050')))
    fig_box.add_trace(go.Box(y=df_h['ADAPTIVE Final (%)'], name='DCA Agent (Ours)', line=dict(color='#4a90d9')))
    
    avg_spy = df_h['SPY Final (%)'].mean()
    fig_box.add_hline(y=avg_spy, line_dash="dot", line_color="green", annotation_text=f"S&P 500 Avg: {avg_spy:.2f}%")

    fig_box.update_layout(title="<b>Final Return Distribution Across Trials</b>", height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.dataframe(df_h.set_index("Trial").style.format("{:.2f}"), use_container_width=True)