import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# == CSS: 메트릭/테이블 스타일 ==
st.markdown("""
<style>
div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricLabel"] * { color: red !important; font-weight: 900 !important; font-size: 1.6rem !important; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricLabel"] * { color: blue !important; font-weight: 900 !important; font-size: 1.6rem !important; }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricLabel"] * { color: green !important; font-weight: 900 !important; font-size: 1.6rem !important; }
div[data-testid="stMetricValue"] { font-weight: 900 !important; font-size: 2.2rem !important; }
thead tr th { font-size: 20px !important; color: black !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Test-Constrained-RL", layout="wide")
st.markdown("## Test-Constrained-RL-ColdStart: S&P 500 Performance")

if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

with st.spinner('데이터 분석 중...'):
    env = SP500Environment()
    agent_raw = RecommendationAgent(env, use_constraints=False)
    agent_static = RecommendationAgent(env, use_constraints=True)

st.sidebar.markdown("### Test Parameters")
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes", 10, max_episodes, min(100, max_episodes))
speed = st.sidebar.slider("Frame Speed", 0.0, 0.5, 0.0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL</b>', line=dict(color='red', width=2), marker=dict(symbol='circle-open', size=6)))
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC</b>', line=dict(color='blue', width=2), marker=dict(symbol='square-open', size=6)))
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>S&P 500 (SPY)</b>', line=dict(color='green', width=2, dash='dot'), marker=dict(symbol='diamond-open', size=6)))

fig.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (S&P 500)</b>", font=dict(size=30, family="Arial Black")),
    xaxis=dict(title="<b>Trading Days</b>", titlefont=dict(size=24), tickfont=dict(size=18), showgrid=True),
    yaxis=dict(title="<b>Total Cumulative Return (%)</b>", titlefont=dict(size=24), tickfont=dict(size=18), showgrid=True),
    legend=dict(font=dict(size=20), x=0.01, y=0.99, borderwidth=1, bgcolor='rgba(255,255,255,0.8)'),
    plot_bgcolor='white', height=600, margin=dict(t=80, b=80, l=80, r=40)
)
fig.add_hline(y=0, line_width=2, line_color="black")

chart_view = st.empty()
chart_view.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
m_u = col1.empty(); m_s = col2.empty(); m_b = col3.empty()

st.markdown("---")
analysis_view = st.empty()

def style_df(val):
    if isinstance(val, (int, float)):
        return f'color: {"red" if val < 0 else "black"}; font-weight: bold; font-size: 16px;'
    return 'color: black; font-weight: bold; font-size: 16px;'

if st.button("Run Evaluation"):
    h_u, h_s, h_b, steps = [0], [0], [0], [0]
    log_data = []

    for i in range(20, 20 + episodes):
        ticker_u, _, r_u = agent_raw.select_action(i)
        ticker_s, _, r_s = agent_static.select_action(i)
        if 'SPY' in env.data.columns:
            sc, sn = float(env.data['SPY'].iloc[i]), float(env.data['SPY'].iloc[i+1])
            r_b = ((sn - sc) / sc) * 100 if sc > 0 else 0.0
        else: r_b = 0.0
        h_u.append(h_u[-1] + r_u); h_s.append(h_s[-1] + r_s); h_b.append(h_b[-1] + r_b)
        current_day = i - 19; steps.append(current_day)
        log_data.append({"Day": current_day, "Vanilla Pick": ticker_u, "Vanilla Return(%)": r_u, "STATIC Pick (Ours)": ticker_s, "STATIC Return(%)": r_s})
        fig.data[0].x = steps; fig.data[0].y = h_u
        fig.data[1].x = steps; fig.data[1].y = h_s
        fig.data[2].x = steps; fig.data[2].y = h_b
        chart_view.plotly_chart(fig, use_container_width=True)
        m_u.metric(label="Unconstrained Return", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
        m_s.metric(label=f"STATIC Return - Bought: {ticker_s}", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
        m_b.metric(label="S&P 500 Index (SPY)", value=f"{h_b[-1]:.2f}%", delta=f"{r_b:.2f}%")
        time.sleep(speed)

    st.session_state.trial_history.append({"Trial": len(st.session_state.trial_history) + 1, "Vanilla Final (%)": h_u[-1], "STATIC Final (%)": h_s[-1], "SPY Final (%)": h_b[-1]})

    with analysis_view.container():
        st.markdown("#### Agent Decision Analysis")
        col_tbl, col_bar = st.columns([1.2, 1])
        with col_tbl:
            st.dataframe(pd.DataFrame(log_data).set_index("Day").style.map(style_df).format("{:.2f}", subset=["Vanilla Return(%)", "STATIC Return(%)"]), height=350, use_container_width=True)
        with col_bar:
            st.plotly_chart(px.bar(pd.DataFrame(log_data)['STATIC Pick (Ours)'].value_counts().reset_index(), x='STATIC Pick (Ours)', y='count', title="<b>Safe-Asset Selection Frequency</b>", color='count', color_continuous_scale='Blues').update_layout(plot_bgcolor='white', height=350), use_container_width=True)

if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### Trial History: Statistical Superiority Analysis")
    df_h = pd.DataFrame(st.session_state.trial_history)
    avg_s, med_s = df_h['STATIC Final (%)'].mean(), df_h['STATIC Final (%)'].median()
    avg_v, med_v = df_h['Vanilla Final (%)'].mean(), df_h['Vanilla Final (%)'].median()
    avg_spy = df_h['SPY Final (%)'].mean()
    st.info(f"### 누적 승률: {((df_h['STATIC Final (%)'] > df_h['Vanilla Final (%)']).sum() / len(df_h)) * 100:.1f}% | 평균 수익률: STATIC {avg_s:.2f}% vs Vanilla {avg_v:.2f}%")
    
    col_box, col_tbl_h = st.columns([2, 1])
    with col_box:
        fig_box = go.Figure()
        # 박스 테두리 및 형태 고정
        fig_box.add_trace(go.Box(y=df_h['Vanilla Final (%)'], name='<b>Vanilla RL</b>', line=dict(color='red', width=3), fillcolor='rgba(255,0,0,0.05)', boxmean=True, width=0.35, offsetgroup='1'))
        fig_box.add_trace(go.Box(y=df_h['STATIC Final (%)'], name='<b>STATIC RL (Ours)</b>', line=dict(color='blue', width=3), fillcolor='rgba(0,0,255,0.05)', boxmean=True, width=0.35, offsetgroup='2'))
        
        # == [수정 포인트 1] 수치 라벨 박스 "바깥쪽" 배치 (xshift 강화) ==
        # xshift를 대폭 늘려 박스 테두리 선과의 겹침을 원천 차단했습니다.
        fig_box.add_annotation(x='<b>Vanilla RL</b>', y=avg_v, text=f"<b>{avg_v:.2f}%</b>", showarrow=False, xshift=-115, font=dict(color='red', size=14, family="Arial Black"))
        fig_box.add_annotation(x='<b>Vanilla RL</b>', y=med_v, text=f"<b>{med_v:.2f}%</b>", showarrow=False, xshift=115, font=dict(color='red', size=14, family="Arial Black"))
        fig_box.add_annotation(x='<b>STATIC RL (Ours)</b>', y=med_s, text=f"<b>{med_s:.2f}%</b>", showarrow=False, xshift=-115, font=dict(color='blue', size=14, family="Arial Black"))
        fig_box.add_annotation(x='<b>STATIC RL (Ours)</b>', y=avg_s, text=f"<b>{avg_s:.2f}%</b>", showarrow=False, xshift=115, font=dict(color='blue', size=14, family="Arial Black"))

        # == [수정 포인트 2] S&P 500 중앙 수동 배치 (에러 해결) ==
        # annotation_position 대신 xref="paper"와 x=0.5를 사용하여 선 중앙 위에 정확히 안착시켰습니다.
        fig_box.add_hline(y=avg_spy, line_width=2.5, line_dash="dot", line_color="green")
        fig_box.add_annotation(x=0.5, xref="paper", y=avg_spy, text=f"<b>S&P 500 {avg_spy:.2f}%</b>", showarrow=False, yshift=15, font=dict(color="green", size=16, family="Arial Black"))

        fig_box.add_annotation(xref="paper", yref="paper", x=1.0, y=1.1, text="<b>[Line Guide]</b><br>점선: Mean | 실선: Median", showarrow=False, font=dict(size=12), bgcolor="white", bordercolor="black", borderwidth=1)
        fig_box.update_layout(title=dict(text="<b>Return Distribution across Trials</b>", font=dict(size=26)), yaxis=dict(title="<b>Final Return (%)</b>"), boxmode='group', boxgroupgap=0.05, plot_bgcolor='white', height=500, margin=dict(t=120, b=60, l=100, r=40))
        fig_box.add_hline(y=0, line_width=2, line_color="black")
        st.plotly_chart(fig_box, use_container_width=True)
    with col_tbl_h:
        st.dataframe(df_h.set_index("Trial").style.map(style_df).format("{:.2f}"), height=500, use_container_width=True)
    st.markdown("#### 차트 해석 가이드: 점선(- - -)은 평균값(Mean), 실선(—)은 중앙값(Median)입니다.")