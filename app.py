import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import config
from agent import RecommendationAgent

# == 대시보드 초기 설정 ==
st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

st.markdown("## >> Test-Constrained-RL-ColdStart: Performance Analysis")
st.markdown("이 테스트는 STATIC 제약 조건이 콜드 스타트 환경에서 정확도를 얼마나 개선하는지 정량적으로 측정합니다[cite: 472].")

# == 차트 및 UI 제어 사이드바 ==
st.sidebar.markdown("### >> Test Parameters")
episodes = st.sidebar.slider("Episodes", 10, 300, config.TOTAL_EPISODES)
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.05)

# == 에이전트 인스턴스화 ==
agent_raw = RecommendationAgent(use_constraints=False)
agent_static = RecommendationAgent(use_constraints=True)

# == Plotly 차트: 학술 논문 스타일 설정 ==
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>Vanilla RL (Unconstrained)</b>', line=dict(color='red', width=4)))
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>RL with STATIC (Ours)</b>', line=dict(color='blue', width=4)))

fig.update_layout(
    title=dict(text="<b>Cumulative Reward Improvement by Constraint Integration</b>", font=dict(size=28, color='black')),
    xaxis=dict(
        title="<b>Training Episodes</b>", 
        titlefont=dict(size=22, color='black'), 
        tickfont=dict(size=18, color='black', family="Arial Black"),
        showgrid=True, gridcolor='lightgray'
    ),
    yaxis=dict(
        title="<b>Total Cumulative Reward</b>", 
        titlefont=dict(size=22, color='black'), 
        tickfont=dict(size=18, color='black', family="Arial Black"),
        showgrid=True, gridcolor='lightgray'
    ),
    legend=dict(font=dict(size=20, color='black'), x=0.01, y=0.99, borderwidth=1),
    plot_bgcolor='white',
    height=650
)

chart_view = st.empty()
chart_view.plotly_chart(fig, use_container_width=True)

# == 실시간 지표 카드 영역 ==
col1, col2 = st.columns(2)
m_u = col1.empty()
m_s = col2.empty()

if st.button(">> Run Evaluation"):
    h_u, h_s, steps = [0], [0], [0]

    for i in range(1, episodes + 1):
        # 1. 시뮬레이션 수행
        _, _, r_u = agent_raw.select_action()
        _, valid_s, r_s = agent_static.select_action()
        
        # !! [Critical Check] STATIC 엔진은 이론적으로 실패를 허용하지 않음 [cite: 449]
        if not valid_s:
            st.error(f"!! [Error] Static Constraint Violation at Step {i}")
            break

        # 2. 데이터 누적
        h_u.append(h_u[-1] + r_u)
        h_s.append(h_s[-1] + r_s)
        steps.append(i)
        
        # 3. 실시간 그래프 업데이트
        fig.data[0].x = steps
        fig.data[0].y = h_u
        fig.data[1].x = steps
        fig.data[1].y = h_s
        chart_view.plotly_chart(fig, use_container_width=True)
        
        # 4. 텍스트 지표 업데이트
        m_u.metric(label="<b>Unconstrained Score</b>", value=h_u[-1], delta=r_u)
        m_s.metric(label="<b>STATIC Score (Ours)</b>", value=h_s[-1], delta=r_s)
        
        time.sleep(speed)

    st.success("== Evaluation Sequence Completed. ==")