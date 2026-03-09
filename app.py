# import streamlit as st
# import pandas as pd
# import numpy as np
# import time
# import plotly.graph_objects as go
# import config
# from agent import RecommendationAgent

# # == 대시보드 초기 설정 ==
# st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

# st.markdown("## >> Test-Constrained-RL-ColdStart: Performance Analysis")
# st.markdown("이 테스트는 STATIC 제약 조건이 콜드 스타트 환경에서 정확도를 얼마나 개선하는지 정량적으로 측정합니다[cite: 472].")

# # == 차트 및 UI 제어 사이드바 ==
# st.sidebar.markdown("### >> Test Parameters")
# episodes = st.sidebar.slider("Episodes", 10, 300, config.TOTAL_EPISODES)
# speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.05)

# # == 에이전트 인스턴스화 ==
# agent_raw = RecommendationAgent(use_constraints=False)
# agent_static = RecommendationAgent(use_constraints=True)

# # == Plotly 차트: 학술 논문 스타일 설정 ==
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>Vanilla RL (Unconstrained)</b>', line=dict(color='red', width=4)))
# fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='<b>RL with STATIC (Ours)</b>', line=dict(color='blue', width=4)))

# fig.update_layout(
#     title=dict(text="<b>Cumulative Reward Improvement by Constraint Integration</b>", font=dict(size=28, color='black')),
#     xaxis=dict(
#         title="<b>Training Episodes</b>", 
#         titlefont=dict(size=22, color='black'), 
#         tickfont=dict(size=18, color='black', family="Arial Black"),
#         showgrid=True, gridcolor='lightgray'
#     ),
#     yaxis=dict(
#         title="<b>Total Cumulative Reward</b>", 
#         titlefont=dict(size=22, color='black'), 
#         tickfont=dict(size=18, color='black', family="Arial Black"),
#         showgrid=True, gridcolor='lightgray'
#     ),
#     legend=dict(font=dict(size=20, color='black'), x=0.01, y=0.99, borderwidth=1),
#     plot_bgcolor='white',
#     height=650
# )

# chart_view = st.empty()
# chart_view.plotly_chart(fig, use_container_width=True)

# # == 실시간 지표 카드 영역 ==
# col1, col2 = st.columns(2)
# m_u = col1.empty()
# m_s = col2.empty()

# if st.button(">> Run Evaluation"):
#     h_u, h_s, steps = [0], [0], [0]

#     for i in range(1, episodes + 1):
#         # 1. 시뮬레이션 수행
#         _, _, r_u = agent_raw.select_action()
#         _, valid_s, r_s = agent_static.select_action()
        
#         # !! [Critical Check] STATIC 엔진은 이론적으로 실패를 허용하지 않음 [cite: 449]
#         if not valid_s:
#             st.error(f"!! [Error] Static Constraint Violation at Step {i}")
#             break

#         # 2. 데이터 누적
#         h_u.append(h_u[-1] + r_u)
#         h_s.append(h_s[-1] + r_s)
#         steps.append(i)
        
#         # 3. 실시간 그래프 업데이트
#         fig.data[0].x = steps
#         fig.data[0].y = h_u
#         fig.data[1].x = steps
#         fig.data[1].y = h_s
#         chart_view.plotly_chart(fig, use_container_width=True)
        
#         # 4. 텍스트 지표 업데이트
#         m_u.metric(label="<b>Unconstrained Score</b>", value=h_u[-1], delta=r_u)
#         m_s.metric(label="<b>STATIC Score (Ours)</b>", value=h_s[-1], delta=r_s)
        
#         time.sleep(speed)

#     st.success("== Evaluation Sequence Completed. ==")


#---------app.py 일부 수정---------
# 에이전트가 호출될 때 current_step을 넘겨주어 날짜별 실제 주가 변동을 반영하도록 합니다. app.py의 for i in range(...) 루프 부분을 아래와 같이 수정합니다.

# (app.py 상단 수정)
from agent import SP500Environment, RecommendationAgent

# 환경 로드 (데이터 다운로드)
env = SP500Environment()
agent_raw = RecommendationAgent(env, use_constraints=False)
agent_static = RecommendationAgent(env, use_constraints=True)

# ... (UI 코드는 동일하게 유지) ...

if st.button(">> Run Evaluation"):
    h_u, h_s, steps = [0], [0], [0]
    
    # 총 거래일(데이터 길이)만큼 에피소드 진행
    total_days = len(env.data) - 1
    
    for i in range(20, min(episodes + 20, total_days)):
        # 1. 시뮬레이션 수행 (현재 날짜 i를 전달)
        _, _, r_u = agent_raw.select_action(current_step=i)
        ticker_s, valid_s, r_s = agent_static.select_action(current_step=i)
        
        # 2. 데이터 누적
        h_u.append(h_u[-1] + r_u)
        h_s.append(h_s[-1] + r_s)
        steps.append(i - 19) # 1부터 시작하도록 조정
        
        # 3. 실시간 그래프 업데이트
        fig.data[0].x = steps
        fig.data[0].y = h_u
        fig.data[1].x = steps
        fig.data[1].y = h_s
        chart_view.plotly_chart(fig, use_container_width=True)
        
        # 4. 지표 업데이트 (퍼센트 단위이므로 소수점 둘째 자리까지 표시)
        m_u.metric(label="<b>Unconstrained Return (%)</b>", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
        m_s.metric(label=f"<b>STATIC Return (%) - Bought: {ticker_s}</b>", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
        
        time.sleep(speed)

    st.success("== Evaluation Sequence Completed. ==")