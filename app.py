import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# == [UI 개선] CSS: 전역 텍스트 확대 제거 및 특정 요소 핀포인트 스타일링 ==
# 이 CSS는 그래프 외부의 지표 카드(Metric)와 테이블 헤더의 가독성만 개선합니다.
st.markdown("""
<style>
/* 1. 상단 지표 카드(Metric) 라벨 색상 및 크기 강제 지정 */
div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricLabel"] * { color: red !important; font-weight: 900 !important; font-size: 1.2rem !important; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricLabel"] * { color: blue !important; font-weight: 900 !important; font-size: 1.2rem !important; }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricLabel"] * { color: green !important; font-weight: 900 !important; font-size: 1.2rem !important; }

/* 2. 지표 결과 수치 폰트 크기 확대 */
div[data-testid="stMetricValue"] { font-weight: 900 !important; font-size: 2.0rem !important; }

/* 3. 테이블 헤더 스타일 조정 (검정색, 굵게) */
thead tr th { font-size: 18px !important; color: black !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Test-Constrained-RL", layout="wide")
st.markdown("## Test-Constrained-RL-ColdStart: S&P 500 Performance")

if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

with st.spinner('실시간 S&P 500 데이터를 분석 중입니다...'):
    env = SP500Environment()
    agent_raw = RecommendationAgent(env, use_constraints=False)
    agent_static = RecommendationAgent(env, use_constraints=True)

st.sidebar.markdown("### Test Parameters")
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, min(100, max_episodes))
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.0)

# == 1. 메인 시뮬레이션 차트 (레이아웃 복구) ==
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL (Unconstrained)</b>', line=dict(color='red', width=2), marker=dict(symbol='circle-open', size=6, line_width=1.5)))
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC (Ours)</b>', line=dict(color='blue', width=2), marker=dict(symbol='square-open', size=6, line_width=1.5)))
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>S&P 500 Index (SPY)</b>', line=dict(color='green', width=2, dash='dot'), marker=dict(symbol='diamond-open', size=6, line_width=1.5)))

fig.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (S&P 500)</b>", font=dict(size=28, color='black')),
    # 폰트 사이즈를 Plotly 표준(Title 16, Tick 14)으로 되돌려 겹침 방지
    xaxis=dict(title="<b>Trading Days</b>", titlefont=dict(size=18), tickfont=dict(size=14), showgrid=True),
    yaxis=dict(title="<b>Total Cumulative Return (%)</b>", titlefont=dict(size=18), tickfont=dict(size=14), showgrid=True),
    # 범례 위치와 마진 조정 (Title과 겹침 방지)
    legend=dict(font=dict(size=16), x=0.01, y=0.98, borderwidth=1, bgcolor='rgba(255,255,255,0.8)'),
    plot_bgcolor='white', height=550, margin=dict(t=100, b=80, l=80, r=40)
)
fig.add_hline(y=0, line_width=2, line_color="black")

chart_view = st.empty()
chart_view.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
m_u = col1.empty(); m_s = col2.empty(); m_b = col3.empty()

st.markdown("---")
analysis_view = st.empty()

# 테이블 스타일링 함수
def style_dataframe(val):
    if isinstance(val, (int, float)):
        color = 'red' if val < 0 else 'black'
        return f'color: {color}; font-weight: bold; font-size: 16px;'
    return 'color: black; font-weight: bold; font-size: 16px;'

if st.button("Run Evaluation"):
    h_u, h_s, h_b, steps = [0], [0], [0], [0]
    log_data = []

    for i in range(20, 20 + episodes):
        ticker_u, _, r_u = agent_raw.select_action(current_step=i); ticker_s, _, r_s = agent_static.select_action(current_step=i)
        if 'SPY' in env.data.columns:
            spy_curr = float(env.data['SPY'].iloc[i]); spy_next = float(env.data['SPY'].iloc[i+1])
            r_b = ((spy_next - spy_curr) / spy_curr) * 100 if spy_curr > 0 else 0.0
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

    df_log = pd.DataFrame(log_data)
    with analysis_view.container():
        st.markdown("#### Agent Decision Analysis")
        col_tbl, col_bar = st.columns([1.2, 1])
        with col_tbl:
            styled_df = df_log.set_index("Day").style.map(style_dataframe).format("{:.2f}", subset=["Vanilla Return(%)", "STATIC Return(%)"])
            st.dataframe(styled_df, height=350, use_container_width=True)
        with col_bar:
            dist_counts = df_log['STATIC Pick (Ours)'].value_counts().reset_index()
            dist_counts.columns = ['Ticker', 'Buy Count']
            fig_bar = px.bar(dist_counts, x='Ticker', y='Buy Count', title="<b>Frequency of Safe-Asset Selection</b>", color='Buy Count', color_continuous_scale='Blues')
            fig_bar.update_layout(title=dict(font=dict(size=22)), xaxis=dict(tickfont=dict(size=11), tickangle=-45), yaxis=dict(tickfont=dict(size=14)), plot_bgcolor='white', height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

# == 하단: 통계 분석 박스 플롯 (형태 복원 및 밀착 최적화) ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### Trial History: Statistical Superiority Analysis")
    history_df = pd.DataFrame(st.session_state.trial_history)
    
    avg_static = history_df['STATIC Final (%)'].mean(); median_static = history_df['STATIC Final (%)'].median()
    avg_vanilla = history_df['Vanilla Final (%)'].mean(); median_vanilla = history_df['Vanilla Final (%)'].median()
    avg_spy = history_df['SPY Final (%)'].mean()

    col_box, col_hist_table = st.columns([2, 1])
    with col_box:
        fig_box = go.Figure()
        
        # [복원 포인트 1] 박스 테두리 가시성 강화 및 밀착
        # 테두리 두께를 늘리고(width=3), 투명도를 미세하게 조정하여 막대기 형태를 명확히 복원합니다.
        fig_box.add_trace(go.Box(
            y=history_df['Vanilla Final (%)'], name='<b>Vanilla RL</b>', 
            line=dict(color='red', width=3), # 테두리 두께 강화
            fillcolor='rgba(255, 0, 0, 0.05)', # 아주 살짝 색을 채워 박스 형태 유지
            boxmean=True, width=0.4, offsetgroup='1'
        ))
        fig_box.add_trace(go.Box(
            y=history_df['STATIC Final (%)'], name='<b>STATIC RL (Ours)</b>', 
            line=dict(color='blue', width=3), # 테두리 두께 강화
            fillcolor='rgba(0, 0, 255, 0.05)', # 아주 살짝 색을 채워 박스 형태 유지
            boxmean=True, width=0.4, offsetgroup='2'
        ))
        
        # [복원 포인트 2] 수치 라벨 위치 재조정 (겹침 방지)
        # 박스 외부 좌우로 수치를 배치합니다.
        # Vanilla (좌측)
        fig_box.add_annotation(x='<b>Vanilla RL</b>', y=avg_vanilla, text=f"<b>{avg_vanilla:.2f}%</b>", showarrow=False, xshift=-85, font=dict(color='red', size=14))
        fig_box.add_annotation(x='<b>Vanilla RL</b>', y=median_vanilla, text=f"<b>{median_vanilla:.2f}%</b>", showarrow=False, xshift=85, font=dict(color='red', size=14))
        # STATIC (우측)
        fig_box.add_annotation(x='<b>STATIC RL (Ours)</b>', y=avg_static, text=f"<b>{avg_static:.2f}%</b>", showarrow=False, xshift=-85, font=dict(color='blue', size=14))
        fig_box.add_annotation(x='<b>STATIC RL (Ours)</b>', y=median_static, text=f"<b>{median_static:.2f}%</b>", showarrow=False, xshift=85, font=dict(color='blue', size=14))

        # [복원 포인트 3] 가이드 및 S&P 500 라벨 배치
        # 안내창을 우측 상단으로, S&P 500 수치를 좌측 끝으로 분리했습니다.
        fig_box.add_annotation(
            xref="paper", yref="paper", x=1.0, y=1.1, 
            text="<b>[Box Line Guide]</b><br>점선(- - -): Mean | 실선(──): Median", 
            showarrow=False, font=dict(size=12), bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1
        )
        
        fig_box.add_hline(
            y=avg_spy, line_width=2, line_dash="dot", line_color="green", 
            annotation_text=f"<b>S&P 500<br>{avg_spy:.2f}%</b>", 
            annotation_position="top left", annotation_font=dict(color="green", size=14), annotation_x=-0.05
        )

        fig_box.update_layout(
            title=dict(text="<b>Return Distribution across Trials</b>", font=dict(size=26)),
            yaxis=dict(title="<b>Final Cumulative Return (%)</b>", tickfont=dict(size=14)),
            xaxis=dict(tickfont=dict(size=16)),
            boxmode='group', boxgroupgap=0.05, # 막대기 사이를 아주 가깝게 밀착
            plot_bgcolor='white', height=500, margin=dict(t=100, b=60, l=100, r=40)
        )
        fig_box.add_hline(y=0, line_width=2, line_color="black")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_hist_table:
        styled_hist = history_df.set_index("Trial").style.map(style_dataframe).format("{:.2f}")
        st.dataframe(styled_hist, height=500, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 차트 해석 가이드: 박스 내부의 점선(- - -)은 평균값(Mean), 실선(—)은 중앙값(Median)입니다. 가로로 뻗은 초록색 점선은 S&P 500 시장 지수입니다.")