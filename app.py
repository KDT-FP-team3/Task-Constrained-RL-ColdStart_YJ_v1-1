import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# == [UI 개선] CSS: 전역 텍스트 스타일링 제거 (그래프 충돌 방지) 및 특정 요소 핀포인트 스타일링 ==
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
    xaxis=dict(title="<b>Trading Days</b>", titlefont=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title="<b>Total Cumulative Return (%)</b>", titlefont=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgray'),
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

# == 2. 하단: 통계 분석 박스 플롯 (망가진 레이아웃 완벽 복구 및 가독성 극대화) ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### Trial History: Statistical Superiority Analysis")
    history_df = pd.DataFrame(st.session_state.trial_history)
    
    # 통계 계산
    avg_static = history_df['STATIC Final (%)'].mean(); median_static = history_df['STATIC Final (%)'].median()
    avg_vanilla = history_df['Vanilla Final (%)'].mean(); median_vanilla = history_df['Vanilla Final (%)'].median()
    avg_spy = history_df['SPY Final (%)'].mean()
    
    # Alpha 요약 텍스트
    st.success(f"시장 평균({avg_spy:.2f}%) 대비 **Alpha 성과**: STATIC **{avg_static - avg_spy:.2f}%p** | Vanilla **{avg_vanilla - avg_spy:.2f}%p**")

    col_box, col_hist_table = st.columns([2, 1])
    with col_box:
        fig_box = go.Figure()
        
        # == [UI 개선] 두 막대기 중앙 밀착 및 너비 슬림화 ==
        # width=0.35로 슬림하게 조정하여 데이터와 라벨 사이의 공간 확보. fillcolor 투명화 유지.
        fig_box.add_trace(go.Box(y=history_df['Vanilla Final (%)'], name='<b>Vanilla RL</b>', line=dict(color='red', width=1.5), fillcolor='rgba(0,0,0,0)', boxmean=True, width=0.35))
        fig_box.add_trace(go.Box(y=history_df['STATIC Final (%)'], name='<b>STATIC RL (Ours)</b>', line=dict(color='blue', width=1.5), fillcolor='rgba(0,0,0,0)', boxmean=True, width=0.35))
        
        # == [요청 반영 2] 안내 라벨창 제거 (... 부분 제거) ==
        # 기존의 add_annotation 가이드 박스 블록을 삭제했습니다.
        
        # Vanilla RL: Mean/Median을 하나의 박스로 합쳐 박스 좌측 바깥에 배치
        # xshift=-110 → 박스 절반 너비(~45px)보다 충분히 크게 설정
        fig_box.add_annotation(
            x='<b>Vanilla RL</b>',
            y=(avg_vanilla + median_vanilla) / 2,
            text=f"<b>Mean: {avg_vanilla:.2f}%</b><br><b>Median: {median_vanilla:.2f}%</b>",
            showarrow=False, xshift=-110, xanchor='right', align='right',
            font=dict(color='red', size=12),
            bgcolor='rgba(255,255,255,0.85)', borderpad=5
        )

        # STATIC RL: Mean/Median을 하나의 박스로 합쳐 박스 우측 바깥에 배치
        fig_box.add_annotation(
            x='<b>STATIC RL (Ours)</b>',
            y=(avg_static + median_static) / 2,
            text=f"<b>Median: {median_static:.2f}%</b><br><b>Mean: {avg_static:.2f}%</b>",
            showarrow=False, xshift=110, xanchor='left', align='left',
            font=dict(color='blue', size=12),
            bgcolor='rgba(255,255,255,0.85)', borderpad=5
        )

        # S&P 500 기준선: 선만 그리고, 라벨은 차트 정중앙(xref='paper', x=0.5)에 별도 배치
        fig_box.add_hline(y=avg_spy, line_width=1.5, line_dash="dot", line_color="green")
        fig_box.add_annotation(
            xref='paper', x=0.5,
            yref='y', y=avg_spy,
            text=f"<b>S&P 500: {avg_spy:.2f}%</b>",
            showarrow=False, xanchor='center', yanchor='bottom', yshift=8,
            font=dict(color='green', size=12),
            bgcolor='rgba(255,255,255,0.85)', borderpad=5
        )

        # == [요청 반영 3] 가로/세로축 텍스트 크기 확대 및 볼드체 적용 ==
        fig_box.update_layout(
            title=dict(text="<b>Return Distribution across Trials</b>", font=dict(size=28, color='black')),
            # y축: 제목(bolder, 22), 틱 라벨(bolder, 18) 확대
            yaxis=dict(
                title="<b>Final Cumulative Return (%)</b>", 
                titlefont=dict(size=22, family="Arial Black"), 
                tickfont=dict(size=18, family="Arial Black"), 
                gridcolor='lightgray'
            ),
            # x축: 제목(bolder, 22), 틱 라벨(bolder, 18) 확대
            xaxis=dict(
                title="<b>Performance Metrics</b>", # 명시적인 가로축 제목 추가
                titlefont=dict(size=22, family="Arial Black"), 
                tickfont=dict(size=18, family="Arial Black"), 
                showgrid=True, gridcolor='lightgray'
            ),
            boxmode='group', boxgroupgap=0.05,
            plot_bgcolor='white', height=550, margin=dict(t=120, b=100, l=120, r=160)
        )
        fig_box.add_hline(y=0, line_width=2, line_color="black")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_hist_table:
        styled_hist = history_df.set_index("Trial").style.map(style_dataframe).format("{:.2f}")
        st.dataframe(styled_hist, height=550, use_container_width=True)
    
    st.markdown("---")
    # 차트 가이드 텍스트는 맨 아래 Markdown으로 명시
    st.markdown("#### 차트 해석 가이드: 점선(- - -)은 평균값(Mean), 실선(—)은 중앙값(Median)입니다. 초록색 점선은 S&P 500 지수입니다.")