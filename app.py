import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# st.set_page_config은 반드시 첫 번째 Streamlit 명령이어야 함
st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

# == [UI 개선] CSS: 지표 카드 및 테이블 가독성 강화 ==
st.markdown("""
<style>
div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricLabel"] * { color: red !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricLabel"] * { color: blue !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricLabel"] * { color: green !important; font-weight: 900 !important; font-size: 1.4rem !important; }
div[data-testid="stMetricValue"] { font-weight: 900 !important; font-size: 2.2rem !important; }
thead tr th { font-size: 18px !important; color: black !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("## Test-Constrained-RL-ColdStart: S&P 500 Performance")

# == 🛠 사이드바: 테스트 및 강화학습 파라미터 제어 ==
st.sidebar.markdown("### ⚙️ System Parameters")
env = SP500Environment()
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, min(100, max_episodes))
# [수정됨] Frame Speed 기본값 0.03, step 0.01 반영
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.03, step=0.01)

# == 실험 재현성을 위한 랜덤 시드 설정 ==
base_seed = st.sidebar.number_input("Base Random Seed", value=2026, step=1, help="실험 재현성을 위한 기본 시드입니다. 각 Trial마다 이 값에 +1씩 더해집니다.")

# == 자동 반복 실행 횟수 설정 ==
auto_runs = st.sidebar.number_input("Auto Run Count", min_value=1, value=1, step=1, help="Run Evaluation 버튼 클릭 시 자동으로 반복 실행할 횟수입니다.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 RL Hyperparameters (Logic: STATIC)")
lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.5, 0.01, step=0.001)
# [수정됨] Gamma 기본값 0.98 반영
gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.98, step=0.01)
eps = st.sidebar.slider("Exploration (ε)", 0.0, 1.0, 0.1, step=0.05)

if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

agent_raw = RecommendationAgent(env, use_constraints=False, lr=lr, gamma=gamma, eps=eps)
agent_static = RecommendationAgent(env, use_constraints=True, lr=lr, gamma=gamma, eps=eps)

# == 📈 메인 수익률 비교 차트 ==
fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL</b>', line=dict(color='red', width=2), marker=dict(symbol='circle-open', size=6)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC</b>', line=dict(color='blue', width=2), marker=dict(symbol='square-open', size=6)))
fig_main.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', name='<b>S&P 500 (SPY)</b>', line=dict(color='green', width=2, dash='dot'), marker=dict(symbol='diamond-open', size=6)))

fig_main.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (S&P 500)</b>", font=dict(size=28)),
    xaxis=dict(title="<b>Trading Days</b>", titlefont=dict(size=18), showgrid=True),
    yaxis=dict(title="<b>Total Cumulative Return (%)</b>", titlefont=dict(size=18), showgrid=True),
    legend=dict(font=dict(size=16), x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    plot_bgcolor='white', height=550, margin=dict(t=80, b=80, l=80, r=40)
)
fig_main.add_hline(y=0, line_width=2, line_color="black")

chart_view = st.empty()
chart_view.plotly_chart(fig_main, use_container_width=True)

col1, col2, col3 = st.columns(3)
m_u, m_s, m_b = col1.empty(), col2.empty(), col3.empty()

st.markdown("---")
# 분석 영역의 구조를 미리 정의 (ID 중복 에러 방지)
analysis_header = st.empty()
col_tbl, col_bar = st.columns([1.2, 1])
tbl_view = col_tbl.empty()
bar_view = col_bar.empty()

def style_df(val):
    if isinstance(val, (int, float)):
        return f'color: {"red" if val < 0 else "black"}; font-weight: bold; font-size: 16px;'
    return 'color: black; font-weight: bold; font-size: 16px;'

if st.button("Run Evaluation"):
    for run in range(auto_runs):
        trial_idx = len(st.session_state.trial_history)
        current_seed = base_seed + trial_idx
        np.random.seed(current_seed)
        st.toast(f"Trial {trial_idx + 1} Started (Seed: {current_seed}) - Run {run + 1}/{auto_runs}")

        h_u, h_s, h_b, steps = [0], [0], [0], [0]
        log_data = []

        for i in range(20, 20 + episodes):
            ticker_u, _, r_u = agent_raw.select_action(current_step=i)
            ticker_s, _, r_s = agent_static.select_action(current_step=i)
            
            if 'SPY' in env.data.columns:
                sc, sn = float(env.data['SPY'].iloc[i]), float(env.data['SPY'].iloc[i+1])
                r_b = ((sn - sc) / sc) * 100 if sc > 0 else 0.0
            else: r_b = 0.0
                
            h_u.append(h_u[-1] + r_u); h_s.append(h_s[-1] + r_s); h_b.append(h_b[-1] + r_b)
            current_day = i - 19; steps.append(current_day) 
            log_data.append({"Day": current_day, "Vanilla Pick": ticker_u, "Vanilla Return(%)": r_u, "STATIC Pick (Ours)": ticker_s, "STATIC Return(%)": r_s})
            
            fig_main.data[0].x = steps; fig_main.data[0].y = h_u
            fig_main.data[1].x = steps; fig_main.data[1].y = h_s
            fig_main.data[2].x = steps; fig_main.data[2].y = h_b
            chart_view.plotly_chart(fig_main, use_container_width=True)
            
            m_u.metric(label="Unconstrained Return", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
            m_s.metric(label=f"STATIC Return - Bought: {ticker_s}", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
            m_b.metric(label="S&P 500 Index (SPY)", value=f"{h_b[-1]:.2f}%", delta=f"{r_b:.2f}%")
            
            if speed > 0:
                time.sleep(speed)

        st.session_state.trial_history.append({
            "Trial": trial_idx + 1, 
            "Seed": current_seed, 
            "Vanilla Final (%)": h_u[-1], 
            "STATIC Final (%)": h_s[-1], 
            "SPY Final (%)": h_b[-1]
        })

        analysis_header.markdown("#### Agent Decision Analysis")
        df_log = pd.DataFrame(log_data).set_index("Day")
        
        # == [버그 해결 포인트] format 함수에 subset 파라미터 추가 ==
        # 숫자형 컬럼에만 "{:.2f}" 포맷팅을 적용하여 ValueError를 방지합니다.
        styled_df = df_log.style.map(style_df).format("{:.2f}", subset=["Vanilla Return(%)", "STATIC Return(%)"])
        tbl_view.dataframe(styled_df, height=350, use_container_width=True)
        
        fig_bar = px.bar(df_log['STATIC Pick (Ours)'].value_counts().reset_index(), x='STATIC Pick (Ours)', y='count', 
                         title="<b>Safe-Asset Selection Frequency</b>", color='count', color_continuous_scale='Blues')
        bar_view.plotly_chart(fig_bar.update_layout(plot_bgcolor='white', height=350), use_container_width=True)

# == 📊 하단: 통계 분석 고도화 (누적 그래프 및 박스 플롯) ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### Trial History: Statistical Analysis (Alpha Performance)")
    df_h = pd.DataFrame(st.session_state.trial_history)
    
    # 통계량 계산
    v_mean, v_max, v_min = df_h['Vanilla Final (%)'].mean(), df_h['Vanilla Final (%)'].max(), df_h['Vanilla Final (%)'].min()
    s_mean, s_max, s_min = df_h['STATIC Final (%)'].mean(), df_h['STATIC Final (%)'].max(), df_h['STATIC Final (%)'].min()
    v_std = df_h['Vanilla Final (%)'].std() if len(df_h) > 1 else 0.0
    s_std = df_h['STATIC Final (%)'].std() if len(df_h) > 1 else 0.0
    avg_spy = df_h['SPY Final (%)'].mean()
    
    st.success(f"시장 평균 대비 **Alpha 기대치(Expected Value)**: STATIC **{s_mean - avg_spy:.2f}%p** | Vanilla **{v_mean - avg_spy:.2f}%p**")

    # == 회차별 누적 성과 추이 그래프 ==
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['Vanilla Final (%)'], mode='lines+markers', name='<b>Vanilla Return</b>', line=dict(color='red', width=2), marker=dict(size=8)))
    fig_trend.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['STATIC Final (%)'], mode='lines+markers', name='<b>STATIC Return (Ours)</b>', line=dict(color='blue', width=2), marker=dict(size=8)))
    
    # Vanilla 선들
    fig_trend.add_hline(y=v_mean, line_dash="solid", line_color="red", opacity=0.4, annotation_text=f"Vanilla Mean", annotation_position="top right")
    fig_trend.add_hline(y=v_max, line_dash="dot", line_color="red", opacity=0.3, annotation_text=f"Vanilla Max", annotation_position="top right")
    fig_trend.add_hline(y=v_min, line_dash="dot", line_color="red", opacity=0.3, annotation_text=f"Vanilla Min", annotation_position="bottom right")
    # STATIC 선들
    fig_trend.add_hline(y=s_mean, line_dash="solid", line_color="blue", opacity=0.4, annotation_text=f"STATIC Mean", annotation_position="top left")
    fig_trend.add_hline(y=s_max, line_dash="dot", line_color="blue", opacity=0.3, annotation_text=f"STATIC Max", annotation_position="top left")
    fig_trend.add_hline(y=s_min, line_dash="dot", line_color="blue", opacity=0.3, annotation_text=f"STATIC Min", annotation_position="bottom left")

    fig_trend.update_layout(
        title=dict(text="<b>Trial-by-Trial Return Progression & Stability</b>", font=dict(size=24, family="Arial Black")),
        xaxis=dict(title="<b>Trial Number</b>", tickmode='linear', dtick=1),
        yaxis=dict(title="<b>Final Return (%)</b>"),
        plot_bgcolor='white', height=400, margin=dict(t=60, b=40, l=40, r=40)
    )
    fig_trend.add_hline(y=0, line_width=2, line_color="black")
    st.plotly_chart(fig_trend, use_container_width=True)

    # == 하단 2단 레이아웃 (박스 플롯 & 통계 테이블) ==
    col_box, col_tbl_h = st.columns([2, 1])
    with col_box:
        fig_box = go.Figure()
        
        # 박스 형태 및 밀착 배치
        fig_box.add_trace(go.Box(y=df_h['Vanilla Final (%)'], x0=1.0, name='<b>Vanilla RL</b>', line=dict(color='red', width=3), fillcolor='rgba(255,0,0,0.05)', boxmean=True, width=0.5))
        fig_box.add_trace(go.Box(y=df_h['STATIC Final (%)'], x0=2.25, name='<b>STATIC RL (Ours)</b>', line=dict(color='blue', width=3), fillcolor='rgba(0,0,255,0.05)', boxmean=True, width=0.5))

        med_v, med_s = df_h['Vanilla Final (%)'].median(), df_h['STATIC Final (%)'].median()

        # 수치 라벨 밀착 배치
        fig_box.add_annotation(x=0.75, y=v_mean, text=f"<b>Mean: {v_mean:.2f}%</b>", showarrow=False, xshift=-4, yshift=8, xanchor='right', font=dict(color='red', size=13, family="Arial Black"))
        fig_box.add_annotation(x=0.75, y=med_v, text=f"<b>Median: {med_v:.2f}%</b>", showarrow=False, xshift=-4, yshift=-8, xanchor='right', font=dict(color='red', size=13, family="Arial Black"))
        fig_box.add_annotation(x=2.5, y=med_s, text=f"<b>Median: {med_s:.2f}%</b>", showarrow=False, xshift=4, yshift=8, xanchor='left', font=dict(color='blue', size=13, family="Arial Black"))
        fig_box.add_annotation(x=2.5, y=s_mean, text=f"<b>Mean: {s_mean:.2f}%</b>", showarrow=False, xshift=4, yshift=-8, xanchor='left', font=dict(color='blue', size=13, family="Arial Black"))

        fig_box.add_hline(y=avg_spy, line_width=2.5, line_dash="dot", line_color="green")
        fig_box.add_annotation(x=1.625, xref="x", y=avg_spy, text=f"<b>S&P 500 (SPY)<br>{avg_spy:.2f}%</b>", showarrow=False, yshift=18, xanchor='center', align='center', font=dict(color="green", size=13, family="Arial Black"), bgcolor="rgba(0,0,0,0)")

        fig_box.update_layout(
            title=dict(text="<b>Return Distribution across Trials</b>", font=dict(size=26, family="Arial Black")),
            yaxis=dict(title="<b>Final Return (%)</b>", titlefont=dict(size=22, family="Arial Black"), tickfont=dict(size=18, family="Arial Black")),
            xaxis=dict(
                title="<b>Performance Metrics</b>", titlefont=dict(size=22, family="Arial Black"),
                tickmode='array', tickvals=[1.0, 2.25], ticktext=['<b>Vanilla RL</b>', '<b>STATIC RL (Ours)</b>'],
                tickfont=dict(size=18, family="Arial Black"), range=[0, 3.0]
            ),
            plot_bgcolor='white', height=550, margin=dict(t=120, b=100, l=80, r=80)
        )
        fig_box.add_hline(y=0, line_width=2, line_color="black")
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col_tbl_h:
        # 테이블 상단 요약 통계량 명시
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 10px;'>
            <h4 style='margin-top:0px; color: black; font-weight: 900;'>📊 통계 요약 (Expected & Risk)</h4>
            <ul style='font-size: 15px; margin-bottom: 0px;'>
                <li><b style='color:red;'>Vanilla 평균(기대치):</b> {v_mean:.2f}% (σ={v_std:.2f}%)</li>
                <li><b style='color:red;'>Vanilla 범위:</b> {v_min:.2f}% ~ {v_max:.2f}%</li>
                <hr style='margin: 8px 0;'>
                <li><b style='color:blue;'>STATIC 평균(기대치):</b> {s_mean:.2f}% (σ={s_std:.2f}%)</li>
                <li><b style='color:blue;'>STATIC 범위:</b> {s_min:.2f}% ~ {s_max:.2f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df_h.set_index("Trial").style.map(style_df).format({"Vanilla Final (%)": "{:.2f}", "STATIC Final (%)": "{:.2f}", "SPY Final (%)": "{:.2f}", "Seed": "{:.0f}"}), height=320, use_container_width=True)