"""
GymVision — Streamlit Interface
Two modes:
  1. Video Upload → Exercise Classification + Form Analysis + Dashboard
  2. Live Webcam → Real-time form analysis (user selects exercise first)
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
import time
from collections import deque

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GymVision",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0a0a0f; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0a0a0f 0%, #12121e 50%, #0d1117 100%);
        border: 1px solid #1e2030;
        border-radius: 20px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0 0 24px 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 4px;
    }

    /* Mode cards */
    .mode-card {
        background: #0f1117;
        border: 1px solid #1e2030;
        border-radius: 16px;
        padding: 32px 28px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        height: 100%;
    }
    .mode-card:hover { border-color: #6366f1; transform: translateY(-2px); }
    .mode-card.active { border-color: #6366f1; background: rgba(99,102,241,0.08); }
    .mode-icon { font-size: 2.8rem; margin-bottom: 12px; }
    .mode-title { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
    .mode-desc { font-size: 0.9rem; color: #64748b; line-height: 1.5; }

    /* Metric tiles */
    .metric-tile {
        background: #0f1117;
        border: 1px solid #1e2030;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .metric-good { color: #22c55e; }
    .metric-warn { color: #f59e0b; }
    .metric-bad  { color: #ef4444; }

    /* Alert banners */
    .alert-good { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3);
                  color: #4ade80; border-radius: 8px; padding: 10px 16px; font-weight: 600; margin: 6px 0; }
    .alert-warn { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3);
                  color: #fbbf24; border-radius: 8px; padding: 10px 16px; font-weight: 600; margin: 6px 0; }
    .alert-bad  { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);
                  color: #f87171; border-radius: 8px; padding: 10px 16px; font-weight: 600; margin: 6px 0; }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 28px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 32px;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    .stSelectbox > div > div { background: #0f1117; border-color: #1e2030; color: #e2e8f0; }
    div[data-testid="stFileUploader"] { background: #0f1117; border: 2px dashed #1e2030; border-radius: 12px; }
    .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #a78bfa); }
    h1,h2,h3 { color: #e2e8f0 !important; }
    p, li { color: #94a3b8; }
    .stMarkdown p { color: #94a3b8; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #a78bfa; border-bottom-color: #a78bfa; }
    .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e2030; }
    .stInfo { background: rgba(99,102,241,0.1); border-color: rgba(99,102,241,0.3); color: #a78bfa; }
    .stSuccess { background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.3); }
    .stWarning { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); }
    footer { display: none; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def draw_neon_line(frame, p1, p2, color=(0, 255, 0)):
    overlay = frame.copy()
    cv2.line(overlay, p1, p2, color, 10, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

def draw_neon_joint(frame, point, color=(0, 255, 255)):
    x, y = point
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), 12, color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

def overlay_hud(frame, lines):
    """Draw a semi-transparent HUD panel on the frame."""
    h, w = frame.shape[:2]
    panel_h = len(lines) * 30 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE ANALYSERS
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_squat(frame, lm, w, h, state):
    """Returns updated state dict and list of current alerts."""
    FLEXION_THRESHOLD   = 95
    EXTENSION_THRESHOLD = 165
    ROM_TARGET          = 70
    VALGUS_THRESHOLD    = 0.05

    def pt(id): return (int(lm[id].x * w), int(lm[id].y * h))
    def pa(id): return np.array([lm[id].x * w, lm[id].y * h])

    hip   = pa(mp.solutions.pose.PoseLandmark.RIGHT_HIP.value)
    knee  = pa(mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value)
    ankle = pa(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value)
    l_knee = pa(mp.solutions.pose.PoseLandmark.LEFT_KNEE.value)
    r_knee = pa(mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value)
    l_ankle = pa(mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value)
    r_ankle = pa(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value)

    angle = calculate_angle(hip, knee, ankle)
    state['angles'].append(angle)
    state['min_angle'] = min(state.get('min_angle', 999), angle)
    state['max_angle'] = max(state.get('max_angle', 0), angle)
    rom = state['max_angle'] - state['min_angle']

    if angle < FLEXION_THRESHOLD and state['stage'] == 'up':
        state['stage'] = 'down'
    if angle > EXTENSION_THRESHOLD and state['stage'] == 'down':
        state['stage'] = 'up'
        state['reps'] += 1
        state['min_angle'] = 999
        state['max_angle'] = 0

    # Knee valgus check is skipped for side-view cameras (X-coords unreliable)
    alerts = []
    if state['stage'] == 'down' and angle < FLEXION_THRESHOLD + 5 and rom < ROM_TARGET:
        alerts.append(('⬇ Go Lower!', (0, 80, 255)))
    if angle > 170 and not state.get('knee_lock_alerted', False):
        alerts.append(('⚠ Locking Knees!', (0, 80, 255)))
        state['knee_lock_alerted'] = True
    if angle < 170:
        state['knee_lock_alerted'] = False

    # Draw skeleton
    for p1, p2, col in [
        (pt(23), pt(25), (0,255,0)),
        (pt(25), pt(27), (0,255,0)),
        (pt(24), pt(26), (0,200,0)),
        (pt(26), pt(28), (0,200,0)),
        (pt(23), pt(24), (0,255,255)),
    ]:
        draw_neon_line(frame, p1, p2, col)
    draw_neon_joint(frame, pt(25), (0,255,255))
    draw_neon_joint(frame, pt(26), (0,200,220))

    hud = [
        (f'Reps: {state["reps"]}', (0, 255, 0)),
        (f'Knee Angle: {int(angle)}°', (255, 255, 255)),
        (f'ROM: {int(rom)}°', (255, 255, 0)),
        (f'Stage: {state["stage"].upper()}', (100, 200, 255)),
    ] + alerts
    overlay_hud(frame, hud)

    state['current_angle'] = angle
    state['current_alerts'] = [a[0] for a in alerts]
    state['rom'] = rom
    return state

def analyse_hammer_curl(frame, lm, w, h, state):
    FLEXION_THRESHOLD   = 80
    EXTENSION_THRESHOLD = 150
    ROM_TARGET          = 100
    SWING_THRESHOLD     = 15

    def pt(id): return (int(lm[id].x * w), int(lm[id].y * h))
    def pa(id): return np.array([lm[id].x * w, lm[id].y * h])

    frame_count = state.get('frame_count', 0) + 1
    state['frame_count'] = frame_count
    if frame_count <= 30:
        lv = state.get('left_votes', 0) + (1 if lm[11].visibility > lm[12].visibility else 0)
        rv = state.get('right_votes', 0) + (0 if lm[11].visibility > lm[12].visibility else 1)
        state['left_votes']  = lv
        state['right_votes'] = rv
    dominant = 'Left' if state.get('left_votes', 0) >= state.get('right_votes', 0) else 'Right'

    l_angle = calculate_angle(pa(11), pa(13), pa(15))
    r_angle = calculate_angle(pa(12), pa(14), pa(16))
    dom_angle = l_angle if dominant == 'Left' else r_angle

    state['angles'].append(dom_angle)
    state['min_angle'] = min(state.get('min_angle', 999), dom_angle)
    state['max_angle'] = max(state.get('max_angle', 0), dom_angle)
    rom = state['max_angle'] - state['min_angle']

    if dom_angle < FLEXION_THRESHOLD and state['stage'] == 'down':
        state['stage'] = 'up'
    if dom_angle > EXTENSION_THRESHOLD and state['stage'] == 'up':
        state['stage'] = 'down'
        state['reps'] += 1
        state['min_angle'] = 999
        state['max_angle'] = 0

    # Shoulder swing
    shoulder_y = (pa(11)[1] + pa(12)[1]) / 2
    prev_sy = state.get('prev_shoulder_y', shoulder_y)
    swing = abs(shoulder_y - prev_sy) > SWING_THRESHOLD
    state['prev_shoulder_y'] = shoulder_y

    alerts = []
    if state['stage'] == 'up' and dom_angle < FLEXION_THRESHOLD + 5 and rom < ROM_TARGET:
        alerts.append(('↕ Increase ROM!', (0, 80, 255)))
    if swing:
        alerts.append(('⚠ Avoid Swinging!', (0, 80, 255)))

    # Skeleton — both arms
    for p1, p2 in [(pt(11), pt(13)), (pt(13), pt(15)), (pt(12), pt(14)), (pt(14), pt(16))]:
        draw_neon_line(frame, p1, p2, (0, 255, 0))
    for j in [13, 14]:
        draw_neon_joint(frame, pt(j), (0, 255, 255))

    hud = [
        (f'Reps: {state["reps"]}', (0, 255, 0)),
        (f'Elbow Angle: {int(dom_angle)}°', (255, 255, 255)),
        (f'ROM: {int(rom)}°', (255, 255, 0)),
        (f'L:{int(l_angle)}° R:{int(r_angle)}°', (0, 255, 255)),
        (f'Side: {dominant}', (180, 180, 255)),
    ] + alerts
    overlay_hud(frame, hud)

    state['current_angle'] = dom_angle
    state['current_alerts'] = [a[0] for a in alerts]
    state['rom'] = rom
    state['dominant_side'] = dominant
    return state

def analyse_plank(frame, lm, w, h, state):
    ALIGNMENT_MIN   = 160
    ALIGNMENT_MAX   = 195
    HEAD_DROP_ANGLE = 155
    ELBOW_WIDE_DIST = 0.25

    def pt(id): return (int(lm[id].x * w), int(lm[id].y * h))
    def pa(id): return np.array([lm[id].x * w, lm[id].y * h])

    shoulder_mid = (pa(11) + pa(12)) / 2
    hip_mid      = (pa(23) + pa(24)) / 2
    ankle_mid    = (pa(27) + pa(28)) / 2
    ear_mid      = (pa(7)  + pa(8))  / 2

    body_angle = calculate_angle(shoulder_mid, hip_mid, ankle_mid)
    head_angle = calculate_angle(ear_mid, shoulder_mid, hip_mid)
    elbow_dist = abs(lm[13].x - lm[14].x)

    state['angles'].append(body_angle)

    if state.get('hold_start') is None:
        state['hold_start'] = time.time()
    hold_duration = time.time() - state['hold_start']
    state['hold_duration'] = hold_duration

    alerts = []
    if body_angle < ALIGNMENT_MIN:
        alerts.append(('⬇ Hip Sag!', (0, 80, 255)))
    elif body_angle > ALIGNMENT_MAX:
        alerts.append(('⬆ Hip Pike!', (0, 80, 255)))
    if head_angle < HEAD_DROP_ANGLE:
        alerts.append(('↘ Head Drop!', (0, 80, 255)))
    if elbow_dist > ELBOW_WIDE_DIST:
        alerts.append(('↔ Elbow Flare!', (0, 80, 255)))

    # Skeleton
    for p1, p2 in [
        (pt(11), pt(23)), (pt(23), pt(27)),
        (pt(12), pt(24)), (pt(24), pt(28)),
        (pt(11), pt(12)), (pt(23), pt(24)),
        (pt(11), pt(13)), (pt(12), pt(14)),
    ]:
        draw_neon_line(frame, p1, p2, (0, 255, 0))

    hud = [
        (f'Hold: {hold_duration:.1f}s', (0, 255, 0)),
        (f'Body Angle: {int(body_angle)}°', (255, 255, 255)),
        (f'Head Angle: {int(head_angle)}°', (255, 255, 0)),
    ] + alerts
    overlay_hud(frame, hud)

    state['current_angle'] = body_angle
    state['current_alerts'] = [a[0] for a in alerts]
    state['reps'] = 0  # plank has no reps
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO UPLOAD MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_video_analysis(video_path, exercise):
    mp_pose = mp.solutions.pose

    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = max(cap.get(cv2.CAP_PROP_FPS), 1)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.mktemp(suffix='_gymvision.mp4')
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    state = {'reps': 0, 'stage': 'up' if exercise == 'Squat' else 'down',
             'angles': [], 'min_angle': 999, 'max_angle': 0,
             'current_alerts': [], 'hold_start': None}

    telem = []
    frame_idx = 0

    prog_bar   = st.progress(0, text="🔍 Analysing frames…")
    frame_slot = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if exercise == 'Squat':
                state = analyse_squat(frame, lm, width, height, state)
            elif exercise == 'Hammer Curl':
                state = analyse_hammer_curl(frame, lm, width, height, state)
            elif exercise == 'Plank':
                state = analyse_plank(frame, lm, width, height, state)

        telem.append({
            'Frame':   frame_idx,
            'Time(s)': round(frame_idx / fps, 2),
            'Angle':   round(state.get('current_angle', 0), 1),
            'Reps':    state['reps'],
            'Alerts':  ' | '.join(state.get('current_alerts', [])),
        })

        out.write(frame)
        frame_idx += 1

        if frame_idx % 10 == 0:
            pct = min(frame_idx / max(total, 1), 1.0)
            prog_bar.progress(pct, text=f"🔍 Analysing frame {frame_idx}/{total}…")
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             channels='RGB', width=None,
                             caption=f"Frame {frame_idx}")

    cap.release()
    out.release()
    pose.close()
    prog_bar.progress(1.0, text="✅ Analysis complete!")
    frame_slot.empty()

    return out_path, pd.DataFrame(telem), state


def show_dashboard(df, state, exercise):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        st.markdown('<div class="section-header">📊 Biomechanics Dashboard</div>', unsafe_allow_html=True)

        df_valid = df[df['Angle'] > 0].copy()

        if exercise == 'Squat':
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=('Knee Angle Over Time', 'Reps Accumulation'),
                                vertical_spacing=0.12)
            fig.add_trace(go.Scatter(x=df_valid['Time(s)'], y=df_valid['Angle'],
                                     name='Knee Angle', line=dict(color='#6366f1', width=2)), row=1, col=1)
            fig.add_hline(y=95,  line_dash='dash', line_color='cyan',  annotation_text='Squat Bottom', row=1, col=1)
            fig.add_hline(y=165, line_dash='dash', line_color='yellow', annotation_text='Standing',    row=1, col=1)
            fig.add_trace(go.Scatter(x=df_valid['Time(s)'], y=df_valid['Reps'],
                                     name='Reps', line=dict(color='#22c55e', width=2)), row=2, col=1)

        elif exercise == 'Hammer Curl':
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=('Elbow Angle Over Time', 'Reps Accumulation'),
                                vertical_spacing=0.12)
            fig.add_trace(go.Scatter(x=df_valid['Time(s)'], y=df_valid['Angle'],
                                     name='Elbow Angle', line=dict(color='#6366f1', width=2)), row=1, col=1)
            fig.add_hline(y=80,  line_dash='dash', line_color='cyan',   annotation_text='Curl Top',    row=1, col=1)
            fig.add_hline(y=150, line_dash='dash', line_color='yellow', annotation_text='Curl Bottom', row=1, col=1)
            fig.add_trace(go.Scatter(x=df_valid['Time(s)'], y=df_valid['Reps'],
                                     name='Reps', line=dict(color='#22c55e', width=2)), row=2, col=1)

        else:  # Plank
            fig = make_subplots(rows=1, cols=1,
                                subplot_titles=('Body Alignment Angle Over Time',))
            fig.add_trace(go.Scatter(x=df_valid['Time(s)'], y=df_valid['Angle'],
                                     name='Body Angle', line=dict(color='#6366f1', width=2)), row=1, col=1)
            fig.add_hline(y=160, line_dash='dash', line_color='red',    annotation_text='Sag Limit',  row=1, col=1)
            fig.add_hline(y=195, line_dash='dash', line_color='orange', annotation_text='Pike Limit', row=1, col=1)
            fig.add_hline(y=180, line_dash='dot',  line_color='green',  annotation_text='Perfect',    row=1, col=1)

        # Alert markers
        df_alerts = df_valid[df_valid['Alerts'] != '']
        if not df_alerts.empty:
            fig.add_trace(go.Scatter(
                x=df_alerts['Time(s)'], y=df_alerts['Angle'],
                mode='markers', marker=dict(color='red', size=9, symbol='x'),
                name='Form Alert', hovertext=df_alerts['Alerts'], hoverinfo='text+x+y'
            ), row=1, col=1)

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0f1117',
            height=500,
            hovermode='x unified',
            showlegend=True,
            font=dict(color='#94a3b8'),
        )
        fig.update_xaxes(gridcolor='#1e2030', title_text='Time (s)')
        fig.update_yaxes(gridcolor='#1e2030', title_text='Degrees (°)', row=1, col=1)
        if exercise in ('Squat', 'Hammer Curl'):
            fig.update_yaxes(gridcolor='#1e2030', title_text='Reps', row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.info("Install plotly for interactive charts: `pip install plotly`")

    # Alert log
    df_alerts = df[df['Alerts'] != '']
    if not df_alerts.empty:
        st.markdown('<div class="section-header">⚠️ Form Alert Log</div>', unsafe_allow_html=True)
        st.dataframe(df_alerts[['Time(s)', 'Reps', 'Alerts']].head(30),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE WEBCAM MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_mode(exercise):
    mp_pose = mp.solutions.pose

    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3);
         border-radius:12px; padding:16px; margin-bottom:20px; color:#a78bfa; font-weight:600;">
        📹 Live analysis active — Exercise: <span style="color:#38bdf8">{exercise}</span>
        &nbsp;|&nbsp; Press <kbd>Stop</kbd> to end session
    </div>
    """, unsafe_allow_html=True)

    col_vid, col_stats = st.columns([3, 1])

    with col_vid:
        frame_slot = st.empty()
    with col_stats:
        reps_slot   = st.empty()
        angle_slot  = st.empty()
        alert_slot  = st.empty()
        hold_slot   = st.empty()

    stop_btn = st.button("⏹ Stop Session", key="stop_live")

    cap   = cv2.VideoCapture(0)
    pose  = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    state = {'reps': 0, 'stage': 'up' if exercise == 'Squat' else 'down',
             'angles': [], 'min_angle': 999, 'max_angle': 0,
             'current_alerts': [], 'hold_start': None}

    if not cap.isOpened():
        st.error("❌ Could not access webcam. Make sure camera permissions are granted.")
        return

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if exercise == 'Squat':
                state = analyse_squat(frame, lm, w, h, state)
            elif exercise == 'Hammer Curl':
                state = analyse_hammer_curl(frame, lm, w, h, state)
            elif exercise == 'Plank':
                state = analyse_plank(frame, lm, w, h, state)

        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         channels='RGB', use_container_width=True)

        # Stats panel
        with col_stats:
            if exercise != 'Plank':
                reps_slot.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value metric-good">{state['reps']}</div>
                    <div class="metric-label">Reps</div>
                </div>""", unsafe_allow_html=True)
                angle_slot.markdown(f"""
                <div class="metric-tile" style="margin-top:12px">
                    <div class="metric-value">{int(state.get('current_angle', 0))}°</div>
                    <div class="metric-label">Joint Angle</div>
                </div>""", unsafe_allow_html=True)
            else:
                hold = state.get('hold_duration', 0)
                hold_slot.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value metric-good">{hold:.1f}s</div>
                    <div class="metric-label">Hold Time</div>
                </div>""", unsafe_allow_html=True)
                angle_slot.markdown(f"""
                <div class="metric-tile" style="margin-top:12px">
                    <div class="metric-value">{int(state.get('current_angle', 0))}°</div>
                    <div class="metric-label">Body Angle</div>
                </div>""", unsafe_allow_html=True)

            alerts = state.get('current_alerts', [])
            if alerts:
                alert_slot.markdown(
                    ''.join(f'<div class="alert-bad">{a}</div>' for a in alerts),
                    unsafe_allow_html=True
                )
            else:
                alert_slot.markdown('<div class="alert-good">✓ Good Form</div>', unsafe_allow_html=True)

    cap.release()
    pose.close()

    st.success(f"Session ended. Total reps: {state['reps']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">GymVision</div>
    <div class="hero-sub">Real-Time Gym Form Analysis · Computer Vision & Pose Estimation</div>
    <span class="hero-badge">MediaPipe BlazePose</span>
    <span class="hero-badge">CNN-LSTM Detection</span>
    <span class="hero-badge">Rule-Based Heuristics</span>
</div>
""", unsafe_allow_html=True)

# ── Mode selection ─────────────────────────────────────────────────────────────
if 'mode' not in st.session_state:
    st.session_state.mode = None

st.markdown('<div class="section-header">🎯 Choose Analysis Mode</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    active1 = 'active' if st.session_state.mode == 'upload' else ''
    st.markdown(f"""
    <div class="mode-card {active1}">
        <div class="mode-icon">📁</div>
        <div class="mode-title">Video Upload</div>
        <div class="mode-desc">Upload a recorded workout video. Get a fully annotated output video,
        rep count, joint angle charts, and a form quality report.</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Select Video Upload", key="btn_upload"):
        st.session_state.mode = 'upload'
        st.rerun()

with col2:
    active2 = 'active' if st.session_state.mode == 'live' else ''
    st.markdown(f"""
    <div class="mode-card {active2}">
        <div class="mode-icon">🎥</div>
        <div class="mode-title">Live Webcam</div>
        <div class="mode-desc">Activate your camera for real-time pose tracking. Select your exercise
        first, then get instant feedback on every rep.</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Select Live Webcam", key="btn_live"):
        st.session_state.mode = 'live'
        st.rerun()

st.markdown("<hr style='border-color:#1e2030; margin:32px 0'>", unsafe_allow_html=True)


# ── VIDEO UPLOAD ───────────────────────────────────────────────────────────────
if st.session_state.mode == 'upload':
    st.markdown('<div class="section-header">📁 Video Upload & Analysis</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Exercise Type**")
        exercise = st.selectbox(
            "Select exercise performed in the video:",
            ['Squat', 'Hammer Curl', 'Plank'],
            label_visibility='collapsed'
        )

        EXERCISE_INFO = {
            'Squat':       {'metrics': ['Knee angle', 'ROM', 'Valgus collapse', 'Rep count'], 'icon': '🏋️'},
            'Hammer Curl': {'metrics': ['Elbow angle', 'ROM', 'Shoulder swing', 'Rep count', 'L/R symmetry'], 'icon': '💪'},
            'Plank':       {'metrics': ['Body alignment', 'Hold duration', 'Hip sag/pike', 'Head drop'], 'icon': '🧘'},
        }
        info = EXERCISE_INFO[exercise]
        st.markdown(f"""
        <div style="background:#0f1117; border:1px solid #1e2030; border-radius:12px; padding:16px; margin-top:12px">
            <div style="font-size:2rem; margin-bottom:8px">{info['icon']}</div>
            <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:8px; font-weight:600; text-transform:uppercase; letter-spacing:1px">Tracked Metrics</div>
            {''.join(f'<div style="color:#a78bfa; font-size:0.9rem; padding:3px 0">✓ {m}</div>' for m in info['metrics'])}
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        uploaded = st.file_uploader(
            "Drop your video here (MP4, MOV, AVI)",
            type=['mp4', 'mov', 'avi'],
            label_visibility='visible'
        )

        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tmp.write(uploaded.read())
            tmp.flush()

            st.markdown('<div class="section-header">▶️ Input Preview</div>', unsafe_allow_html=True)
            st.video(tmp.name)

            if st.button(f"🚀 Analyse {exercise} Video", key="run_analysis"):
                with st.spinner("Processing…"):
                    out_path, df, final_state = run_video_analysis(tmp.name, exercise)

                st.success("✅ Analysis complete!")

                # ── Summary metrics ──
                st.markdown('<div class="section-header">📈 Session Summary</div>', unsafe_allow_html=True)
                mc = st.columns(4)
                if exercise != 'Plank':
                    avg_angle = df[df['Angle'] > 0]['Angle'].mean()
                    min_angle = df[df['Angle'] > 0]['Angle'].min()
                    alert_pct = (df['Alerts'] != '').sum() / max(len(df), 1) * 100

                    mc[0].metric("Total Reps",   final_state['reps'])
                    mc[1].metric("Avg Angle",    f"{avg_angle:.1f}°")
                    mc[2].metric("Min Angle",    f"{min_angle:.1f}°")
                    mc[3].metric("Form Alerts",  f"{alert_pct:.1f}%")
                else:
                    hold = final_state.get('hold_duration', 0)
                    avg_angle = df[df['Angle'] > 0]['Angle'].mean()
                    alert_pct = (df['Alerts'] != '').sum() / max(len(df), 1) * 100
                    mc[0].metric("Hold Time",   f"{hold:.1f}s")
                    mc[1].metric("Avg Angle",   f"{avg_angle:.1f}°")
                    mc[2].metric("Form Alerts", f"{alert_pct:.1f}%")
                    mc[3].metric("Frames",      len(df))

                # ── Output video ──
                st.markdown('<div class="section-header">🎬 Analysed Output Video</div>', unsafe_allow_html=True)
                with open(out_path, 'rb') as vf:
                    st.download_button(
                        "⬇️ Download Annotated Video",
                        data=vf,
                        file_name=f"{exercise.lower().replace(' ','_')}_gymvision.mp4",
                        mime='video/mp4'
                    )
                st.video(out_path)

                # ── Dashboard ──
                show_dashboard(df, final_state, exercise)

                # Cleanup
                try:
                    os.unlink(tmp.name)
                except PermissionError:
                    pass  # Windows: file still held by video player; will be cleaned up by OS


# ── LIVE WEBCAM ────────────────────────────────────────────────────────────────
elif st.session_state.mode == 'live':
    st.markdown('<div class="section-header">🎥 Live Webcam Analysis</div>', unsafe_allow_html=True)

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        st.markdown("**Select Your Exercise**")
        live_exercise = st.selectbox(
            "Exercise:",
            ['Squat', 'Hammer Curl', 'Plank'],
            label_visibility='collapsed',
            key='live_exercise'
        )

        st.markdown("""
        <div style="background:#0f1117; border:1px solid #1e2030; border-radius:12px; padding:16px; margin-top:12px">
            <div style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px">Setup Tips</div>
            <div style="color:#64748b; font-size:0.85rem; line-height:1.7">
                📐 Position camera so your full body is visible<br>
                💡 Ensure good lighting on your body<br>
                🧍 Stand 6–8 feet from camera<br>
                📱 Side-view works best for squats & planks
            </div>
        </div>
        """, unsafe_allow_html=True)

        start_btn = st.button("▶ Start Live Session", key="start_live")

    with col_info:
        st.markdown(f"""
        <div style="background:#0f1117; border:1px solid #1e2030; border-radius:12px; padding:24px">
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:16px">
                What you'll see in real-time
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
                <div style="background:#12121e; border-radius:8px; padding:12px; text-align:center">
                    <div style="font-size:1.4rem; color:#22c55e; font-weight:800">Reps</div>
                    <div style="color:#64748b; font-size:0.8rem">Automatic counter</div>
                </div>
                <div style="background:#12121e; border-radius:8px; padding:12px; text-align:center">
                    <div style="font-size:1.4rem; color:#6366f1; font-weight:800">Angles</div>
                    <div style="color:#64748b; font-size:0.8rem">Joint tracking</div>
                </div>
                <div style="background:#12121e; border-radius:8px; padding:12px; text-align:center">
                    <div style="font-size:1.4rem; color:#f59e0b; font-weight:800">Alerts</div>
                    <div style="color:#64748b; font-size:0.8rem">Form corrections</div>
                </div>
                <div style="background:#12121e; border-radius:8px; padding:12px; text-align:center">
                    <div style="font-size:1.4rem; color:#38bdf8; font-weight:800">Skeleton</div>
                    <div style="color:#64748b; font-size:0.8rem">33 pose landmarks</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if start_btn:
        st.markdown("<hr style='border-color:#1e2030; margin:20px 0'>", unsafe_allow_html=True)
        run_live_mode(live_exercise)


# ── Footer ──────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding:40px 0; color:#1e2030">
        <div style="font-size:2rem; margin-bottom:8px">💪</div>
        <div style="color:#334155; font-size:0.9rem">Select a mode above to get started</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:48px; padding:20px; border-top:1px solid #0f1117; color:#1e2030; font-size:0.8rem">
    GymVision · CSE3089 Computer Vision · MediaPipe + OpenCV + Streamlit
</div>
""", unsafe_allow_html=True)
