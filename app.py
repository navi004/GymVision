"""
GymVision — Main Entry Point
Run with:  streamlit run app.py
"""

import streamlit as st
import tempfile
import os

from styles import CSS
from video_processor import run_video_analysis, show_dashboard
from live_mode import run_live_mode

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="GymVision",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-title">GymVision</div>
    <div class="hero-sub">Real-Time Gym Form Analysis · Computer Vision & Pose Estimation</div>
    <span class="hero-badge">MediaPipe BlazePose</span>
    <span class="hero-badge">CNN-LSTM Detection</span>
    <span class="hero-badge">Rule-Based Heuristics</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO UPLOAD MODE
# ═══════════════════════════════════════════════════════════════════════════════

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
            'Squat':       {'metrics': ['Knee angle', 'ROM', 'Rep count', 'Knee-lock detection'], 'icon': '🏋️'},
            'Hammer Curl': {'metrics': ['Elbow angle', 'ROM', 'Shoulder swing', 'Rep count', 'L/R symmetry'], 'icon': '💪'},
            'Plank':       {'metrics': ['Body alignment', 'Hold duration', 'Hip sag/pike', 'Head drop'], 'icon': '🧘'},
        }
        info = EXERCISE_INFO[exercise]
        st.markdown(f"""
        <div style="background:#0f1117; border:1px solid #1e2030; border-radius:12px; padding:16px; margin-top:12px">
            <div style="font-size:2rem; margin-bottom:8px">{info['icon']}</div>
            <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:8px; font-weight:600;
                        text-transform:uppercase; letter-spacing:1px">Tracked Metrics</div>
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
            tmp.close()

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
                    mc[0].metric("Total Reps",  final_state['reps'])
                    mc[1].metric("Avg Angle",   f"{avg_angle:.1f}°")
                    mc[2].metric("Min Angle",   f"{min_angle:.1f}°")
                    mc[3].metric("Form Alerts", f"{alert_pct:.1f}%")
                else:
                    hold      = final_state.get('hold_duration', 0)
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
                        file_name=f"{exercise.lower().replace(' ', '_')}_gymvision.mp4",
                        mime='video/mp4'
                    )

                with open(out_path, 'rb') as vf:
                    video_bytes = vf.read()
                import base64
                video_b64 = base64.b64encode(video_bytes).decode()
                st.markdown(f"""
                <video width="100%" controls autoplay
                    onloadedmetadata="this.playbackRate=0.25;">
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                </video>
                """, unsafe_allow_html=True)


                # ── Dashboard ──
                show_dashboard(df, final_state, exercise)

                # ── Cleanup temp input file ──
                try:
                    os.unlink(tmp.name)
                except PermissionError:
                    pass  # Windows holds file open; OS will clean it up


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE WEBCAM MODE
# ═══════════════════════════════════════════════════════════════════════════════

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
            <div style="color:#94a3b8; font-size:0.85rem; font-weight:600;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:8px">Setup Tips</div>
            <div style="color:#64748b; font-size:0.85rem; line-height:1.7">
                Position camera so your full body is visible<br>
                Ensure good lighting on your body<br>
                Stand 6-8 feet from camera<br>
                Side-view works best for squats and planks
            </div>
        </div>
        """, unsafe_allow_html=True)

        start_btn = st.button("▶ Start Live Session", key="start_live")

    with col_info:
        st.markdown("""
        <div style="background:#0f1117; border:1px solid #1e2030; border-radius:12px; padding:24px">
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:16px">
                What you will see in real-time
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


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT (no mode selected yet)
# ═══════════════════════════════════════════════════════════════════════════════

else:
    st.markdown("""
    <div style="text-align:center; padding:40px 0;">
        <div style="font-size:2rem; margin-bottom:8px">💪</div>
        <div style="color:#334155; font-size:0.9rem">Select a mode above to get started</div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:48px; padding:20px;
            border-top:1px solid #0f1117; color:#334155; font-size:0.8rem">
    GymVision · CSE3089 Computer Vision · MediaPipe + OpenCV + Streamlit
</div>
""", unsafe_allow_html=True)
