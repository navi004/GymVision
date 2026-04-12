import cv2
import streamlit as st

from analysers import analyse_squat, analyse_hammer_curl, analyse_plank


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE WEBCAM MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_mode(exercise):
    """
    Opens webcam, runs pose analysis frame-by-frame, and updates the
    Streamlit UI with live reps, joint angle, and form alerts.
    Press the Stop button to end the session.
    """
    import mediapipe as mp
    mp_pose = mp.solutions.pose

    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3);
         border-radius:12px; padding:16px; margin-bottom:20px; color:#a78bfa; font-weight:600;">
        📹 Live analysis active — Exercise:
        <span style="color:#38bdf8">{exercise}</span>
        &nbsp;|&nbsp; Press <strong>Stop</strong> to end session
    </div>
    """, unsafe_allow_html=True)

    col_vid, col_stats = st.columns([3, 1])

    with col_vid:
        frame_slot = st.empty()
    with col_stats:
        reps_slot  = st.empty()
        angle_slot = st.empty()
        alert_slot = st.empty()
        hold_slot  = st.empty()

    stop_btn = st.button("⏹ Stop Session", key="stop_live")

    cap  = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    state = {
        'reps': 0,
        'stage': 'up' if exercise == 'Squat' else 'down',
        'angles': [],
        'min_angle': 999,
        'max_angle': 0,
        'current_alerts': [],
        'hold_start': None,
    }

    if not cap.isOpened():
        st.error("❌ Could not access webcam. Make sure camera permissions are granted.")
        return

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if exercise == 'Squat':
                state = analyse_squat(frame, lm, w, h, state)
            elif exercise == 'Hammer Curl':
                state = analyse_hammer_curl(frame, lm, w, h, state)
            elif exercise == 'Plank':
                state = analyse_plank(frame, lm, w, h, state)

        frame_slot.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels='RGB',
            use_container_width=True
        )

        # ── Stats panel ──
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
                alert_slot.markdown(
                    '<div class="alert-good">✓ Good Form</div>',
                    unsafe_allow_html=True
                )

    cap.release()
    pose.close()
    st.success(f"Session ended — Total reps: {state['reps']}")
