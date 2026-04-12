import cv2
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

from analysers import analyse_squat, analyse_hammer_curl, analyse_plank


# ===============================================================================
# VIDEO ANALYSIS
# ===============================================================================

def run_video_analysis(video_path, exercise):
    """
    Processes every frame with the appropriate analyser.
    Telemetry now captures all per-frame series needed for the full dashboards.
    Returns (output_video_path, telemetry_dataframe, final_state).
    """
    import mediapipe as mp
    mp_pose = mp.solutions.pose

    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = max(cap.get(cv2.CAP_PROP_FPS), 1)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.mktemp(suffix='_gymvision.mp4')
    # avc1 = H.264 — plays inline in browsers; falls back to mp4v if unavailable
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    state = {
        'reps':             0,
        'stage':            'up' if exercise == 'Squat' else 'down',
        'angles':           [],
        'min_angle':        999,
        'max_angle':        0,
        'current_alerts':   [],
        'hold_start':       None,
        'alert_hold_frames': 0,
        'held_alerts':      [],
        'rep_durations':    [],
        'rep_start_time':   0,
    }

    telem     = []
    frame_idx = 0

    prog_bar   = st.progress(0, text="Analysing frames...")
    frame_slot = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time        = round(frame_idx / fps, 2)
        state['current_time'] = current_time

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

        # --- Collect per-frame telemetry row ---
        row = {
            'Frame':   frame_idx,
            'Time(s)': current_time,
            'Angle':   round(state.get('current_angle', 0), 1),
            'Reps':    state['reps'],
            'Alerts':  ' | '.join(state.get('current_alerts', [])),
        }

        # Exercise-specific extra columns
        if exercise == 'Squat':
            hip_d  = state.get('hip_depth_series',  [0])
            lv     = state.get('l_valgus_series',   [0])
            rv     = state.get('r_valgus_series',   [0])
            row['Hip_Depth']  = hip_d[-1]  if hip_d  else 0
            row['L_Valgus']   = lv[-1]     if lv     else 0
            row['R_Valgus']   = rv[-1]     if rv     else 0

        elif exercise == 'Hammer Curl':
            ls  = state.get('l_angle_series',         [0])
            rs  = state.get('r_angle_series',         [0])
            sd  = state.get('shoulder_delta_series',  [0])
            row['L_Angle']         = ls[-1] if ls else 0
            row['R_Angle']         = rs[-1] if rs else 0
            row['Shoulder_Delta']  = sd[-1] if sd else 0

        elif exercise == 'Plank':
            ha  = state.get('head_angle_series',  [0])
            ed  = state.get('elbow_dist_series',  [0])
            hh  = state.get('hip_height_series',  [0])
            row['Head_Angle']  = ha[-1] if ha else 0
            row['Elbow_Dist']  = ed[-1] if ed else 0
            row['Hip_Height']  = hh[-1] if hh else 0

        telem.append(row)
        out.write(frame)
        frame_idx += 1

        if frame_idx % 10 == 0:
            pct = min(frame_idx / max(total, 1), 1.0)
            prog_bar.progress(pct, text=f"Analysing frame {frame_idx}/{total}...")
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             channels='RGB', use_container_width=True,
                             caption=f"Frame {frame_idx}")

    cap.release()
    out.release()
    pose.close()
    prog_bar.progress(1.0, text="Analysis complete!")
    frame_slot.empty()

    return out_path, pd.DataFrame(telem), state


# ===============================================================================
# BIOMECHANICS DASHBOARDS  — full 4-panel versions matching the notebooks
# ===============================================================================

# Shared Plotly layout defaults
_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='#0a0a0f',
    plot_bgcolor='#0f1117',
    hovermode='x unified',
    showlegend=True,
    font=dict(color='#94a3b8'),
    height=1100,
)
_AXIS = dict(gridcolor='#1e2030')


def _alert_scatter(df, y_col):
    """Red X markers on chart 1 wherever an alert was recorded."""
    import plotly.graph_objects as go
    df_a = df[(df['Alerts'] != '') & (df[y_col] > 0)]
    if df_a.empty:
        return None
    return go.Scatter(
        x=df_a['Time(s)'], y=df_a[y_col],
        mode='markers',
        marker=dict(color='red', size=9, symbol='x'),
        name='Alert Triggered',
        hovertext=df_a['Alerts'],
        hoverinfo='text+x+y'
    )


def _rep_shading(fig, df, row=1, col=1):
    """Alternating light shading behind each rep interval."""
    import plotly.graph_objects as go
    if 'Reps' not in df.columns:
        return
    rep_colors = ['rgba(200,200,200,0.08)', 'rgba(255,255,255,0)']
    for rep_num in df['Reps'].unique():
        if rep_num == 0:
            continue
        seg = df[df['Reps'] == rep_num]
        if seg.empty:
            continue
        fig.add_vrect(
            x0=seg['Time(s)'].min(), x1=seg['Time(s)'].max(),
            fillcolor=rep_colors[int(rep_num) % 2],
            layer='below', line_width=0,
            annotation_text=f'Rep {int(rep_num)}',
            annotation_position='top left',
            row=row, col=col
        )


def show_dashboard(df, state, exercise):
    """Renders the full 4-panel interactive dashboard matching the notebook output."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.info("Install plotly: `pip install plotly`")
        return

    st.markdown('<div class="section-header">📊 Biomechanics Dashboard</div>',
                unsafe_allow_html=True)

    df_v = df[df['Angle'] > 0].copy()

    # ─────────────────────────────────────────────
    # SQUAT  — 4 panels
    # ─────────────────────────────────────────────
    if exercise == 'Squat':
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.07,
            subplot_titles=(
                '1. Knee Angle & Form Alerts',
                '2. Hip Depth Tracker  (Higher Value = Deeper Squat)',
                '3. Knee Valgus  (Higher = More Collapse)',
                '4. Rep Duration & Fatigue Analysis',
            ),
            row_heights=[0.34, 0.22, 0.22, 0.22]
        )

        # Panel 1 — knee angle
        fig.add_trace(go.Scatter(
            x=df_v['Time(s)'], y=df_v['Angle'],
            name='Knee Angle', line=dict(color='royalblue', width=2)
        ), row=1, col=1)
        fig.add_hline(y=95,  line_dash='dash', line_color='cyan',
                      annotation_text='Squat Bottom', row=1, col=1)
        fig.add_hline(y=165, line_dash='dash', line_color='yellow',
                      annotation_text='Standing',     row=1, col=1)
        _rep_shading(fig, df_v, row=1, col=1)
        a = _alert_scatter(df_v, 'Angle')
        if a: fig.add_trace(a, row=1, col=1)

        # Panel 2 — hip depth
        if 'Hip_Depth' in df_v.columns:
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['Hip_Depth'],
                name='Hip Depth', line=dict(color='darkorange', width=2)
            ), row=2, col=1)

        # Panel 3 — valgus
        if 'L_Valgus' in df_v.columns:
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['L_Valgus'],
                name='Left Valgus', line=dict(color='mediumseagreen', width=2)
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['R_Valgus'],
                name='Right Valgus', line=dict(color='mediumpurple', width=2)
            ), row=3, col=1)

        # Panel 4 — rep durations bar
        rep_durs = state.get('rep_durations', [])
        if rep_durs:
            fig.add_trace(go.Bar(
                x=[f'Rep {i+1}' for i in range(len(rep_durs))],
                y=rep_durs,
                name='Time per Rep',
                marker_color='indianred'
            ), row=4, col=1)

        fig.update_yaxes(title_text='Degrees (°)',  gridcolor='#1e2030', row=1, col=1)
        fig.update_yaxes(title_text='Vertical Dist.', gridcolor='#1e2030', row=2, col=1)
        fig.update_yaxes(title_text='Valgus Ratio',  gridcolor='#1e2030', row=3, col=1)
        fig.update_yaxes(title_text='Seconds',       gridcolor='#1e2030', row=4, col=1)
        fig.update_xaxes(title_text='Time in Video (s)', gridcolor='#1e2030', row=3, col=1)
        fig.update_layout(
            **_LAYOUT,
            title_text='Squat Biomechanics & Performance Dashboard'
        )

    # ─────────────────────────────────────────────
    # HAMMER CURL — 4 panels
    # ─────────────────────────────────────────────
    elif exercise == 'Hammer Curl':
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.07,
            subplot_titles=(
                '1. Elbow Angle & Form Alerts',
                '2. Left vs Right Arm Symmetry',
                '3. Shoulder Stability  (lower = less swinging)',
                '4. Rep Duration & Fatigue Analysis',
            ),
            row_heights=[0.35, 0.25, 0.20, 0.20]
        )

        # Panel 1 — dominant arm angle
        fig.add_trace(go.Scatter(
            x=df_v['Time(s)'], y=df_v['Angle'],
            name='Dom Arm Angle', line=dict(color='royalblue', width=2)
        ), row=1, col=1)
        fig.add_hline(y=80,  line_dash='dash', line_color='cyan',
                      annotation_text='Curl Top',    row=1, col=1)
        fig.add_hline(y=150, line_dash='dash', line_color='yellow',
                      annotation_text='Curl Bottom', row=1, col=1)
        _rep_shading(fig, df_v, row=1, col=1)
        a = _alert_scatter(df_v, 'Angle')
        if a: fig.add_trace(a, row=1, col=1)

        # Panel 2 — L vs R symmetry
        if 'L_Angle' in df_v.columns:
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['L_Angle'],
                name='Left Arm', line=dict(color='teal', width=2)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['R_Angle'],
                name='Right Arm', line=dict(color='mediumpurple', width=2)
            ), row=2, col=1)

        # Panel 3 — shoulder stability
        if 'Shoulder_Delta' in df_v.columns:
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['Shoulder_Delta'],
                name='Shoulder Movement', line=dict(color='darkorange', width=2)
            ), row=3, col=1)
            fig.add_hline(y=15, line_dash='dash', line_color='red',
                          annotation_text='Swing Threshold', row=3, col=1)

        # Panel 4 — rep durations
        rep_durs = state.get('rep_durations', [])
        if rep_durs:
            fig.add_trace(go.Bar(
                x=[f'Rep {i+1}' for i in range(len(rep_durs))],
                y=rep_durs,
                name='Time per Rep',
                marker_color='indianred'
            ), row=4, col=1)

        fig.update_yaxes(title_text='Degrees (°)',  gridcolor='#1e2030', row=1, col=1)
        fig.update_yaxes(title_text='Degrees (°)',  gridcolor='#1e2030', row=2, col=1)
        fig.update_yaxes(title_text='Pixel Delta',  gridcolor='#1e2030', row=3, col=1)
        fig.update_yaxes(title_text='Seconds',      gridcolor='#1e2030', row=4, col=1)
        fig.update_xaxes(title_text='Time in Video (s)', gridcolor='#1e2030', row=4, col=1)
        fig.update_layout(
            **_LAYOUT,
            title_text='Hammer Curl Biomechanics & Performance Dashboard'
        )

    # ─────────────────────────────────────────────
    # PLANK — 4 panels
    # ─────────────────────────────────────────────
    else:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                '1. Body Alignment Angle & Form Alerts  (ideal = 160deg-195deg)',
                '2. Hip Height Tracker  (flat line = good plank)',
                '3. Head / Neck Angle  (lower = head dropping)',
                '4. Elbow Separation  (higher = more flare)',
            ),
            row_heights=[0.34, 0.22, 0.22, 0.22]
        )

        # Panel 1 — body alignment
        fig.add_trace(go.Scatter(
            x=df_v['Time(s)'], y=df_v['Angle'],
            name='Body Angle', line=dict(color='royalblue', width=2)
        ), row=1, col=1)
        # Good zone shading
        fig.add_hrect(y0=160, y1=195,
                      fillcolor='rgba(34,197,94,0.07)',
                      layer='below', line_width=0,
                      annotation_text='Good Zone',
                      annotation_position='top right',
                      row=1, col=1)
        fig.add_hline(y=160, line_dash='dash', line_color='red',
                      annotation_text='Sag Limit',  row=1, col=1)
        fig.add_hline(y=195, line_dash='dash', line_color='darkorange',
                      annotation_text='Pike Limit', row=1, col=1)
        a = _alert_scatter(df_v, 'Angle')
        if a: fig.add_trace(a, row=1, col=1)

        # Panel 2 — hip height
        if 'Hip_Height' in df_v.columns:
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['Hip_Height'],
                name='Hip Height', line=dict(color='darkorange', width=2)
            ), row=2, col=1)
            fig.add_hline(y=0, line_dash='dot', line_color='white',
                          annotation_text='Level', row=2, col=1)

        # Panel 3 — head/neck angle
        if 'Head_Angle' in df_v.columns:
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['Head_Angle'],
                name='Head Angle', line=dict(color='mediumpurple', width=2)
            ), row=3, col=1)
            fig.add_hline(y=155, line_dash='dash', line_color='red',
                          annotation_text='Drop Threshold', row=3, col=1)

        # Panel 4 — elbow separation
        if 'Elbow_Dist' in df_v.columns:
            fig.add_trace(go.Scatter(
                x=df_v['Time(s)'], y=df_v['Elbow_Dist'],
                name='Elbow Separation', line=dict(color='mediumseagreen', width=2)
            ), row=4, col=1)
            fig.add_hline(y=0.25, line_dash='dash', line_color='red',
                          annotation_text='Flare Threshold', row=4, col=1)

        fig.update_yaxes(title_text='Degrees (°)',       gridcolor='#1e2030', row=1, col=1)
        fig.update_yaxes(title_text='Normalised Ht.',    gridcolor='#1e2030', row=2, col=1)
        fig.update_yaxes(title_text='Degrees (°)',       gridcolor='#1e2030', row=3, col=1)
        fig.update_yaxes(title_text='Separation Ratio',  gridcolor='#1e2030', row=4, col=1)
        fig.update_xaxes(title_text='Time in Video (s)', gridcolor='#1e2030', row=4, col=1)
        fig.update_layout(
            **_LAYOUT,
            title_text='Plank Biomechanics & Performance Dashboard'
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── Form alert summary table (deduplicated) ──────────────────────────────
    df_alert_log = df[df['Alerts'] != ''].copy()
    if not df_alert_log.empty:
        st.markdown('<div class="section-header">⚠️ Form Alert Log</div>',
                    unsafe_allow_html=True)
        # Show unique alert messages with first occurrence time
        summary = (
            df_alert_log
            .groupby('Alerts')
            .agg(First_At=('Time(s)', 'min'),
                 Count=('Alerts', 'count'))
            .reset_index()
            .sort_values('First_At')
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Full log expandable
        with st.expander("Show full alert log"):
            st.dataframe(
                df_alert_log[['Time(s)', 'Reps', 'Alerts']],
                use_container_width=True, hide_index=True
            )
