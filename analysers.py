import cv2
import numpy as np
import mediapipe as mp
import time


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
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
    """Draw a semi-transparent HUD panel on the top-left of the frame."""
    panel_h = len(lines) * 30 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_alert_overlay(frame, alerts, hold_counter):
    """
    Holds the red border + big alert text for hold_counter frames after
    an error fires — so it stays visible for ~2 seconds instead of 1 frame.
    """
    if hold_counter <= 0 or not alerts:
        return
    h, w = frame.shape[:2]
    # Red border — brighter when freshly triggered, fades as counter drops
    intensity = min(hold_counter * 4, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, intensity), 20)
    # Big alert text at the bottom center
    for i, text in enumerate(alerts):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        x = (w - text_size[0]) // 2
        y = h - 60 - (i * 50)
        cv2.rectangle(frame, (x - 10, y - 35), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# SQUAT ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_squat(frame, lm, w, h, state):
    """
    Tracks: knee angle, ROM, hip depth, L/R knee valgus ratio,
    knee-lock detection, rep duration timing.
    """
    FLEXION_THRESHOLD   = 95
    EXTENSION_THRESHOLD = 165
    ROM_TARGET          = 70

    def pt(id): return (int(lm[id].x * w), int(lm[id].y * h))
    def pa(id): return np.array([lm[id].x * w, lm[id].y * h])

    hip   = pa(mp.solutions.pose.PoseLandmark.RIGHT_HIP.value)
    knee  = pa(mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value)
    ankle = pa(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value)

    angle = calculate_angle(hip, knee, ankle)
    state['angles'].append(angle)
    state['min_angle'] = min(state.get('min_angle', 999), angle)
    state['max_angle'] = max(state.get('max_angle', 0), angle)
    rom = state['max_angle'] - state['min_angle']

    # Hip depth — vertical distance between hip and knee (higher = deeper squat)
    hip_depth = abs(hip[1] - knee[1])
    state.setdefault('hip_depth_series', []).append(round(hip_depth, 2))

    # Knee valgus ratio (both sides) — works even from slight angle
    l_knee  = pa(25); r_knee  = pa(26)
    l_ankle = pa(27); r_ankle = pa(28)
    l_hip   = pa(23); r_hip   = pa(24)

    def valgus_ratio(hip_pt, knee_pt, ankle_pt):
        hip_knee_x   = abs(hip_pt[0]   - knee_pt[0])
        knee_ankle_x = abs(knee_pt[0]  - ankle_pt[0])
        return round(knee_ankle_x / max(hip_knee_x + knee_ankle_x, 1), 4)

    l_valgus = valgus_ratio(l_hip, l_knee, l_ankle)
    r_valgus = valgus_ratio(r_hip, r_knee, r_ankle)
    state.setdefault('l_valgus_series', []).append(l_valgus)
    state.setdefault('r_valgus_series', []).append(r_valgus)

    # Rep counting — count on ascent
    if angle < FLEXION_THRESHOLD and state['stage'] == 'up':
        state['stage'] = 'down'
        state['rep_start_time'] = state.get('current_time', 0)
    if angle > EXTENSION_THRESHOLD and state['stage'] == 'down':
        state['stage'] = 'up'
        state['reps'] += 1
        # Store rep duration
        rep_dur = round(state.get('current_time', 0) - state.get('rep_start_time', 0), 2)
        state.setdefault('rep_durations', []).append(rep_dur)
        state['min_angle'] = 999
        state['max_angle'] = 0

    # Alerts
    alerts = []
    if state['stage'] == 'down' and angle < FLEXION_THRESHOLD + 5 and rom < ROM_TARGET:
        alerts.append(('Go Lower!', (0, 80, 255)))
    if angle > 170 and not state.get('knee_lock_alerted', False):
        alerts.append(('Locking Knees!', (0, 80, 255)))
        state['knee_lock_alerted'] = True
    if angle < 170:
        state['knee_lock_alerted'] = False

    # Skeleton
    for p1, p2, col in [
        (pt(23), pt(25), (0, 255, 0)),
        (pt(25), pt(27), (0, 255, 0)),
        (pt(24), pt(26), (0, 200, 0)),
        (pt(26), pt(28), (0, 200, 0)),
        (pt(23), pt(24), (0, 255, 255)),
    ]:
        draw_neon_line(frame, p1, p2, col)
    draw_neon_joint(frame, pt(25), (0, 255, 255))
    draw_neon_joint(frame, pt(26), (0, 200, 220))

    hud = [
        (f'Reps: {state["reps"]}',          (0, 255, 0)),
        (f'Knee Angle: {int(angle)}',        (255, 255, 255)),
        (f'ROM: {int(rom)}',                 (255, 255, 0)),
        (f'Stage: {state["stage"].upper()}', (100, 200, 255)),
    ] + alerts
    overlay_hud(frame, hud)

    # Alert hold logic — keeps overlay for ~2 seconds
    if alerts:
        state['alert_hold_frames'] = 60
        state['held_alerts'] = [a[0] for a in alerts]
    elif state.get('alert_hold_frames', 0) > 0:
        state['alert_hold_frames'] -= 1
    else:
        state['held_alerts'] = []

    draw_alert_overlay(frame, state.get('held_alerts', []), state.get('alert_hold_frames', 0))

    state['current_angle']  = angle
    state['current_alerts'] = [a[0] for a in alerts]
    state['rom']            = rom
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# HAMMER CURL ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_hammer_curl(frame, lm, w, h, state):
    """
    Tracks: dominant arm elbow angle, L/R symmetry, ROM,
    shoulder swing delta, rep duration timing.
    """
    FLEXION_THRESHOLD   = 80
    EXTENSION_THRESHOLD = 150
    ROM_TARGET          = 100
    SWING_THRESHOLD     = 15

    def pt(id): return (int(lm[id].x * w), int(lm[id].y * h))
    def pa(id): return np.array([lm[id].x * w, lm[id].y * h])

    # Dominant side detection (voted over first 30 frames)
    frame_count = state.get('frame_count', 0) + 1
    state['frame_count'] = frame_count
    if frame_count <= 30:
        state['left_votes']  = state.get('left_votes', 0) + (1 if lm[11].visibility > lm[12].visibility else 0)
        state['right_votes'] = state.get('right_votes', 0) + (0 if lm[11].visibility > lm[12].visibility else 1)
    dominant = 'Left' if state.get('left_votes', 0) >= state.get('right_votes', 0) else 'Right'

    l_angle   = calculate_angle(pa(11), pa(13), pa(15))
    r_angle   = calculate_angle(pa(12), pa(14), pa(16))
    dom_angle = l_angle if dominant == 'Left' else r_angle

    state['angles'].append(dom_angle)
    state.setdefault('l_angle_series', []).append(round(l_angle, 2))
    state.setdefault('r_angle_series', []).append(round(r_angle, 2))

    state['min_angle'] = min(state.get('min_angle', 999), dom_angle)
    state['max_angle'] = max(state.get('max_angle', 0), dom_angle)
    rom = state['max_angle'] - state['min_angle']

    # Rep counting
    if dom_angle < FLEXION_THRESHOLD and state['stage'] == 'down':
        state['stage'] = 'up'
        state['rep_start_time'] = state.get('current_time', 0)
    if dom_angle > EXTENSION_THRESHOLD and state['stage'] == 'up':
        state['stage'] = 'down'
        state['reps'] += 1
        rep_dur = round(state.get('current_time', 0) - state.get('rep_start_time', 0), 2)
        state.setdefault('rep_durations', []).append(rep_dur)
        state['min_angle'] = 999
        state['max_angle'] = 0

    # Shoulder swing
    shoulder_y = (pa(11)[1] + pa(12)[1]) / 2
    prev_sy    = state.get('prev_shoulder_y', shoulder_y)
    delta      = abs(shoulder_y - prev_sy)
    swing      = delta > SWING_THRESHOLD
    state['prev_shoulder_y'] = shoulder_y
    state.setdefault('shoulder_delta_series', []).append(round(delta, 2))

    alerts = []
    if state['stage'] == 'up' and dom_angle < FLEXION_THRESHOLD + 5 and rom < ROM_TARGET:
        alerts.append(('Increase ROM!', (0, 80, 255)))
    if swing:
        alerts.append(('Avoid Swinging!', (0, 80, 255)))

    # Skeleton — both arms
    for p1, p2 in [(pt(11), pt(13)), (pt(13), pt(15)), (pt(12), pt(14)), (pt(14), pt(16))]:
        draw_neon_line(frame, p1, p2, (0, 255, 0))
    for j in [13, 14]:
        draw_neon_joint(frame, pt(j), (0, 255, 255))

    hud = [
        (f'Reps: {state["reps"]}',              (0, 255, 0)),
        (f'Elbow Angle: {int(dom_angle)}',       (255, 255, 255)),
        (f'ROM: {int(rom)}',                     (255, 255, 0)),
        (f'L:{int(l_angle)}  R:{int(r_angle)}', (0, 255, 255)),
        (f'Side: {dominant}',                    (180, 180, 255)),
    ] + alerts
    overlay_hud(frame, hud)

    # Alert hold logic
    if alerts:
        state['alert_hold_frames'] = 60
        state['held_alerts'] = [a[0] for a in alerts]
    elif state.get('alert_hold_frames', 0) > 0:
        state['alert_hold_frames'] -= 1
    else:
        state['held_alerts'] = []

    draw_alert_overlay(frame, state.get('held_alerts', []), state.get('alert_hold_frames', 0))

    state['current_angle']  = dom_angle
    state['current_alerts'] = [a[0] for a in alerts]
    state['rom']            = rom
    state['dominant_side']  = dominant
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# PLANK ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_plank(frame, lm, w, h, state):
    """
    Tracks: body alignment angle, hip height (normalised),
    head/neck angle, elbow separation ratio, hold timer.
    """
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

    # Hip height normalised — distance from shoulder to hip relative to body length
    body_len   = max(np.linalg.norm(shoulder_mid - ankle_mid), 1)
    hip_height = (shoulder_mid[1] - hip_mid[1]) / body_len  # positive = hip above shoulder line

    state['angles'].append(body_angle)
    state.setdefault('head_angle_series',  []).append(round(head_angle, 2))
    state.setdefault('elbow_dist_series',  []).append(round(elbow_dist, 4))
    state.setdefault('hip_height_series',  []).append(round(hip_height, 4))

    # Hold timer
    if state.get('hold_start') is None:
        state['hold_start'] = time.time()
    state['hold_duration'] = time.time() - state['hold_start']

    alerts = []
    if body_angle < ALIGNMENT_MIN:
        alerts.append(('Hip Sag!',     (0, 80, 255)))
    elif body_angle > ALIGNMENT_MAX:
        alerts.append(('Hip Pike!',    (0, 80, 255)))
    if head_angle < HEAD_DROP_ANGLE:
        alerts.append(('Head Drop!',   (0, 80, 255)))
    if elbow_dist > ELBOW_WIDE_DIST:
        alerts.append(('Elbow Flare!', (0, 80, 255)))

    # Skeleton
    for p1, p2 in [
        (pt(11), pt(23)), (pt(23), pt(27)),
        (pt(12), pt(24)), (pt(24), pt(28)),
        (pt(11), pt(12)), (pt(23), pt(24)),
        (pt(11), pt(13)), (pt(12), pt(14)),
    ]:
        draw_neon_line(frame, p1, p2, (0, 255, 0))

    hold = state['hold_duration']
    hud = [
        (f'Hold: {hold:.1f}s',             (0, 255, 0)),
        (f'Body Angle: {int(body_angle)}', (255, 255, 255)),
        (f'Head Angle: {int(head_angle)}', (255, 255, 0)),
    ] + alerts
    overlay_hud(frame, hud)

    # Alert hold logic
    if alerts:
        state['alert_hold_frames'] = 60
        state['held_alerts'] = [a[0] for a in alerts]
    elif state.get('alert_hold_frames', 0) > 0:
        state['alert_hold_frames'] -= 1
    else:
        state['held_alerts'] = []

    draw_alert_overlay(frame, state.get('held_alerts', []), state.get('alert_hold_frames', 0))

    state['current_angle']  = body_angle
    state['current_alerts'] = [a[0] for a in alerts]
    state['reps']           = 0
    return state
