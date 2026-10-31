"""Webcam -> MediaPipe Hands -> /hand/state (pa_interfaces/HandState) via rosbridge.

Runs on the Windows host (the Docker container cannot reach the webcam easily):

    pip install -r host/requirements.txt
    python host/hand_bridge.py --crop 0.2 0.1 0.8 0.9 --mirror

Privacy: point the camera down at your desk so it frames ONLY your hands
(never your face). Frames are processed in memory and never saved. The preview
window by default draws only the hand skeleton and the table square on a black
canvas; `--show-camera` shows the real image (check the framing first).

Calibration: the `--crop` rectangle (normalized image coords u0 v0 u1 v1,
u right / v down) is mapped onto the table square of scene.yaml
(workspace.table_half_size). A palm outside that rectangle counts as "no hand
over the table". Keys in the preview: q / Esc quit, m toggles mirroring.
"""

import argparse
import time

from hand_mapping import (Calibration, HandSmoother, image_to_table, load_table_half_size,
                          palm_center, table_to_image)

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--crop", type=float, nargs=4, default=[0.0, 0.0, 1.0, 1.0],
                    metavar=("U0", "V0", "U1", "V1"), help="image rectangle mapped to the table")
    ap.add_argument("--mirror", action="store_true", help="flip left/right")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate the image frame (deg, CCW) before mapping")
    ap.add_argument("--half-size", type=float, default=None,
                    help="table half size [m] (default: scene.yaml workspace.table_half_size)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--topic", default="/hand/state")
    ap.add_argument("--msg-type", default="pa_interfaces/HandState")
    ap.add_argument("--rate", type=float, default=15.0, help="publish rate [Hz]")
    ap.add_argument("--alpha", type=float, default=0.4, help="smoothing factor (1 = none)")
    ap.add_argument("--min-detection", type=float, default=0.6)
    ap.add_argument("--no-ros", action="store_true", help="do not connect to rosbridge (camera test)")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--show-camera", action="store_true",
                    help="preview the real camera image (make sure no face is in view)")
    return ap.parse_args()


def connect(args):
    import roslibpy
    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run(timeout=10)
    topic = roslibpy.Topic(client, args.topic, args.msg_type)
    topic.advertise()
    print(f"connected to ws://{args.host}:{args.port}, publishing {args.topic}")
    return client, topic


def make_msg(present, x, y):
    now = time.time()
    return {
        "header": {"stamp": {"sec": int(now), "nanosec": int((now % 1) * 1e9)}, "frame_id": "world"},
        "present": bool(present),
        "x": float(x),
        "y": float(y),
    }


def draw_preview(cv2, np, frame, cal, landmarks, state, show_camera):
    h, w = frame.shape[:2]
    canvas = frame.copy() if show_camera else np.zeros_like(frame)
    hs = cal.half_size
    corners = [table_to_image(x, y, cal) for x, y in ((-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs))]
    pts = np.array([[int(u * w), int(v * h)] for u, v in corners], dtype=np.int32)
    cv2.polylines(canvas, [pts], True, (0, 200, 255), 2)
    top = table_to_image(0.0, hs, cal)  # world +y edge
    cv2.putText(canvas, "+y", (int(top[0] * w) - 10, int(top[1] * h) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    if landmarks is not None:
        p = [(int(l.x * w), int(l.y * h)) for l in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, p[a], p[b], (0, 255, 0), 2)
        u, v = palm_center(landmarks)
        cv2.circle(canvas, (int(u * w), int(v * h)), 8, (0, 0, 255), -1)
    present, x, y = state
    txt = f"hand {'YES' if present else 'no '}  x={x * 100:+.1f}cm y={y * 100:+.1f}cm" \
          f"{'  [mirror]' if cal.mirror else ''}"
    cv2.putText(canvas, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def main():
    args = parse_args()
    import cv2
    import mediapipe as mp
    import numpy as np

    cal = Calibration(crop=tuple(args.crop),
                      half_size=args.half_size or load_table_half_size(),
                      mirror=args.mirror, rotate_deg=args.rotate)
    print(f"calibration: {cal}")
    smoother = HandSmoother(alpha=args.alpha)

    client = topic = None
    if not args.no_ros:
        client, topic = connect(args)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")

    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                                     min_detection_confidence=args.min_detection,
                                     min_tracking_confidence=0.5)
    period = 1.0 / args.rate
    next_pub = time.monotonic()
    state = (False, 0.0, 0.0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                time.sleep(0.1)
                continue
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = res.multi_hand_landmarks[0].landmark if res.multi_hand_landmarks else None
            xy = None
            if landmarks is not None:
                x, y, inside = image_to_table(*palm_center(landmarks), cal)
                if inside:
                    xy = (x, y)
            state = smoother.update(xy)

            now = time.monotonic()
            if now >= next_pub:
                next_pub = max(next_pub + period, now)
                if topic is not None:
                    if not client.is_connected:
                        print("rosbridge disconnected")
                    else:
                        topic.publish(make_msg(*state))

            if not args.no_preview:
                cv2.imshow("hand_bridge (q: quit, m: mirror)",
                           draw_preview(cv2, np, frame, cal, landmarks, state, args.show_camera))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    cal = Calibration(cal.crop, cal.half_size, not cal.mirror, cal.rotate_deg)
            del frame  # never kept or written to disk
    except KeyboardInterrupt:
        pass
    finally:
        if topic is not None:
            topic.publish(make_msg(False, *state[1:]))  # release the robot
            time.sleep(0.2)
            topic.unadvertise()
            client.terminate()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
