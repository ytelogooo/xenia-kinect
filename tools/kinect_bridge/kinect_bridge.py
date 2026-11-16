#!/usr/bin/env python3
"""Camera-based Kinect skeleton bridge for xenia-canary.

Reads a webcam (or a phone used as a webcam), tracks one body with Google
MediaPipe Pose, converts it to the Kinect v1 20-joint skeleton and sends it to
Xenia over UDP (enable it there with kinect_skeleton_udp_port = 27050).

    python kinect_bridge.py --list            # find your camera's index
    python kinect_bridge.py --camera 1        # run (press Q in the preview)

It also sends the camera picture (XKCL packets), lined up with the skeleton
the way a Kinect's colour camera is, for the games' player pictures, face
squares and recordings (Kinect Sports' Show Off & Share, Just Dance 4's
Autodance).

Coordinates sent are Kinect skeleton space in metres: +x = the sensor's left
(the player's right when facing it), +y up, +z away from the sensor, with the
emulated sensor 0.8 m above the floor.
"""

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

EMULATED_SENSOR_HEIGHT_M = 0.8  # must match the emulated depth scene in Xenia

# MediaPipe Pose landmark indices (left/right = the person's own side).
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX = 17, 18, 19, 20
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# Kinect NUI_SKELETON_POSITION_INDEX order.
KINECT_JOINTS = [
    "hip_center", "spine", "shoulder_center", "head",
    "shoulder_left", "elbow_left", "wrist_left", "hand_left",
    "shoulder_right", "elbow_right", "wrist_right", "hand_right",
    "hip_left", "knee_left", "ankle_left", "foot_left",
    "hip_right", "knee_right", "ankle_right", "foot_right",
]

DRAW_BONES = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (L_ANKLE, L_FOOT), (R_ANKLE, R_FOOT),
]


def ensure_model(kind):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f"pose_landmarker_{kind}.task")
    if not os.path.exists(path):
        print(f"Downloading the MediaPipe pose model ({kind})...")
        urllib.request.urlretrieve(MODEL_URLS[kind], path)
    return path


def open_camera(index, width, height):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


class LatestFrame:
    """Reads the camera on its own thread and keeps only the newest frame.

    Read directly, a camera (or a phone camera app's driver) queues frames
    while tracking takes longer than a frame, and the bridge then works on
    pictures that are a few frames old: a delay that never catches up.
    """

    def __init__(self, cap):
        self.cap = cap
        self.condition = threading.Condition()
        self.frame = None
        self.number = 0
        self.taken = 0
        self.stopped = False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            ok, frame = self.cap.read()
            with self.condition:
                if not ok:
                    self.stopped = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.number += 1
                self.condition.notify_all()

    def read(self, timeout=5.0):
        """The newest frame not handed out yet (waits for one)."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.stopped or self.number != self.taken, timeout)
            if self.stopped or self.number == self.taken:
                return False, None, 0
            dropped = self.number - self.taken - 1
            self.taken = self.number
            return True, self.frame, dropped


def camera_delivers_picture(cap, attempts=8, time_limit=4.0):
    """True if the camera returns frames that aren't all black."""
    start = time.monotonic()
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.max() > 10:
            return True
        if time.monotonic() - start > time_limit:
            break
    return False


def open_working_camera(index, width, height):
    """Opens the camera, falling back to 640x480 if the requested size only
    gives black frames (some phone webcam apps, e.g. EpocCam's free mode,
    refuse higher resolutions)."""
    sizes = [(width, height)]
    if (width, height) != (640, 480):
        sizes.append((640, 480))
    for w, h in sizes:
        cap = open_camera(index, w, h)
        if cap is None:
            return None
        if camera_delivers_picture(cap):
            if (w, h) != (width, height):
                print(f"Camera gave no picture at {width}x{height}; "
                      f"using {w}x{h}.")
            return cap
        cap.release()
    print("Warning: the camera only delivers black frames. Is the phone app "
          "connected, and is no other program using the camera?")
    return open_camera(index, width, height)


def list_cameras():
    print("Probing camera indices 0-7 (this can take a few seconds)...")
    for i in range(8):
        cap = open_camera(i, 1280, 720)
        if cap is None:
            continue
        ok, frame = cap.read()
        if ok:
            print(f"  camera {i}: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()


def lerp(a, b, t):
    return a + (b - a) * t


def to_kinect(world, image, visibility, width, height, fx, camera_height, state):
    """Returns (20x3 joints in metres, 20 joint states) in Kinect space."""
    w = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64)
    px = np.array([[p.x * width, p.y * height] for p in image], dtype=np.float64)

    hip_w = (w[L_HIP] + w[R_HIP]) / 2
    sh_w = (w[L_SHOULDER] + w[R_SHOULDER]) / 2
    hip_px = (px[L_HIP] + px[R_HIP]) / 2
    sh_px = (px[L_SHOULDER] + px[R_SHOULDER]) / 2

    # Distance from apparent size: metric lengths projected on the image plane
    # (torso + shoulder width) against their length in pixels.
    metric = (np.linalg.norm((sh_w - hip_w)[:2]) +
              np.linalg.norm((w[L_SHOULDER] - w[R_SHOULDER])[:2]))
    pixels = (np.linalg.norm(sh_px - hip_px) +
              np.linalg.norm(px[L_SHOULDER] - px[R_SHOULDER]))
    if pixels < 1.0:
        return None
    z = min(max(fx * metric / pixels, 0.8), 4.5)
    state["z"] = z if state.get("z") is None else lerp(state["z"], z, 0.25)
    z = state["z"]

    cx, cy = width / 2, height / 2
    hip = np.array([
        -(hip_px[0] - cx) * z / fx,  # image right = sensor's right = -x
        -(hip_px[1] - cy) * z / fx + (camera_height - EMULATED_SENSOR_HEIGHT_M),
        z,
    ])

    def joint(index):
        rel = w[index] - hip_w  # MediaPipe world: x right, y down, z away
        return hip + np.array([-rel[0], -rel[1], rel[2]])

    def mid(a, b):
        return (joint(a) + joint(b)) / 2

    hip_center = hip
    shoulder_center = mid(L_SHOULDER, R_SHOULDER)
    joints = [
        hip_center,
        lerp(hip_center, shoulder_center, 0.3),
        shoulder_center,
        mid(L_EAR, R_EAR),
        joint(L_SHOULDER), joint(L_ELBOW), joint(L_WRIST), mid(L_INDEX, L_PINKY),
        joint(R_SHOULDER), joint(R_ELBOW), joint(R_WRIST), mid(R_INDEX, R_PINKY),
        joint(L_HIP), joint(L_KNEE), joint(L_ANKLE), joint(L_FOOT),
        joint(R_HIP), joint(R_KNEE), joint(R_ANKLE), joint(R_FOOT),
    ]
    sources = [
        (L_HIP, R_HIP), (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER),
        (L_SHOULDER, R_SHOULDER),
        (L_EAR, R_EAR), (L_SHOULDER,), (L_ELBOW,), (L_WRIST,),
        (L_INDEX, L_PINKY), (R_SHOULDER,), (R_ELBOW,), (R_WRIST,),
        (R_INDEX, R_PINKY), (L_HIP,), (L_KNEE,), (L_ANKLE,), (L_FOOT,),
        (R_HIP,), (R_KNEE,), (R_ANKLE,), (R_FOOT,),
    ]
    states = []
    for src in sources:
        v = min(visibility[i] for i in src)
        states.append(2 if v > 0.6 else 1)

    # A single camera judges how far a hand reaches towards it poorly, so
    # Kinect "push to press" gestures barely register. Exaggerate forward
    # reach (towards the camera, relative to the shoulder) by push_gain.
    joints = np.array(joints)
    gain = state.get("push_gain", 1.0)
    if gain != 1.0:
        for shoulder, arm in ((4, (5, 6, 7)), (8, (9, 10, 11))):
            for j, g in zip(arm, (math.sqrt(gain), gain, gain)):
                forward = joints[j][2] - joints[shoulder][2]
                if forward < 0:
                    joints[j][2] = joints[shoulder][2] + forward * g

    # A hand MediaPipe can't see (behind the back, hidden by the body) is
    # guessed, often in front of the body, where games take it for the hand
    # you are using. Keep hidden arms behind the hips instead, like Kinect's
    # inferred joints.
    joints = np.array(joints)
    for side in ((5, 6, 7), (9, 10, 11)):  # elbow, wrist, hand
        if states[side[1]] < 2 and states[side[2]] < 2:
            for j in side[1:]:
                joints[j][2] = max(joints[j][2], hip[2] + 0.15)

    smoothed = state.get("joints")
    joints = np.array(joints)
    if smoothed is not None:
        joints = lerp(smoothed, joints, state["alpha"])
    state["joints"] = joints
    return joints, states


def place_on_floor(joints, world, calib, now):
    """Sets the skeleton's height so the feet stand on the emulated floor.

    The camera's height and tilt are unknown (a tilted camera shifts people
    up or down more the further away they are), so the standing height comes
    from MediaPipe's metric skeleton: hips above the lowest toe. Fast vertical
    movement in the image (jumping) is added on top, while slow drift
    (walking towards a tilted camera) is filtered out.
    """
    toes = max(world[L_FOOT].y, world[R_FOOT].y, world[L_HEEL].y,
               world[R_HEEL].y)  # MediaPipe world: y points down
    hip_y_world = (world[L_HIP].y + world[R_HIP].y) / 2
    standing_hip = -EMULATED_SENSOR_HEIGHT_M + max(0.2, toes - hip_y_world)

    raw = joints[0][1]  # hip height from the camera image (tilt-affected)
    last = calib.get("time")
    if calib.get("slow") is None or last is None or now - last > 0.5:
        calib["slow"] = raw
    else:
        k = min(1.0, (now - last) / 1.5)  # ~1.5 s time constant
        calib["slow"] += (raw - calib["slow"]) * k
    calib["time"] = now
    jump = min(max(raw - calib["slow"], -0.4), 0.8)
    joints[:, 1] += (standing_hip + jump) - raw


# The Kinect's 640x480 colour camera: focal length in pixels. The picture is
# sent as if taken by it, so Xenia (and the games) find the player where the
# skeleton says, using the Kinect's own skeleton-to-colour mapping.
KINECT_COLOUR_FOCAL = 531.15
COLOUR_WIDTH, COLOUR_HEIGHT = 640, 480
COLOUR_ROWS_PER_PACKET = 16


class ColourSender:
    """Sends the camera picture to Xenia, as the Kinect SDK bridge does.

    XKCL v1: "XKCL", u32 version 1, u32 frame, u16 width, u16 height,
    u16 first row, u16 rows, then rows * width * 3 bytes (B, G, R).
    XKFO v1 (twice a second): "XKFO", u32 version 1, u32 flags (1 = whole
    picture), f32 face offset in metres.
    From Xenia, XKCM v1: "XKCM", u32 version 1, u32 flags (1 = the game
    wants the whole picture for the next 3 s, it crops the faces itself).
    Little-endian.
    """

    def __init__(self, sock, target, args):
        self.sock = sock
        self.target = target
        self.face_crop = not args.no_face_crop
        self.face_size = args.face_size
        self.face_offset = args.face_offset
        self.flip = not args.colour_no_flip
        self.every = max(1, args.colour_every)
        self.counter = 0
        self.frame_number = 0
        self.whole_until = 0.0
        self.last_settings = 0.0
        self.dy = None  # vertical shift lining the picture up with the skeleton
        self.crop = None
        threading.Thread(target=self.receive, daemon=True).start()

    def receive(self):
        while True:
            try:
                p, _ = self.sock.recvfrom(2048)
            except OSError:
                time.sleep(0.2)
                continue
            if len(p) >= 12 and p[:4] == b"XKCM" and \
                    struct.unpack_from("<I", p, 4)[0] == 1:
                whole = struct.unpack_from("<I", p, 8)[0] & 1
                was = self.wants_whole()
                self.whole_until = time.monotonic() + 3.0 if whole else 0.0
                if bool(whole) != was:
                    self.send_settings()

    def wants_whole(self):
        return time.monotonic() < self.whole_until

    def cropping(self):
        return self.face_crop and not self.wants_whole()

    def send_settings(self):
        flags = 0 if self.face_crop else 1
        try:
            self.sock.sendto(struct.pack("<4sIIf", b"XKFO", 1, flags,
                                         self.face_offset), self.target)
        except OSError:
            pass
        self.last_settings = time.monotonic()

    def picture(self, frame, fx, head_px, head):
        """The camera frame as the Kinect's colour camera would see it.

        head_px: the head's (x, y) in the camera frame; head: its Kinect
        skeleton position. Horizontally the skeleton comes from the picture
        (+x = the sensor's left, shown on the right as in a Kinect's
        picture); vertically it is moved to stand on the emulated floor, so
        the picture is moved to match.
        """
        height, width = frame.shape[:2]
        k = KINECT_COLOUR_FOCAL / fx
        sx = -k if self.flip else k
        if head_px is not None and head is not None and head[2] > 0.3:
            target_y = (COLOUR_HEIGHT / 2 -
                        head[1] * KINECT_COLOUR_FOCAL / head[2])
            dy = target_y - (COLOUR_HEIGHT / 2 + (head_px[1] - height / 2) * k)
            self.dy = dy if self.dy is None else lerp(self.dy, dy, 0.1)
        dy = self.dy or 0.0
        m = np.float32([
            [sx, 0, COLOUR_WIDTH / 2 - sx * width / 2],
            [0, k, COLOUR_HEIGHT / 2 - k * height / 2 + dy]])
        return cv2.warpAffine(frame, m, (COLOUR_WIDTH, COLOUR_HEIGHT),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    def face(self, picture, head):
        """A square around the head stretched to 640x480, as the SDK bridge
        sends it (the games squeeze it back into their square slots)."""
        if head is None or head[2] < 0.5:
            self.crop = None
            return picture
        f = KINECT_COLOUR_FOCAL
        x = COLOUR_WIDTH / 2 + head[0] * f / head[2]
        y = COLOUR_HEIGHT / 2 - head[1] * f / head[2] + \
            self.face_offset * f / head[2]
        side = min(max(self.face_size * f / head[2], 64.0), COLOUR_HEIGHT)
        if self.crop is None:
            self.crop = [x, y, side]
        else:
            self.crop = [lerp(a, b, 0.4) for a, b in zip(self.crop, (x, y, side))]
        x, y, side = self.crop
        x0 = min(max(x - side / 2, 0), COLOUR_WIDTH - side)
        y0 = min(max(y - side / 2, 0), COLOUR_HEIGHT - side)
        sx = side / COLOUR_WIDTH
        sy = side / COLOUR_HEIGHT
        m = np.float32([[sx, 0, x0], [0, sy, y0]])
        return cv2.warpAffine(picture, m, (COLOUR_WIDTH, COLOUR_HEIGHT),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def send(self, frame, fx, head_px, head):
        now = time.monotonic()
        if now - self.last_settings >= 0.5:
            self.send_settings()
        # Line up every frame (the vertical shift follows the player), send
        # every Nth.
        picture = self.picture(frame, fx, head_px, head)
        self.counter += 1
        if self.counter < self.every:
            return picture
        self.counter = 0
        sent = self.face(picture, head) if self.cropping() else picture
        self.frame_number = (self.frame_number + 1) & 0xFFFFFFFF
        data = np.ascontiguousarray(sent).tobytes()
        row_bytes = COLOUR_WIDTH * 3
        for first in range(0, COLOUR_HEIGHT, COLOUR_ROWS_PER_PACKET):
            rows = min(COLOUR_ROWS_PER_PACKET, COLOUR_HEIGHT - first)
            header = struct.pack("<4sIIHHHH", b"XKCL", 1, self.frame_number,
                                 COLOUR_WIDTH, COLOUR_HEIGHT, first, rows)
            try:
                self.sock.sendto(header + data[first * row_bytes:
                                               (first + rows) * row_bytes],
                                 self.target)
            except OSError:
                pass
        return picture


def make_packet(sequence, joints, states):
    tracked = joints is not None
    if not tracked:
        joints = np.zeros((20, 3))
        states = [0] * 20
    flat = [float(v) for v in np.asarray(joints).reshape(-1)]
    return (struct.pack("<4sIII", b"XKSK", 1, sequence, 1 if tracked else 0) +
            struct.pack("<60f", *flat) + bytes(states))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="list cameras and exit")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hfov", type=float, default=70.0,
                        help="camera horizontal field of view in degrees "
                             "(affects distance estimates; default 70)")
    parser.add_argument("--camera-height", type=float, default=None,
                        help="camera height above the floor in metres "
                             "(default: keep your feet on the floor "
                             "automatically, whatever the camera's height "
                             "and tilt)")
    parser.add_argument("--invert-x", action="store_true",
                        help="diagnostic: flip left/right positions without "
                             "swapping which hand is which")
    parser.add_argument("--mirror", action="store_true",
                        help="the camera app already mirrors the picture")
    parser.add_argument("--model", choices=MODEL_URLS, default="full",
                        help="lite = fastest, heavy = most accurate")
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="0 = frozen, 1 = no smoothing (default 0.5)")
    parser.add_argument("--push-gain", type=float, default=1.6,
                        help="exaggerate how far your hands reach towards "
                             "the camera, for push-to-press menus "
                             "(1 = off, default 1.6)")
    parser.add_argument("--hold", type=float, default=0.5,
                        help="seconds to keep the last skeleton when tracking "
                             "drops out (default 0.5)")
    parser.add_argument("--no-colour", action="store_true",
                        help="don't send the camera picture")
    parser.add_argument("--colour-every", type=int, default=1,
                        help="send every Nth camera picture (default 1)")
    parser.add_argument("--colour-no-flip", action="store_true",
                        help="send the picture as the camera sees it instead "
                             "of mirrored like a Kinect's (F in the preview "
                             "toggles it)")
    parser.add_argument("--no-face-crop", action="store_true",
                        help="always send the whole picture, never a square "
                             "around the face")
    parser.add_argument("--face-size", type=float, default=0.30,
                        help="width of the face square in metres "
                             "(default 0.30; X/Z in the preview)")
    parser.add_argument("--face-offset", type=float, default=0.08,
                        help="move the face square down (metres, default "
                             "0.08; W/S in the preview)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27050)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    model = ensure_model(args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = open_working_camera(args.camera, args.width, args.height)
    if cap is None:
        sys.exit(f"Could not open camera {args.camera}; try --list.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 0))  # Xenia answers to this address (XKCM)
    target = (args.host, args.port)
    colour = None if args.no_colour else ColourSender(sock, target, args)
    state = {"alpha": min(max(args.smoothing, 0.05), 1.0),
             "push_gain": max(args.push_gain, 1.0)}
    calib = {}
    sequence = 0
    start = time.monotonic()
    last_ts = -1
    last_good = None
    fps, fps_time, fps_frames = 0.0, time.monotonic(), 0
    skipped, skipped_rate = 0, 0.0
    reader = LatestFrame(cap)
    print(f"Sending skeletons to {target[0]}:{target[1]}. Press Q in the "
          "preview window to quit.")

    while True:
        ok, frame, dropped = reader.read()
        skipped += dropped
        if not ok:
            print("Camera stopped delivering frames.")
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        camera_frame = frame
        if not args.no_preview:
            frame = frame.copy()  # the preview draws on this one
        height, width = frame.shape[:2]
        fx = (width / 2) / math.tan(math.radians(args.hfov) / 2)

        ts = int((time.monotonic() - start) * 1000)
        ts = max(ts, last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        joints = states = None
        head_px = None
        if result.pose_world_landmarks:
            image_lm = result.pose_landmarks[0]
            head_px = ((image_lm[L_EAR].x + image_lm[R_EAR].x) / 2 * width,
                       (image_lm[L_EAR].y + image_lm[R_EAR].y) / 2 * height)
            visibility = [p.visibility for p in image_lm]
            converted = to_kinect(result.pose_world_landmarks[0], image_lm,
                                  visibility, width, height, fx,
                                  EMULATED_SENSOR_HEIGHT_M, state)
            if converted is not None:
                joints, states = converted
                joints = joints.copy()
                if args.invert_x:
                    joints[:, 0] = -joints[:, 0]
                if args.camera_height is not None:
                    joints[:, 1] += args.camera_height - EMULATED_SENSOR_HEIGHT_M
                else:
                    place_on_floor(joints, result.pose_world_landmarks[0],
                                   calib, time.monotonic())
            if not args.no_preview:
                pts = [(int(p.x * width), int(p.y * height)) for p in image_lm]
                for a, b in DRAW_BONES:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 3)
                # Label the hands as Xenia sees them; if "R" is on your left
                # hand, restart with --mirror.
                for index, label in ((R_WRIST, "R"), (L_WRIST, "L")):
                    cv2.putText(frame, label, (pts[index][0] + 10,
                                               pts[index][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        else:
            state.pop("joints", None)
            state.pop("z", None)

        # Bridge short tracking dropouts (dim rooms, motion blur): keep
        # sending the last skeleton for a moment instead of making the
        # player vanish, which cancels games' hover-to-select timers.
        now = time.monotonic()
        if joints is not None:
            last_good = (now, joints, states)
        elif last_good is not None and now - last_good[0] < args.hold:
            joints, states = last_good[1], last_good[2]
        sock.sendto(make_packet(sequence, joints, states), target)
        sequence += 1
        if colour is not None:
            tracked_now = joints is not None and head_px is not None
            colour.send(camera_frame, fx, head_px if tracked_now else None,
                        joints[3] if tracked_now else None)

        fps_frames += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = fps_frames / (now - fps_time)
            skipped_rate = skipped / (now - fps_time)
            fps_time, fps_frames, skipped = now, 0, 0

        if not args.no_preview:
            if joints is not None:
                hip, hand = joints[0], joints[11]
                # side: + = to your right of the camera's centre line.
                text = (f"tracked  dist {hip[2]:.2f} m  side {hip[0]:+.2f} m  "
                        f"hips {hip[1] + EMULATED_SENSOR_HEIGHT_M:.2f} m  "
                        f"{fps:.0f} fps")
                if skipped_rate >= 0.5:
                    # Tracking slower than the camera: older frames skipped
                    # so the picture stays current.
                    text += f" (+{skipped_rate:.0f} skipped)"
                color = (0, 255, 0)
            else:
                text = f"no player  {fps:.0f} fps"
                color = (0, 0, 255)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color, 2)
            if colour is not None:
                mode = ("whole picture (the game's)" if not colour.cropping()
                        else "face square")
                info = (f"picture: {mode}{', mirrored' if colour.flip else ''}"
                        f"  face size {colour.face_size:.2f} m (X/Z)  "
                        f"offset {colour.face_offset:+.2f} m (W/S)  F flips")
                cv2.putText(frame, info, (20, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Kinect bridge (Q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if colour is not None:
                if key in (ord("w"), ord("W"), ord("s"), ord("S")):
                    step = 0.01 if key in (ord("w"), ord("W")) else -0.01
                    colour.face_offset = min(max(colour.face_offset + step,
                                                 -0.3), 0.3)
                    colour.send_settings()
                elif key in (ord("x"), ord("X"), ord("z"), ord("Z")):
                    step = 0.02 if key in (ord("x"), ord("X")) else -0.02
                    colour.face_size = min(max(colour.face_size + step,
                                               0.15), 1.0)
                elif key in (ord("f"), ord("F")):
                    colour.flip = not colour.flip

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
