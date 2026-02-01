import mediapipe as mp
import numpy as np
import cv2
import time

mp_pose = mp.solutions.pose


class PostureDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, calibration_time=3):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # --- Calibration Base Values ---
        self.calibrated = False
        self.base_forward = 0.1
        self.base_shoulder_diff = 0.02
        self.base_yaw = 0.02
        self.base_pitch = 0.05

        self.calibration_time = calibration_time

        # --- Stability Tracking ---
        self.last_state = "UNKNOWN"
        self.candidate_state = None
        self.state_start_time = time.time()
        self.hold_time = 1.0  # seconds required to confirm posture

    def get_point(self, landmarks, idx):
        try:
            return np.array([landmarks[idx].x, landmarks[idx].y])
        except:
            return None

    def calibrate(self, frame):
        start = time.time()
        yaw_list, pitch_list, forward_list, shoulder_list = [], [], [], []

        while time.time() - start < self.calibration_time:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark

            nose = self.get_point(lm, mp_pose.PoseLandmark.NOSE.value)
            l_shoulder = self.get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            r_shoulder = self.get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

            if nose is None or l_shoulder is None or r_shoulder is None:
                continue

            shoulder_mid = (l_shoulder + r_shoulder) / 2

            yaw_list.append(abs(nose[0] - shoulder_mid[0]))
            pitch_list.append(abs(nose[1] - shoulder_mid[1]))
            forward_list.append(np.linalg.norm(nose - shoulder_mid))
            shoulder_list.append(abs(l_shoulder[1] - r_shoulder[1]))

        if yaw_list: self.base_yaw = np.mean(yaw_list)
        if pitch_list: self.base_pitch = np.mean(pitch_list)
        if forward_list: self.base_forward = np.mean(forward_list)
        if shoulder_list: self.base_shoulder_diff = np.mean(shoulder_list)

        self.calibrated = True

    def analyze(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            result = {"posture": "UNKNOWN", "score": 0.0, "confidence": 0.0}

            if not res.pose_landmarks:
                return result

            lm = res.pose_landmarks.landmark

            nose = self.get_point(lm, mp_pose.PoseLandmark.NOSE.value)
            l_shoulder = self.get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            r_shoulder = self.get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

            if nose is None or l_shoulder is None or r_shoulder is None:
                return result

            shoulder_mid = (l_shoulder + r_shoulder) / 2

            # --- Motion Metrics ---
            yaw = abs(nose[0] - shoulder_mid[0])             # Left / Right tilt
            pitch = abs(nose[1] - shoulder_mid[1])           # Up / Down tilt
            forward_head = np.linalg.norm(nose - shoulder_mid)  # Forward lean
            shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])  # Shoulder tilt

            # --- Calibration ---
            if not self.calibrated:
                self.calibrate(frame)

            yaw_score = np.clip((yaw - self.base_yaw) / (self.base_yaw + 1e-5), 0, 1)
            pitch_score = np.clip((pitch - self.base_pitch) / (self.base_pitch + 1e-5), 0, 1)
            forward_score = np.clip((forward_head - self.base_forward) / (self.base_forward + 1e-5), 0, 1)
            shoulder_score = np.clip((shoulder_diff - self.base_shoulder_diff) / (self.base_shoulder_diff + 1e-5), 0, 1)

            # --- Weighted Score ---
            score = (
                yaw_score * 0.30 +
                pitch_score * 0.30 +
                forward_score * 0.25 +
                shoulder_score * 0.15
            )

            raw_state = "OK" if score < 0.1 else "BAD"
            current_time = time.time()

            # --- Candidate State Tracking ---
            if raw_state != self.last_state:
                if self.candidate_state != raw_state:
                    self.candidate_state = raw_state
                    self.state_start_time = current_time
            else:
                self.candidate_state = None

            # --- Confirm state only if stable for hold_time ---
            if self.candidate_state:
                elapsed = current_time - self.state_start_time
                if elapsed >= self.hold_time:
                    self.last_state = self.candidate_state
                    self.candidate_state = None

            final_state = self.last_state

            # --- Confidence ---
            if self.candidate_state:
                elapsed = current_time - self.state_start_time
                confidence = min(1.0, elapsed / self.hold_time)
            else:
                confidence = 1.0

            result["posture"] = final_state
            result["score"] = float(score)
            result["confidence"] = confidence

            return result

        except Exception as e:
            print("Posture error:", e)
            return {"posture": "UNKNOWN", "score": 0.0, "confidence": 0.0}
