import sys
import cv2
import time

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from posture_detector import PostureDetector


class PostureApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PostureDetector Pro")
        self.setGeometry(200, 200, 960, 640)

        # Video display
        self.video_label = QLabel("Camera Feed")
        self.video_label.setStyleSheet("background:black; color:white; font-size:16px;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_video = QPushButton("Load Video")
        self.btn_stop = QPushButton("Stop")

        # Status labels
        self.status_label = QLabel("Posture: -")
        self.score_label = QLabel("Score: -")
        self.conf_label = QLabel("Confidence: -")
        self.timer_label = QLabel("Stable Time: 0s")

        for label in [self.status_label, self.score_label, self.conf_label, self.timer_label]:
            label.setStyleSheet("font-size:15px;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_webcam)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_stop)

        layout.addLayout(btn_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.timer_label)

        self.setLayout(layout)

        # Camera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Detector
        self.detector = PostureDetector()

        # Stable posture tracking
        self.last_posture = "UNKNOWN"
        self.stable_posture = "UNKNOWN"
        self.state_start_time = time.time()
        self.required_stable_time = 1  # seconds

        # Buttons events
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_stop.clicked.connect(self.stop)

    def start_webcam(self):
        self.stop()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def load_video(self):
        self.stop()
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.timer.start(30)

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        analysis = self.detector.analyze(frame)

        posture = analysis.get("posture", "UNKNOWN")
        score = analysis.get("score", 0.0)
        confidence = analysis.get("confidence", 0.0)

        # --------- Stability Filter (5 seconds rule) ----------
        now = time.time()

        if posture != self.last_posture:
            self.state_start_time = now
            self.last_posture = posture

        stable_time = now - self.state_start_time

        if stable_time >= self.required_stable_time:
            self.stable_posture = posture

        # Color logic
        if self.stable_posture == "OK":
            color = (0, 200, 0)
        elif self.stable_posture == "BAD":
            color = (0, 0, 255)
        else:
            color = (180, 180, 180)

        # Overlay text on frame
        cv2.putText(frame, f"Posture: {self.stable_posture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Score: {round(score, 3)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Confidence: {round(confidence, 2)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Stable Time: {int(stable_time)}s", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Update UI labels
        self.status_label.setText(f"Posture: {self.stable_posture}")
        self.score_label.setText(f"Score: {round(score, 3)}")
        self.conf_label.setText(f"Confidence: {round(confidence, 2)}")
        self.timer_label.setText(f"Stable Time: {int(stable_time)}s")

        # Convert frame to Qt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PostureApp()
    window.show()
    sys.exit(app.exec_())
