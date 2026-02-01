import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from posture_detector import PostureDetector


class PostureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture AI - MVP Industrial")
        self.setGeometry(200, 200, 900, 600)

        self.video_label = QLabel("Video Feed")
        self.video_label.setStyleSheet("background:black")

        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_video = QPushButton("Load Video")
        self.btn_stop = QPushButton("Stop")

        self.status_label = QLabel("Posture: -")
        self.score_label = QLabel("Score: -")
        self.conf_label = QLabel("Confidence: -")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_webcam)
        button_layout.addWidget(self.btn_video)
        button_layout.addWidget(self.btn_stop)

        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.conf_label)

        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.detector = PostureDetector()

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

        posture = analysis["posture"]
        score = analysis["score"]
        confidence = analysis["confidence"]

        if posture == "OK":
            color = (0, 255, 0)
        elif posture == "BAD":
            color = (0, 0, 255)
        elif posture == "HOLD":
            color = (0, 165, 255)  # نارنجی
        else:
            color = (200, 200, 200)

        cv2.putText(frame, f"Posture: {posture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Score: {round(score,2)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"Confidence: {round(confidence,2)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        self.status_label.setText(f"Posture: {posture}")
        self.score_label.setText(f"Score: {round(score,2)}")
        self.conf_label.setText(f"Confidence: {round(confidence,2)}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PostureApp()
    win.show()
    sys.exit(app.exec_())
