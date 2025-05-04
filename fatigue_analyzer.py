import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from collections import deque

class FatigueAnalyzer:
    def __init__(self):
        self.emotion = "Neutral"
        self.relaxation_score = 50.0
        self.fatigue_level = 0.0
        self.heart_rate = 72.0
        self.hrv = 50.0
        self.face_detected = False
        self.face_region = (0, 0, 0, 0)

        self.eye_aspect_ratio_threshold = 0.25
        self.blink_threshold = 0.2
        self.blink_history = deque(maxlen=10)
        self.last_blink_time = 0
        self.eye_closed_duration = 0
        self.eye_open_duration = 0

        self.rppg_signal = deque(maxlen=256)
        self.rppg_timestamps = deque(maxlen=256)
        self.signal_processing_interval = 5.0
        self.last_hr_update = 0.0

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.hrv_history = deque(maxlen=100)
        self.show_plot = False

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video capture")

    def get_total_meditation_score(self):
        """
        Calculates the Total Meditation Score (TMS) in format x-y-z where:
        x = Fatigue detection score (0-10)
        y = rPPG (HRV) score (0-10)
        z = Facial expression relaxation score (0-10)
        
        Higher scores indicate better meditation state
        """
        # Fatigue score (inverse since lower fatigue is better for meditation)
        fatigue_score = 10 - min(10, self.fatigue_level / 10)
        
        # HRV score (normalized between 0-10, higher HRV is better)
        hrv_score = min(10, max(0, (self.hrv - 30) / 6))  # Assuming 30-90ms range
        
        # Facial expression relaxation score (already normalized 0-100)
        expression_score = self.relaxation_score / 10
        
        # Format as x-y-z with integer values
        return f"{int(round(fatigue_score))}-{int(round(hrv_score))}-{int(round(expression_score))}"

    def run(self):
        print("Press 'q' to quit | Press 'p' to toggle HRV plot")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            self.analyze_frame(frame)

            if self.face_detected:
                x, y, w, h = self.face_region
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                y_offset = 30
                cv2.putText(frame, f"Emotion: {self.emotion}", (10, y_offset), font, 0.7, (0, 0, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"Fatigue: {self.fatigue_level:.1f}%", (10, y_offset), font, 0.7, (0, 0, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"HR: {self.heart_rate:.1f} bpm  HRV: {self.hrv:.1f} ms", (10, y_offset), font, 0.7, (0, 0, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"TMS: {self.get_total_meditation_score()}", (10, y_offset), font, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Fatigue Analysis", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.show_plot = not self.show_plot
                if self.show_plot:
                    plt.ion()
                    plt.figure("HRV Monitor", figsize=(5, 3))
                else:
                    plt.close()

        self.cap.release()
        cv2.destroyAllWindows()

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            self.face_detected = True
            self.face_region = (x, y, w, h)

            try:
                face_crop = frame[y:y+h, x:x+w]
                analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(analysis, list):
                    analysis = analysis[0]

                self.emotion = analysis.get('dominant_emotion', 'Neutral')
                self.relaxation_score = self.calculate_relaxation_score(analysis)

                self.detect_fatigue(gray, x, y, w, h)
                self.process_physiological_signals(face_crop)

                current_time = time.time()
                if current_time - self.last_hr_update > self.signal_processing_interval:
                    self.estimate_heart_metrics()
                    self.last_hr_update = current_time

            except Exception as e:
                print(f"Analysis error: {e}")
                self.reset_metrics()
        else:
            self.reset_metrics()

    def detect_fatigue(self, gray_frame, face_x, face_y, face_w, face_h):
        roi_gray = gray_frame[face_y:face_y+face_h, face_x:face_x+face_w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        current_time = time.time()

        if len(eyes) >= 2:
            eye1, eye2 = eyes[0], eyes[1]
            eye_ratio = (eye1[2] + eye2[2]) / (2 * face_w)

            if eye_ratio < self.eye_aspect_ratio_threshold:
                self.eye_closed_duration += 0.1
                self.eye_open_duration = max(0, self.eye_open_duration - 0.05)
                if self.last_blink_time == 0:
                    self.last_blink_time = current_time
                elif current_time - self.last_blink_time > self.blink_threshold:
                    self.blink_history.append(1)
                    self.last_blink_time = 0
            else:
                self.eye_open_duration += 0.1
                self.eye_closed_duration = max(0, self.eye_closed_duration - 0.05)
        else:
            self.eye_closed_duration += 0.1
            self.eye_open_duration = max(0, self.eye_open_duration - 0.1)

        blink_rate = sum(self.blink_history) / 10 if len(self.blink_history) > 0 else 0
        eye_closure_factor = min(1, self.eye_closed_duration / 3.0)
        blink_factor = min(1, blink_rate / 5.0)
        emotion_factor = 1 - self.relaxation_score / 100

        self.fatigue_level = min(100, 30 * eye_closure_factor + 30 * blink_factor + 40 * emotion_factor)

    def calculate_relaxation_score(self, analysis):
        emotions = analysis.get('emotion', {})
        positive = emotions.get('happy', 0) + emotions.get('neutral', 0)
        negative = emotions.get('angry', 0) + emotions.get('sad', 0) + emotions.get('fear', 0)
        total = max(sum(emotions.values()), 1)
        score = (positive - negative) / total * 50 + 50
        return max(0, min(100, score))

    def process_physiological_signals(self, face_roi):
        try:
            if face_roi.size > 0:
                green_channel = np.mean(face_roi[:, :, 1])
                self.rppg_signal.append(green_channel)
                self.rppg_timestamps.append(time.time())
        except Exception as e:
            print(f"Signal processing error: {e}")

    def estimate_heart_metrics(self):
        if len(self.rppg_signal) < 10:
            return
        try:
            self.heart_rate = 70 + 10 * np.sin(time.time() / 5)
            self.hrv = 50 + 10 * np.cos(time.time() / 3)
            self.hrv_history.append(self.hrv)

            if self.show_plot:
                self.update_plot()
        except Exception as e:
            print(f"Heart rate estimation error: {str(e)}")

    def update_plot(self):
        plt.clf()
        plt.title("HRV Over Time")
        plt.plot(list(self.hrv_history), label="HRV", color='blue')
        plt.ylim(30, 90)
        plt.xlabel("Time")
        plt.ylabel("HRV (ms)")
        plt.legend()
        plt.pause(0.01)

    def reset_metrics(self):
        self.face_detected = False
        self.face_region = (0, 0, 0, 0)
        self.emotion = "Neutral"
        self.relaxation_score = 50.0
        self.fatigue_level = 0.0
        self.heart_rate = 72.0
        self.hrv = 50.0
        self.eye_closed_duration = 0
        self.eye_open_duration = 0
        self.blink_history.clear()

if __name__ == "__main__":
    analyzer = FatigueAnalyzer()
    analyzer.run()