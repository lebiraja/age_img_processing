import cv2
import numpy as np
import time
from typing import Tuple, List

# Configs (could be moved to YAML)
MODEL_CONFIG = {
    "age_prototxt": "age_deploy.prototxt",
    "age_model": "age_net.caffemodel",
    "gender_prototxt": "gender_deploy.prototxt",
    "gender_model": "gender_net.caffemodel",
    "face_prototxt": "deploy.prototxt",
    "face_model": "res10_300x300_ssd_iter_140000.caffemodel",
    "mean_values": (78.4263377603, 87.7689143744, 114.895847746),
    "age_list": ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'],
    "gender_list": ['Male', 'Female']
}

class AgeGenderDetector:
    def __init__(self):
        self.age_net = cv2.dnn.readNetFromCaffe(MODEL_CONFIG["age_prototxt"], MODEL_CONFIG["age_model"])
        self.gender_net = cv2.dnn.readNetFromCaffe(MODEL_CONFIG["gender_prototxt"], MODEL_CONFIG["gender_model"])
        self.face_net = cv2.dnn.readNetFromCaffe(MODEL_CONFIG["face_prototxt"], MODEL_CONFIG["face_model"])
        
    def predict_age_gender(self, face_img: np.ndarray) -> Tuple[str, float, str, float]:
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_CONFIG["mean_values"], swapRB=False)
        
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = MODEL_CONFIG["gender_list"][gender_preds[0].argmax()]
        gender_conf = gender_preds[0].max()
        
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = MODEL_CONFIG["age_list"][age_preds[0].argmax()]
        age_conf = age_preds[0].max()
        
        return gender, gender_conf, age, age_conf

def main():
    detector = AgeGenderDetector()
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)  # Mirror effect
        frame_count += 1
        
        # Process every 3rd frame for better performance
        if frame_count % 3 == 0:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            
            detector.face_net.setInput(blob)
            detections = detector.face_net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    
                    # Expand face region slightly
                    padding = 20
                    x, y = max(0, x-padding), max(0, y-padding)
                    x2, y2 = min(w, x2+padding), min(h, y2+padding)
                    
                    face_img = frame[y:y2, x:x2]
                    if face_img.size == 0:
                        continue
                        
                    gender, gender_conf, age, age_conf = detector.predict_age_gender(face_img)
                    
                    label = f"{gender} ({gender_conf:.1f}) | {age} ({age_conf:.1f})"
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show FPS
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Age & Gender Detection", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"snapshot_{int(time.time())}.jpg", frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()