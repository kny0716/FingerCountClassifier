import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
from utils import extract_features

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 모델 로드
model = joblib.load("knn_model.pkl")

# 실시간 인식
cap = cv.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        results_dict = {'korea': '-', 'china': '-', 'japan': '-', "india": '-'}
        feature_vector = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extract_features(hand_landmarks.landmark)
                feature_vector.extend(features)

            if feature_vector:
                try:
                    pred = model.predict([feature_vector])[0]  # 예: 'china_6'
                    culture_name, number = pred.split('_')
                    if culture_name in results_dict:
                        results_dict[culture_name] = number
                except:
                    pass

        # 결과 출력
        y_offset = 40
        for culture_name in ['korea', 'china', 'japan', 'india']:
            display_text = f'{culture_name.capitalize()}: {results_dict[culture_name]}'
            cv.putText(frame, display_text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        cv.putText(frame, f"Hands detected: {hand_count}", (10, y_offset + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv.imshow("Cultural Finger Counter (KNN) - Multi", frame)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

cap.release()
cv.destroyAllWindows()