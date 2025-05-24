import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
from utils import extract_features


# 모델 로드
model_1hand = joblib.load("knn_model_1hand.pkl")
model_2hand = joblib.load("knn_model_2hand.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

culture_list = ['korea', 'china', 'japan', 'india']

label_equivalence = {
    'korea_1' : ['china_1', 'japan_1'],
    'china_1' : ['korea_1', 'japan_1'],
    'japan_1' : ['korea_1', 'china_1'],
    'korea_2' : ['china_2', 'japan_2'],
    'china_2' : ['korea_2', 'japan_2'],
    'japan_2' : ['korea_2', 'china_2'],
    'korea_3' : ['china_3', 'japan_3'],
    'china_3' : ['korea_3', 'japan_3'],
    'japan_3' : ['korea_3', 'china_3'],
    'korea_4' : ['china_4', 'japan_4','india_1'],
    'china_4' : ['korea_4', 'japan_4', 'india_1'],
    'japan_4' : ['korea_4', 'china_4', 'india_1'],
    'india_1' : ['korea_4', 'china_4', 'japan_4'],
    'korea_5' : ['china_5', 'japan_5'],
    'china_5' : ['korea_5', 'japan_5'],
    'japan_5' : ['korea_5', 'china_5'],
    'korea_10' : ['japan_10'],
    'japan_10' : ['korea_10'],
}


# 예측 결과 해석 함수
def predict_label(model, feature_vector):
    try:
        pred = model.predict([feature_vector])[0]
        culture, number = pred.split("_")
        return culture, number
    except:
        return None, None

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

        hand_count = 0
        feature_vector = []

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extract_features(hand_landmarks.landmark)
                feature_vector.extend(features)

        prediction_result =  {c: '-' for c in culture_list}
        
        if feature_vector:
            if hand_count == 1 and len(feature_vector) == 15:
                culture, number = predict_label(model_1hand, feature_vector)
            elif hand_count == 2 and len(feature_vector) == 30:
                culture, number = predict_label(model_2hand, feature_vector)
            else:
                culture, number = None, None

            if culture in prediction_result:
                prediction_result[culture] = number

                label_key = f"{culture}_{number}"

                # 해당 제스처와 의미상 동일한 라벨들
                equivalents = label_equivalence.get(label_key, [])
                for eq in equivalents:
                    eq_culture, eq_number = eq.split('_')
                    prediction_result[eq_culture] = eq_number

        # 결과 출력
        y = 30
        for c in culture_list:
            text = f"{c.capitalize()}: {prediction_result[c]}"
            cv.putText(frame, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        cv.putText(frame, f"Hands detected: {hand_count}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv.imshow("Cultural Finger Counter", frame)

        key = cv.waitKey(10) & 0xFF
        if key in [27, ord('q')]:
            break

cap.release()
cv.destroyAllWindows()