import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
from utils import extract_features

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

flag_imgs = {
    'korea': cv.imread('image/korea.png', cv.IMREAD_UNCHANGED),
    'china': cv.imread('image/china.png', cv.IMREAD_UNCHANGED),
    'japan': cv.imread('image/japan.png', cv.IMREAD_UNCHANGED),
    'india': cv.imread('image/india.png', cv.IMREAD_UNCHANGED),
}

def predict_label(model, feature_vector):
    try:
        pred = model.predict([feature_vector])[0]
        culture, number = pred.split("_")
        return culture, number
    except:
        return None, None
    
def predict_label_with_threshold(model, features, threshold=50.0):
    dist, indices = model.kneighbors([features], n_neighbors=1)
    if dist[0][0] > threshold:
        return None, None  # 너무 멀면 인식 불가
    return model.predict([features])[0].split('_')
    
def overlay_image_alpha(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x + w > bw or y + h > bh:
        return  

    if overlay.shape[2] == 4:
        b, g, r, a = cv.split(overlay)
        overlay_rgb = cv.merge((b, g, r))
        mask = cv.merge((a, a, a)) / 255.0
    else:
        overlay_rgb = overlay
        mask = np.ones_like(overlay, dtype=np.float32)

    roi = background[y:y+h, x:x+w].astype(float)
    blended = roi * (1 - mask) + overlay_rgb.astype(float) * mask
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

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
                # culture, number = predict_label(model_1hand, feature_vector)
                culture, number = predict_label_with_threshold(model_1hand, feature_vector, threshold=50)
            elif hand_count == 2 and len(feature_vector) == 30:
                # culture, number = predict_label(model_2hand, feature_vector)
                culture, number = predict_label_with_threshold(model_2hand, feature_vector, threshold=70)
            else:
                culture, number = None, None

            if culture in prediction_result:
                prediction_result[culture] = number

                label_key = f"{culture}_{number}"

                equivalents = label_equivalence.get(label_key, [])
                for eq in equivalents:
                    eq_culture, eq_number = eq.split('_')
                    prediction_result[eq_culture] = eq_number

        y = 30
        for idx, culture in enumerate(culture_list):

            flag = cv.resize(flag_imgs[culture], (40, 40))
            overlay_image_alpha(frame, flag, x=10, y=y - 30)    

            text = f"{culture.capitalize()}: {prediction_result[culture]}"
            cv.putText(frame, text, (60, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            y += 40

        
        cv.imshow("FingerCountClassifier", frame)

        key = cv.waitKey(10) & 0xFF
        if key in [27, ord('q')]:
            break

cap.release()
cv.destroyAllWindows()