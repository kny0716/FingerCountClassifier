import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import os
from utils import extract_features ,extract_features_india

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def save_to_csv(features, label, filename='gesture_dataset.csv'):
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(features + [label])

def select_label():
    options = [
    *[f'korea_{i}' for i in range(1, 11)],
    *[f'china_{i}' for i in range(1, 11)],
    *[f'japan_{i}' for i in range(1, 11)],
    *[f'india_{i}' for i in range(1, 11)],
]
    print("\n[라벨 선택] 아래에서 숫자를 입력해 주세요:")
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    while True:
        try:
            choice = int(input("번호 선택: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except:
            pass
        print("유효하지 않은 입력입니다. 다시 입력해 주세요.")

def count_existing_samples(label, filename='gesture_dataset.csv'):
    if not os.path.exists(filename):
        return 0
    count = 0
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[-1] == label:
                count += 1
    return count

cap = cv.VideoCapture(0)
label = select_label()
sample_count = count_existing_samples(label)

print(f"스페이스바를 누르면 '{label}' 샘플이 저장됩니다. (ESC로 종료)")
print(f"현재까지 저장된 {label} 샘플 수: {sample_count}")

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        hand_counst = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        cv.putText(frame, f'Hands: {hand_counst}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv.putText(frame, f'Label: {label} | Count: {sample_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.imshow("Data Collection", frame)

        key = cv.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32:  # SPACE
            if results.multi_hand_landmarks:
                all_features = []
                for hand in results.multi_hand_landmarks:
                    if label.startswith('india'):
                        features = extract_features_india(hand.landmark)
                    else:
                        features = extract_features(hand.landmark)
                    all_features.extend(features)
                save_to_csv(all_features, label)
                sample_count += 1
                print(f"샘플 저장됨: {label} ({sample_count})")
            else:
                print("손이 인식되지 않았습니다.")

cap.release()
cv.destroyAllWindows()
