import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from collections import Counter


# CSV 로드 함수
def load_dataset(filename='gesture_dataset.csv'):
    X, y = [], []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            *features, label = row
            if len(features) not in [15, 30]:
                continue  # 손 1개 or 2개만 허용
            X.append([float(x) for x in features])
            y.append(label)
    return np.array(X), np.array(y)

# 특징 벡터에서 손 개수 분리
def split_by_hand_count(X, y):
    X1, y1, X2, y2 = [], [], [], []
    for xi, yi in zip(X, y):
        if len(xi) == 15:
            X1.append(xi)
            y1.append(yi)
        elif len(xi) == 30:
            X2.append(xi)
            y2.append(yi)
    return np.array(X1), np.array(y1), np.array(X2), np.array(y2)

# 데이터 로딩
X, y = load_dataset()
X1, y1, X2, y2 = split_by_hand_count(X, y)

print(f"[INFO] 한 손 데이터: {len(X1)}개")
print(f"[INFO] 두 손 데이터: {len(X2)}개")

# 원하는 모델만 학습 (지금은 전체 통합)
X_all = np.concatenate([X1, X2], axis=0)
y_all = np.concatenate([y1, y2], axis=0)

# 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# 모델 학습
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 정확도 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[✔] 모델 정확도: {acc * 100:.2f}%")

# 라벨별 샘플 개수 출력
label_count = Counter(y_all)
for label, count in label_count.items():
    print(f"{label}: {count}개")

# 모델 저장
joblib.dump(model, 'knn_model.pkl')
print("[✔] 모델이 'knn_model.pkl'로 저장되었습니다.")
