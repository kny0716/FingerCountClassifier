import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from collections import Counter

def load_dataset(filename='gesture_dataset.csv'):
    one_hand_X, one_hand_y = [], []
    two_hand_X, two_hand_y = [], []
    india_X, india_y = [], []

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            features = list(map(float, row[:-1]))
            label = row[-1]

            if len(features) == 15:
                one_hand_X.append(features)
                one_hand_y.append(label)
            elif len(features) == 30:
                if label.startswith("india_"):
                    india_X.append(features)
                    india_y.append(label)
                else:
                    two_hand_X.append(features)
                    two_hand_y.append(label)

    return (one_hand_X, one_hand_y), (two_hand_X, two_hand_y), (india_X, india_y)

def train_and_save_model(X, y, model_filename):
    if len(X) < 5:
        print(f"'{model_filename}' 학습 데이터가 너무 적습니다. ({len(X)}개)")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_filename} 정확도: {acc * 100:.2f}%")

    print(f"라벨별 샘플 수:")
    for label, count in Counter(y).items():
        print(f"    {label}: {count}개")

    joblib.dump(model, model_filename)
    print(f"모델 저장됨: {model_filename}\n")

if __name__ == "__main__":
    (one_X, one_y), (two_X, two_y), (india_X, india_y) = load_dataset()

    train_and_save_model(one_X, one_y, "knn_model_1hand.pkl")
    train_and_save_model(two_X, two_y, "knn_model_2hand.pkl")
    train_and_save_model(india_X, india_y, "knn_model_india.pkl")