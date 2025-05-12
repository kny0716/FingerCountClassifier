# FingerCountClassifier (문화별 손가락 숫자 인식기)

이 프로젝트는 **문화마다 다른 손가락 숫자 세기 방식**을 인식하기 위한 머신러닝 기반 손 제스처 분류기입니다. **MediaPipe**로 손의 랜드마크를 추출하고, **K-최근접 이웃 (KNN)** 알고리즘을 통해 손 모양을 분류합니다.

웹캠을 통해 **한국식**, **중국식**, **일본식** 숫자 제스처를 실시간으로 인식할 수 있습니다.

---

## 📸 데모 예시



---

## 📂 프로젝트 구조

```bash
CulturalFingerCount/
├── data/                      # (선택) 예시 이미지 저장용
├── assets/                   # 국기 이미지 등 UI 요소
│   ├── korea.png
│   ├── china.png
│   └── japan.png
├── gesture_dataset.csv       # 수집된 제스처 학습 데이터
├── knn_model.pkl             # 학습된 KNN 모델 파일
├── data_collection.py        # 제스처 데이터 수집용 스크립트
├── train_knn.py              # KNN 모델 학습용 스크립트
├── main.py                   # 실시간 인식 실행 파일
├── utils.py                  # 각도 계산 등 보조 함수
├── requirements.txt          # 필요한 라이브러리 목록
└── README.md                 # 프로젝트 설명 파일
```

---

## 🧠 주요 기능

* MediaPipe로 손 제스처 데이터 수집
* 손가락 관절 각도 기반 특징 벡터 생성
* 수집된 데이터로 KNN 모델 학습
* 문화별 숫자 표현 방식에 따라 실시간 분류
* 웹캠으로 손 모양 인식 결과 시각화

---

## 💻 실행 방법

### 1. 의존 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 학습 데이터 수집

```bash
python data_collection.py
# 예: korea_3, china_2, japan_1 등의 라벨을 입력하여 저장
```

### 3. 모델 학습

```bash
python train_knn.py
```

### 4. 실시간 인식 실행

```bash
python main.py
```

* `k` 키 → 한국식 인식 모드
* `c` 키 → 중국식 인식 모드
* `j` 키 → 일본식 인식 모드
* `ESC` 키 → 종료

---

## 🌏 문화별 숫자 표현 예시

| 문화권 | 숫자 3 표현 예시                   |
| --- | ---------------------------- |
| 한국  | 엄지 + 검지 + 중지                 |
| 중국  | 손바닥을 옆으로 펼쳐 3 지시 (사용자 정의 가능) |
| 일본  | 검지 + 중지 + 약지                 |

추가적인 규칙은 사용자가 자유롭게 정의할 수 있습니다!

---

## 📦 사용 기술 스택

* Python 3.x
* MediaPipe Hands
* OpenCV
* NumPy
* scikit-learn (KNN 알고리즘)

---

## 📜 라이선스

MIT License — 자유롭게 사용, 수정, 기여가 가능합니다.
