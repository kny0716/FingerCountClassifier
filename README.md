# FingerCountClassifier 

<br>


이 프로젝트는 사람의 손 모양을 인식하여 문화권마다 다르게 표현되는 숫자 제스처를 자동으로 판별하고 시각화하는 프로그램입니다.


**MediaPipe**로 손의 랜드마크를 추출하고, **KNN** 알고리즘을 통해 손 모양을 분류합니다.


웹캠을 통해 **한국식**, **중국식**, **일본식**, **인도식** 숫자 제스처를 실시간으로 인식할 수 있습니다.

<br>

---


## :earth_asia: 나라별 손가락 숫자 세기 방식


#### - 중국: 1부터 5까지는 한국, 일본과 동일. 6~10은 특수한 손 모양 사용. (10은 3가지 방법으로 표현함)


![중국 손가락 숫자](https://github.com/user-attachments/assets/7301843b-dbc6-4d96-b9e1-aa327b9a841f)
![china_10](https://github.com/user-attachments/assets/8c1324e7-5676-4028-8eff-a5d50c0b3b5f)



*이미지 출처: [nate 뉴스 - 중국에선 한 손으로 ‘10’까지 표현한다?](https://news.nate.com/view/20110718n23765)*  


<br>


#### - 일본: 1부터 5까지는 한국, 중국과 동일. 6~10은 양손을 조합하여 표현.

  
![일본 손가락 숫자](https://github.com/user-attachments/assets/cf449d0e-cec3-47c7-8efa-13a33377eac7)


*이미지 출처: [시사일본어사 공식 블로그 - 사소하지만 궁금했던 일본 이야기](https://blog.naver.com/japansisa/222134378192)*  


<br>


#### - 인도: 손가락의 마디를 짚으며 숫자를 셈.

  
![인도 손가락 숫자](https://github.com/user-attachments/assets/5d0b6a5e-e1bf-47ba-9272-54308635f0f4)


*이미지 출처: [동아일보 - 수학이 발달한 인도에선 손가락 마디로 20까지 세요](https://www.donga.com/news/Society/article/all/20200721/102097268/1)*  


<br>


---

## :computer: 데모 예시



https://github.com/user-attachments/assets/05a3094b-ce35-4925-9429-7a8e40158462




<br>

---


## :file_folder: 프로젝트 구조

```bash
FingerCountClassifier/
├── image/                   # 국기 이미지 등 UI 요소
│   ├── korea.png
│   ├── china.png
│   ├── japan.png
│   └── india.png
├── gesture_dataset.csv       # 수집된 손 모양 학습 데이터
├── knn_model_1hand.pkl       # 학습된 한 손 KNN 모델 파일
├── knn_model_2hand.pkl       # 학습된 두 손 KNN 모델 파일
├── knn_model_india.pkl       # 학습된 인도 KNN 모델 파일
├── data_collection.py        # 손 모양 데이터 수집용 스크립트
├── train_knn.py              # KNN 모델 학습용 스크립트
├── main.py                   # 실시간 인식 실행 파일
├── utils.py                  # 손 각도 계산, 손 랜드마크 추출 함수
├── requirements.txt          # 필요한 라이브러리 목록
└── README.md                 # 프로젝트 설명 파일
```

<br>

---

## :ballot_box_with_check: 주요 기능

* MediaPipe로 손 제스처 데이터 수집
* 손가락 관절 각도 기반 특징 벡터 생성
* 수집된 데이터로 KNN 모델 학습
* 문화별 숫자 표현 방식에 따라 실시간 분류
* 웹캠으로 손 모양 인식 결과 시각화

<br>

---

## :grey_exclamation: 실행 방법

#### 1. 의존 라이브러리 설치

```bash
pip install -r requirements.txt
```

#### 2. 학습 데이터 수집

```bash
python data_collection.py
# 예: korea_4, china_3, japan_2, india_1 등의 라벨을 입력하여 저장
```

#### 3. 모델 학습

```bash
python train_knn.py
# knn_model_1hand.pkl, knn_model_2hand.pkl, knn_model_india.pkl 생성
```

#### 4. 실시간 인식 실행

```bash
python main.py
# 생성된 한 손, 두 손 모델로 culture와 number 예측
# 예측 결과, culture가 인도라면 인도 모델로 한 번 더 예측 (인도는 다른 나라에 비해 모양 구분이 힘들기 때문에 더 많은 특징 추출)
```

* `ESC` 키 → 종료

---


<br>

## :warning: 발생한 문제점 및 개선 방향
- MediaPipe로 손 제스쳐 데이터를 수집하여, 손가락 관절 각도 기반으로 구분하는 방식으로 구현되어 각도가 비슷한 인도 숫자 손 제스쳐가 올바르게 구분되지 않은 문제점이 있었음
  <br>
   -> 인도 제스처의 경우 손가락 관절 각도뿐만 아니라 상대 좌표 및 거리 데이터를 추가적으로 수집하여 별도 모델로 학습시켜 정확도를 높임
  <br>
   -> 기존 모델보다 비교적 정확하지만, 오인식 경우가 잦음

  
- KNN 모델의 특성상, 예외적인 손 모양에 민감하게 반응하여 오인식 가능성이 있음
  <br>
   -> 거리 기반 분류가 아닌 딥러닝 기반 분류 모델 사용 고려

  
- 손의 회전 각도에 따라 인식률이 달라지는 한계가 있음
  <br>
   -> 다양한 각도로 샘플을 보강했으나 손가락이 겹치거나 일부 손가락이 가려진 경우에는 MediaPipe가 landmark를 정확히 추출하지 못하는 경우가 발생함
  <br>
   -> 이로 인해 잘못된 landmark 정보가 분류기로 전달되어 예측이 부정확해질 수 있음
  <br>
   -> 이러한 경우를 줄이기 위해 landmark 정확도를 판단하거나, 보정 알고리즘을 추가하는 개선이 필요함


