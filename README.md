<div style="text-align: center;">
    <h3>사용 예제 1</h3>
    <p>야구공 찾기</p>
    <p>이미지 취득 → 라벨링 → 학습 → 결과 확인</p>
    <img src="https://images.oneclickai.work/movie/introduce/yolo3.webp" alt="Alt Text" width="1200">
</div>

<br></br><br></br>

<div style="text-align: center;">
    <h3>사용 예제 2</h3>
    <p>내 차 어디갔어</p>
    <p>이미지 취득 → 라벨링 → 학습 → 결과 확인</p>
    <img src="https://images.oneclickai.work/movie/introduce/yolo1.webp" alt="Alt Text" width="1200">
</div>

<br></br><br></br>

# OneClickAI Python 패키지
YOLO 모델을 쉽게 학습해보고 바로 실행해 볼 수 있도록 하는 패키지 입니다. 
이론 교육에 앞서 미리 모델을 체험해보고 YOLO 모델의 구조 및 실행 방식에 대해 알아 볼 수 있습니다. 모델은 Tensorflow 기반의 모델로 작성되었습니다.

OneClickAI에서 제공하는 교육용 Python 패키지는 인공지능(AI) 학습을 위한 필수 도구들을 손쉽게 설치하고 활용할 수 있도록 도와줍니다. 
이 패키지를 통해 TensorFlow, OpenCV와 같은 필수 라이브러리를 한 번에 설치하고, 추가적인 부가 기능도 손쉽게 통합할 수 있습니다.

<br></br><br></br>

## 설치 방법

아래의 명령어를 통해 OneClickAI 패키지를 설치할 수 있습니다:

```bash
pip install oneclickai
```

<br></br><br></br>

## 주요 설치 패키지
- oneclickai  
  자체적으로 모델을 쉽게 학습해보고 테스트 해 볼 수 있는 기능 제공 (설명서는 oneclickai.co.kr 참고)

- TensorFlow  
  구글에서 개발한 오픈소스 딥러닝 라이브러리로, 다양한 머신러닝 모델을 구축하고 훈련할 수 있습니다.

- OpenCV  
  이미지 및 비디오 처리에 널리 사용되는 라이브러리로, 실시간 컴퓨터 비전 애플리케이션 개발에 필수적입니다.

- Matplotlib  
  데이터 시각화 라이브러리로, 모델 학습 결과 그래프를 자동으로 그려줍니다.


<br></br><br></br>

# YOLO 모델 예제 코드

- 이미지 1장  

```python
from oneclickai.YOLO import load_model, predict, draw_result, COCO_CLASS_NAMES
import cv2
import numpy as np


# model path: 여기에 모델 위치를 넣어주세요. 상대위치 or 절대위치
# model path: "YOLO_coco" coco data로 학습한 기본모델 활용
model = load_model("YOLO_coco") 

# image path: 여기에 이미지 파일 위치를 넣어주세요.
image = cv2.imread('/path/to/imagefile')

# 결과 확인: 입력으로 모델, 이미지 변수, confidence, iou
# conf: 이 값 이상인 경우에만 박스가 출력됩니다 (0~1 사이)
# iou: 겹치는 박스를 제거하는 기준값입니다 (0~1 사이, 낮을수록 더 많이 제거)
result_annotation = predict(model, image, conf=0.5, iou=0.5)

# 결과 이미지 그려주기
# COCO_CLASS_NAMES: COCO 데이터셋의 80개 클래스 이름이 저장된 리스트입니다
result_image = draw_result(np.array(image), result_annotation, class_names=COCO_CLASS_NAMES)
cv2.imshow('image', result_image)

# ESC 누르면 창 닫기
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

```

<br></br>

- 웹캠 스트리밍  

```python

from oneclickai.YOLO import stream, load_model, COCO_CLASS_NAMES

# model path: 여기에 모델 위치를 넣어주세요. 상대위치 or 절대위치
# model path: "YOLO_coco" coco data로 학습한 기본모델 활용
model = load_model("YOLO_coco")

# 결과 확인: 모델, confidence, iou, 클래스 리스트, 카메라 번호(첫번째 카메라:0, 두번째 카메라:1, ...)
# 화면 우측 상단에 FPS가 실시간으로 표시됩니다
# 종료하려면 'q' 키를 누르세요
stream(model, conf=0.5, iou=0.5, class_names=COCO_CLASS_NAMES, video_source=0)

```

<br></br>

- 모델학습  

```python

from oneclickai.YOLO import fit_yolo_model

# 이미지(.png, .jpg)와 라벨(.txt)이 같은 폴더에 있으면 자동으로 학습/검증(8:2) 분리
data_path  = './yolo_dataset'
label_path = './yolo_dataset'

# 모델 학습
# epochs: 전체 학습 반복 횟수
# batch_size: 한 번에 처리할 이미지 수 (GPU 메모리에 맞게 조절)
# save_tflite: True로 설정하면 학습 완료 후 .tflite 파일도 함께 저장됩니다
fit_yolo_model(data_path, label_path, epochs=30, batch_size=8, save_tflite=True)

```

학습/검증 데이터를 직접 지정하고 싶다면 아래처럼 사용할 수 있습니다:

```python

fit_yolo_model(
    train_data_path  = './train',   # 학습 이미지 폴더
    train_label_path = './train',   # 학습 라벨 폴더 (이미지와 같은 폴더)
    val_data_path    = './val',     # 검증 이미지 폴더
    val_label_path   = './val',     # 검증 라벨 폴더 (이미지와 같은 폴더)
    epochs=30, batch_size=8, save_tflite=True
)

```

학습이 완료되면 날짜/시간 이름의 폴더에 아래 파일들이 자동으로 저장됩니다:
- `yolo_model_best.h5` — 검증 손실 기준 가장 좋은 모델
- `yolo_model_last.h5` — 마지막 epoch 모델
- `yolo_model_best.tflite` / `yolo_model_last.tflite` — TFLite 변환 모델 (`save_tflite=True` 시)
- `training_history.png` — 학습/검증 손실 그래프

<br></br><br></br>

# 직접 체험해보기

아래 Google Colab 링크에서 설치 없이 바로 YOLO 모델을 체험해볼 수 있습니다.

[Google Colab에서 직접 해보기 →](https://drive.google.com/file/d/1mAzizV2mxIRMlAZJdFvuaC6r_QnPsRin/view?usp=sharing)

<br></br><br></br>

# 부가 기능
OneClickAI 패키지는 기본 제공되는 라이브러리 외에도 교육 목적에 맞는 다양한 부가 기능을 지속적으로 추가할 예정입니다.
최신 업데이트 및 추가 기능에 대한 정보는 [OneclickAI 공식 사이트](http://www.oneclickai.co.kr) 를 방문하여 확인하시기 바랍니다.

# 지원 및 문의
사용 중 문의사항이나 지원이 필요하신 경우, [원클릭 에이아이](http://www.oneclickai.co.kr) 문의 페이지를 통해 연락주시기 바랍니다.


<br></br><br></br>



![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)
