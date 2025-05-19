![Main image](./public/main.png)

<br></br><br></br>

<div style="text-align: center;">
    <img src="./public/yolo1.webp" alt="Alt Text" width="1200">
</div>

<br></br><br></br>

# OneClickAI Python 패키지
YOLO 모델을 쉽게 학습해보고 바로 실행해 볼 수 있도록 하는 패키지 입니다. 
이론 교육에 앞서 미리 모델을 체험해보고 YOLO 모델의 구조 및 실행 방식에 대해 알아 볼 수 있습니다. 모델은 Tensorflow 기반의 모델로 작성되었습니다.

OneClickAI에서 제공하는 교육용 Python 패키지는 인공지능(AI) 학습을 위한 필수 도구들을 손쉽게 설치하고 활용할 수 있도록 도와줍니다. 
이 패키지를 통해 TensorFlow, Ultralytics, OpenCV와 같은 필수 라이브러리를 한 번에 설치하고, 추가적인 부가 기능도 손쉽게 통합할 수 있습니다.

<br></br><br></br>

## 주요 설치 패키지
- **oneclickai**  
  자체적으로 모델을 쉽게 학습해보고 테스트 해 볼 수 있는 기능 제공 (설명서는 oneclickai.co.kr 참고)

- **TensorFlow**  
  구글에서 개발한 오픈소스 딥러닝 라이브러리로, 다양한 머신러닝 모델을 구축하고 훈련할 수 있습니다.

- **Ultralytics**  
  최신 YOLOv8 모델을 제공하는 라이브러리로, 객체 탐지 및 컴퓨터 비전 작업에 활용됩니다.

- **OpenCV**  
  이미지 및 비디오 처리에 널리 사용되는 라이브러리로, 실시간 컴퓨터 비전 애플리케이션 개발에 필수적입니다.

<br></br><br></br>

## 설치 방법

아래의 명령어를 통해 OneClickAI 패키지를 설치할 수 있습니다:

```bash
pip install oneclickai
```

<br></br><br></br>

# YOLO 모델 예제 코드

- **이미지 1장**  

```
from oneclickai.YOLO import load_model, predict, draw_result
import cv2
import numpy as np


# model path: 여기에 모델 위치를 넣어주세요. 상대위치 or 절대위치
# model path: "YOLO_coco" coco data로 학습한 기본모델 활용
model = load_model("YOLO_coco") 

# image path: 여기에 이미지 파일 위치를 넣어주세요.
image = cv2.imread('/path/to/imagefile')/255.0

# class_names: 리스트 변수로 모델이 학습한 클래스의 이름을 넣어주세요 (한글 X)
coco_cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# 결과 확인: 입력으로 모델, 이미지 변수, confidence
# 모델 출력이 confidence 값 이상인 경우에만 출력 됨
result_annotation = predict(model, image, conf=0.4)

# 결과 이미지 그려주기
result_image = draw_result(np.array(image), result_annotation, class_names = coco_cls_names)
cv2.imshow('image', result_image)

# ESC 누르면 창 닫기
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

```

<br></br>

- **스트리밍**  

```

from oneclickai.YOLO import stream, load_model

# model path: 여기에 모델 위치를 넣어주세요. 상대위치 or 절대위치
# model path: "YOLO_coco" coco data로 학습한 기본모델 활용
model = load_model("YOLO_coco")


# class_names: 리스트 변수로 모델이 학습한 클래스의 이름을 넣어주세요 (한글 X)
coco_cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# 결과 확인: 모델, confidence, 클래스 리스트, 카메라 번호(첫번째 카메라:0, 두번째 카메라:1, ...)
stream(model, conf=0.5, class_names=coco_cls_names, video_source=0)

```

<br></br>

- **모델학습**  

```

from oneclickai.YOLO import fit_yolo_model

# 학습용 데이터 위치, 이미지 데이터(.png, .jpg, .jpeg), 라벨 데이터(.txt)
train_data_path = './yolo_dataset'
train_label_path = './yolo_dataset'

# 검증용 데이터 위치, 이미지 데이터(.png, .jpg, .jpeg), 라벨 데이터(.txt)
val_data_path = './yolo_dataset'
val_label_path = './yolo_dataset'

# 모델 학습 (데이터 위치, epochs)
fit_yolo_model(train_data_path, train_label_path, val_data_path, val_label_path, epochs=30)

```

<br></br><br></br>

# 부가 기능
OneClickAI 패키지는 기본 제공되는 라이브러리 외에도 교육 목적에 맞는 다양한 부가 기능을 지속적으로 추가할 예정입니다.
최신 업데이트 및 추가 기능에 대한 정보는 [OneclickAI 공식 사이트](http://www.oneclickai.co.kr) 를 방문하여 확인하시기 바랍니다.

# 지원 및 문의
사용 중 문의사항이나 지원이 필요하신 경우, [원클릭 에이아이](http://www.oneclickai.co.kr) 문의 페이지를 통해 연락주시기 바랍니다.


<br></br><br></br>



![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

