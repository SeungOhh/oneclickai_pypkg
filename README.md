![Main image](./public/main.png)

<br></br><br></br><br></br>

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

model = load_model("YOLO_coco")

# image path
image = cv2.imread('https://github.com/SeungOhh/oneclickai_pypkg/blob/main/public/test1.png')/255.0

# coco dataset cls names
coco_cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



result_annotation = predict(model, image, conf=0.4)
result_image = draw_result(np.array(image), result_annotation, class_names = coco_cls_names)
cv2.imshow('image', result_image)


# Close the window when 'Esc' is pressed
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

```

<br></br>

- **스트리밍**  

```

from oneclickai.YOLO import stream, load_model

# example usage
model = load_model("YOLO_coco")

# coco dataset cls names
coco_cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

stream(model, conf=0.5, class_names=coco_cls_names, video_source=0)

```

<br></br><br></br>

# 부가 기능
OneClickAI 패키지는 기본 제공되는 라이브러리 외에도 교육 목적에 맞는 다양한 부가 기능을 지속적으로 추가할 예정입니다.
최신 업데이트 및 추가 기능에 대한 정보는 [OneclickAI 공식 사이트](http://www.oneclickai.co.kr) 를 방문하여 확인하시기 바랍니다.

# 지원 및 문의
사용 중 문의사항이나 지원이 필요하신 경우, [원클릭 에이아이](http://www.oneclickai.co.kr) 문의 페이지를 통해 연락주시기 바랍니다.






