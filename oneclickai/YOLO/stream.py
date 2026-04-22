import cv2
import numpy as np
import time
from .drawImage import draw_result, _IN_COLAB
from .predict import predict

def stream(model, conf=0.5, iou=0.5, class_names=None, video_source=0):

    if _IN_COLAB:
        print("stream()은 Colab 환경에서 지원되지 않습니다.")
        print("이미지 단건 분석은 predict()와 draw_result()를 사용하세요.")
        return

    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        print("Error: Could not open video.")
        return

    prev_time = time.time()

    while True:
        _, frame = capture.read()

        annotations = predict(model, frame, conf=conf, iou=iou)
        disp_image = draw_result(np.array(frame), annotations, class_names)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        h, w = disp_image.shape[:2]
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(disp_image, fps_text, (w - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('frame', disp_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


# example usage
if __name__ == '__main__':
    from load_model import load_model
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

    stream(model, conf=0.5, class_names=coco_cls_names)