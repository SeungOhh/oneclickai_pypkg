import cv2
import numpy as np
from .drawImage import draw_result
from .predict import predict

def stream(model, conf=0.5, cls_names=None):

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open video.")
        return



    while True:
        _, frame = capture.read()
        frame = frame/255.0

        annotations = predict(model, frame, conf=conf)
        disp_image = draw_result(np.array(frame), annotations, cls_names)

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

    stream(model, conf=0.5, cls_names=coco_cls_names)