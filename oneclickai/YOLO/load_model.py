import tensorflow as tf
import gdown
import os
from .yoloLoss import yolo_loss_tf


COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# YOLO_coco 모델 저장 경로 — 한글 경로에서 TF가 파일을 읽지 못하므로 홈 디렉터리에 저장
_COCO_MODEL_PATH = os.path.join(os.path.expanduser('~'), 'yolo_tf.h5')


def load_model(model_path=None):

    if model_path is None:
        raise ValueError("model_path is None, please provide a valid path or try \"YOLO_coco\"")

    elif model_path == "YOLO_coco":
        download_model_from_gdrive(
            "https://drive.google.com/file/d/1HuiUq4q1mJdlX9PKmGnL505YPTHE8DLZ/view?usp=sharing",
            destination=_COCO_MODEL_PATH
        )
        model_path = _COCO_MODEL_PATH

    if model_path.endswith('.tflite'):
        model = tf.lite.Interpreter(model_path)
        model.allocate_tensors()
    else:
        model = tf.keras.models.load_model(model_path, custom_objects={"yolo_loss_tf": yolo_loss_tf})

    return model


def download_model_from_gdrive(url, destination):
    dest = os.path.abspath(destination)

    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"[gdown] Found existing file → {dest}  (skipping download)")
        return dest

    print("[gdown] Downloading model …")
    gdown.download(url, dest, quiet=False, fuzzy=True)
    print(f"[gdown] Model saved to {dest}")
    return dest
