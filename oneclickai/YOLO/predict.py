import cv2
import numpy as np
import os

from . import labelDecode
from .drawImage import draw_result


def predict(model, frame, conf=0.5):
    img_size = 320
    high_stride = model.output_shape[0][1]
    low_stride = model.output_shape[1][1]
    
    frame = cv2.resize(frame, (img_size, img_size))

    result = model.predict(frame[np.newaxis, ...])
    annotations = labelDecode.decode(result[0][0], result[1][0], high_stride, low_stride, conf)
    return annotations



def predict_and_show(model, frame, conf=0.5, cls_names=None):
    img_size = 320
    high_stride = model.output_shape[0][1]
    low_stride = model.output_shape[1][1]
    
    frame = cv2.resize(frame, (img_size, img_size))

    result = model.predict(frame[np.newaxis, ...])
    annotations = labelDecode.decode(result[0][0], result[1][0], high_stride, low_stride, conf)
    result_image = draw_result(np.array(image), annotations, cls_names)
    cv2.imshow('image', result_image)
    cv2.waitKey(0)
    return annotations








# test code
if __name__ == '__main__':
    from load_model import load_model

    data_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/images/train2017/'
    label_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/labels_bbox/train2017/'

    file_img = os.listdir(data_path)
    file_txt = os.listdir(label_path)

    model = load_model("YOLO")
    

    # # method1: load model and predict
    # for i in range(5):
    #     image = cv2.imread(data_path + file_img[i])/255.0
    #     result_annotation = predict(model, image)
    #     result_image = draw_result(np.array(image), result_annotation)
    #     cv2.imshow('image', result_image)
    #     cv2.waitKey(0)


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
    

    # method2: load model and predict
    for i in range(5):
        image = cv2.imread(data_path + file_img[i])/255.0
        result = predict_and_show(model, image, cls_names=coco_cls_names)
