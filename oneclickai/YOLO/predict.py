import cv2
import numpy as np
import os
import tensorflow as tf

try:
    from google.colab.patches import cv2_imshow as _cv2_imshow
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

from . import labelDecode
from .drawImage import draw_result


def predict(model, frame, conf=0.5, iou=0.5):
    frame = frame/255.0
    if isinstance(model, tf.keras.Model):
        return predictTF(model, frame, conf=conf, iou=iou)
    elif isinstance(model, tf.lite.Interpreter):
        return predictTFLite(model, frame, conf=conf, iou=iou)
    else:
        print("something is wrong with the model")
    
    
    







def predictTF(model, frame, conf=0.5, iou=0.5):
    img_size = 320
    frame = cv2.resize(frame, (img_size, img_size))

    high_stride = model.output_shape[0][1]
    low_stride = model.output_shape[1][1]

    result = model.predict(frame[np.newaxis, ...])
#     print(np.shape(result[0][0]), np.shape(result[1][0]))
    annotations = labelDecode.decode(result[0][0], result[1][0], high_stride, low_stride, conf, iou_threshold=iou)
    return annotations



def predictTFLite(model, frame, conf=0.5, iou=0.5):
    img_size = 320
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype(np.float32)
    frame = np.expand_dims(frame, axis=0)

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Get shape of each output
    high_stride_shape = output_details[0]['shape']  # e.g., [1, 52, 52, 255]
    low_stride_shape  = output_details[1]['shape']  # e.g., [1, 13, 13, 255]

    # Access second dimension (e.g., height)
    high_stride = high_stride_shape[1]
    low_stride  = low_stride_shape[1]

#     print(high_stride_shape, low_stride_shape, high_stride, low_stride)

    model.set_tensor(input_details[0]['index'], frame)
    model.invoke()
    result1 = model.get_tensor(output_details[0]['index'])
    result2 = model.get_tensor(output_details[1]['index'])

    annotations = labelDecode.decode(result1[0], result2[0], high_stride, low_stride, conf, iou_threshold=iou)
    return annotations

    
    
    
    



def predict_and_show(model, frame, conf=0.5, iou=0.5, class_names=None):
    annotations = predict(model, frame, conf=conf, iou=iou)
    result_image = draw_result(np.array(frame), annotations, class_names)
    if _IN_COLAB:
        _cv2_imshow(result_image)
    else:
        cv2.imshow('image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
        result = predict_and_show(model, image, class_names=coco_cls_names)

