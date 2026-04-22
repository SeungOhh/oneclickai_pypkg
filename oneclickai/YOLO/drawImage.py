#%%
import numpy as np
import cv2
import os

try:
    from google.colab.patches import cv2_imshow as _cv2_imshow
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False


# display yolo result: each image with waitkey
def draw_result_imshow(image, annotation, class_names=None):
    disp_image = draw_result(image, annotation, class_names)
    if _IN_COLAB:
        _cv2_imshow(disp_image)
    else:
        cv2.imshow('image', disp_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def _class_color(class_id):
    hue = int(class_id * 37) % 180
    color = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# display yolo result
def draw_result(image, annotation, class_names=None):

    img_size_y = image.shape[0]
    img_size_x = image.shape[1]

    disp_image = np.array(image, dtype=np.uint8)

    for box in annotation:
        class_id, x, y, w, h, conf = box
        x = x * img_size_x
        y = y * img_size_y
        w = w * img_size_x
        h = h * img_size_y

        if (class_names != None) and (class_id >= len(class_names)):
            continue

        if class_names is not None:
            cls_name = class_names[int(class_id)]
        else:
            cls_name = str(int(class_id))

        box_color = _class_color(class_id)
        text_color = box_color

        label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(disp_image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), box_color, 2)
        cv2.putText(disp_image, label, (int(x-w/2+5), int(y+h/2-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return disp_image





if __name__ == '__main__':
    data_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/images/train2017/'
    label_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/labels_bbox/train2017/'

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

    file_img = os.listdir(data_path)
    file_txt = os.listdir(label_path)

    for i in range(5):
        image = cv2.imread(data_path + file_img[i])/255.0
        image = cv2.resize(image, (416, 416))
        annotation = np.loadtxt(label_path + file_txt[i])

        if len(annotation.shape) == 1:
            annotation = annotation[np.newaxis, :]

        # print(file_img[i], annotation)
        print(image)
        # display result
        result = draw_result(image, annotation, coco_cls_names)
        cv2.imshow('image', result)
        cv2.waitKey(1000)


