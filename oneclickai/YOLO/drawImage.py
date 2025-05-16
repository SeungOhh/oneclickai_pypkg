#%%
import numpy as np
import cv2
import os


# display yolo result: each image with waitkey
def draw_result_imshow(image, annotation, class_names=None):
    disp_image = draw_result(image, annotation, class_names)
    cv2.imshow('image', disp_image)
    cv2.waitKey(0)



# display yolo result
def draw_result(image, annotation, class_names=None):

    # draw boxes
    img_size_y = image.shape[0]
    img_size_x = image.shape[1]

    # draw image first
    disp_image = image.copy()*255
    disp_image = disp_image.astype(np.uint8)
    
    # print(disp_image)
    for box in annotation:
        class_id, x, y, w, h = box
        x = x * img_size_x
        y = y * img_size_y
        w = w * img_size_x
        h = h * img_size_y
        cv2.rectangle(disp_image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)


        # if coco dataset, display class name
        if class_names is not None:
            cls_name = class_names[int(class_id)]
        else: 
            cls_name = str(int(class_id))
            
        # display class name
        cv2.putText(disp_image, cls_name, (int(x-w/2+5), int(y+h/2-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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


