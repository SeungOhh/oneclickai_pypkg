#%%
import numpy as np
import os
import cv2
import tensorflow as tf

from .readTxt import load_data_from_txt
from .labelEncode import encode
from .createDataAugumentation import random_horizontal_flip, random_color_jitter, random_crop, letterbox_and_pad


def data_generator(num_classes, high_stride, low_stride, img_size,
                   data_path, label_path, batch_size=32, shuffle=True, prob=0.5):

    # List image files (without extension)
    image_files = [f[:-4] for f in os.listdir(data_path) if f.endswith('.jpg')]
    num_images = len(image_files)
    indices = np.arange(num_images)

    # Preallocate label shapes
    high_label_shape = (high_stride, high_stride, num_classes + 5)
    low_label_shape = (low_stride, low_stride, num_classes + 5)

    # shuffle
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_indices = indices[start:end]
        current_batch_size = len(batch_indices)
        
        # Preallocate arrays for the batch
        x_batch = np.empty((current_batch_size, img_size, img_size, 3), dtype=np.float32)
        y_batch_high = np.empty((current_batch_size, *high_label_shape), dtype=np.float32)
        y_batch_low = np.empty((current_batch_size, *low_label_shape), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            image_id = image_files[idx]
            # Load and process the image
            image_path = os.path.normpath(os.path.join(data_path, image_id + '.jpg'))
            image = cv2.imread(image_path)

            # Load annotation
            label_file = os.path.join(label_path, image_id + '.txt')
            annotation = load_data_from_txt(label_file)

            # image resize
            # image = cv2.resize(image, (img_size, img_size))

            # pad if necessary with a probabiliy
            image, annotation = letterbox_and_pad(image, annotation, final_size=img_size, prob=prob)

            # Data Augmentation
            image, annotation = random_horizontal_flip(image, annotation, prob=prob)
            image = random_color_jitter(image, brightness_range=0.2, contrast_range=0.4, prob=prob)
            image, annotation = random_crop(image, annotation, scale_range=(0.5, 1.0), prob=prob)

            # Encode labels
            high_label, low_label = encode(
                annotation, num_classes, high_stride, low_stride, area_threshold=0.08
            )
            
            image = image / 255.0  # Normalize
            x_batch[i] = image
            y_batch_high[i] = high_label
            y_batch_low[i] = low_label
        

        yield x_batch, (y_batch_high, y_batch_low)


# Wrap the generator in a tf.data.Dataset:
def create_tf_dataset(num_classes, high_stride, low_stride, img_size,
                      data_path, label_path, batch_size=32, shuffle=True, prob=0.5):
    output_signature = (
        tf.TensorSpec(shape=(None, img_size, img_size, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, high_stride, high_stride, num_classes+5), dtype=tf.float32),
            tf.TensorSpec(shape=(None, low_stride, low_stride, num_classes+5), dtype=tf.float32)
        )
    )
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(num_classes, high_stride, low_stride, img_size,
                                 data_path, label_path, batch_size, shuffle, prob),
        output_signature=output_signature
    )
    # Prefetch allows the next batch to be prepared while the current one is being processed.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


#%%
# Example usage:
if __name__ == '__main__':
    from drawImage import draw_result_imshow
    import labelDecode
    data_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/images/val2017'
    label_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/labels_bbox/val2017'
    num_classes = 80
    high_stride = 20
    low_stride = 5
    img_size = 360
    batch_size = 32
    
    # Create the generator
    train_gen = data_generator(num_classes, high_stride, low_stride, img_size,
                               data_path, label_path, batch_size=1, shuffle=True)
    
    x_batch, [y_batch_high, y_batch_low] = next(train_gen)
    print("x_batch shape:", x_batch.shape)
    print("y_batch_high shape:", y_batch_high.shape)
    print("y_batch_low shape:", y_batch_low.shape)

    # Create the tf.data.Dataset
    train_dataset = create_tf_dataset(num_classes, high_stride, low_stride, img_size,
                                      data_path, label_path, batch_size=batch_size, shuffle=True)
    

    coco_cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    for i in range(10):
        x_data, [y_data_high, y_data_low] = next(iter(train_gen))
        print(x_data.shape, y_data_high.shape, y_data_low.shape)
        box = labelDecode.decode(y_data_high[0], y_data_low[0], high_stride, low_stride, conf_threshold=0.3, original_image=True)
        print(box)
        draw_result_imshow(np.array(x_data[0]), box, coco_cls_names)
# #%%

