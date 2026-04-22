#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from datetime import datetime
import shutil
import random
import matplotlib.pyplot as plt

# import modules
from . import yoloModel8
from . import createDataGenerator

from .yoloLoss import yolo_loss_tf
from .load_model import load_model


def _save_tflite(keras_model_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(
        tf.keras.models.load_model(keras_model_path, custom_objects={"yolo_loss_tf": yolo_loss_tf})
    )
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)


def _split_dataset(data_path, label_path, val_ratio=0.2, seed=42):
    """train 데이터를 8:2로 분리해 임시 폴더에 복사 후 경로 반환"""
    imgs = sorted([f for f in os.listdir(data_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    pairs = [(f, os.path.splitext(f)[0] + '.txt')
             for f in imgs
             if os.path.exists(os.path.join(label_path, os.path.splitext(f)[0] + '.txt'))]

    random.seed(seed)
    random.shuffle(pairs)
    split = int(len(pairs) * (1 - val_ratio))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    print(f"데이터 자동 분리: 학습 {len(train_pairs)}개 / 검증 {len(val_pairs)}개")

    tmp = '_yolo_tmp_split'
    t_img = os.path.join(tmp, 'images', 'train')
    t_lbl = os.path.join(tmp, 'labels', 'train')
    v_img = os.path.join(tmp, 'images', 'val')
    v_lbl = os.path.join(tmp, 'labels', 'val')
    for d in [t_img, t_lbl, v_img, v_lbl]:
        os.makedirs(d, exist_ok=True)

    for img, lbl in train_pairs:
        shutil.copy(os.path.join(data_path,  img), os.path.join(t_img, img))
        shutil.copy(os.path.join(label_path, lbl), os.path.join(t_lbl, lbl))
    for img, lbl in val_pairs:
        shutil.copy(os.path.join(data_path,  img), os.path.join(v_img, img))
        shutil.copy(os.path.join(label_path, lbl), os.path.join(v_lbl, lbl))

    return t_img, t_lbl, v_img, v_lbl, tmp


def fit_yolo_model(
    train_data_path,        # 학습 이미지 폴더 경로
    train_label_path,       # 학습 라벨 폴더 경로 (YOLO 형식 .txt)
    val_data_path=None,     # 검증 이미지 폴더 경로. None이면 학습 데이터에서 자동 8:2 분리
    val_label_path=None,    # 검증 라벨 폴더 경로. None이면 학습 데이터에서 자동 8:2 분리
    epochs=100,             # 전체 학습 반복 횟수
    batch_size=8,           # 한 번에 처리할 이미지 수 (GPU 메모리에 맞게 조절)
    save_tflite=True,       # True로 설정하면 학습 완료 후 best/last 모델을 .tflite로도 저장
):

    # val 경로가 없으면 train 데이터에서 8:2 자동 분리
    tmp_dir = None
    if val_data_path is None or val_label_path is None:
        train_data_path, train_label_path, val_data_path, val_label_path, tmp_dir = \
            _split_dataset(train_data_path, train_label_path)

    # create folder name with current time to save models
    folder_name = datetime.now().strftime("%Y%m%d_%H%M")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # save model extension, if tensorflow version is equal or less than 2.15.0, use .h5 extension
    # else use .keras extension
    if tf.__version__ <= '2.15.0':
        ext = '.h5'
    else:
        # use .keras extension
        ext = '.keras'


    # model file name
    model_file_best = folder_name + '/yolo_model_best' + ext
    model_file_last = folder_name + '/yolo_model_last' + ext


    num_classes = 80  # Example number of classes
    img_size = 320

    # create model
    model = load_model('YOLO_coco')
    # model = yoloModel8.create_yolov8_model(num_classes, img_size)
    high_stride = model.output_shape[0][1]
    low_stride = model.output_shape[1][1]
    # model.summary()

    # compile and fit model using adam solver with learning rate scheduler
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss=[yolo_loss_tf, yolo_loss_tf])

    # create data generator
    train_gen = createDataGenerator.create_tf_dataset(num_classes, high_stride, low_stride, img_size, train_data_path, train_label_path, batch_size=batch_size, shuffle=True, prob=0.5)
    val_gen = createDataGenerator.create_tf_dataset(num_classes, high_stride, low_stride, img_size, val_data_path, val_label_path, batch_size=batch_size, shuffle=False, prob=0.0)



    # create call back to save the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_file_best,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # save model every epoch with epoch number in file name
    # model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=folder_name + '/yolo_model_epoch_{epoch}' + ext,
    #     save_weights_only=False,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=False)


    ## tensorboard callback
    # tensorboard_callback = TensorBoard(log_dir = './logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
    #                                    histogram_freq=1, profile_batch='5,10')

    # train using generator
    history = model.fit(
        train_gen,
        # steps_per_epoch=4000,
        validation_data=val_gen,
        epochs=epochs,
        batch_size=batch_size,
        # use_multiprocessing=True,  # Use multiple processes to run the generator in parallel
        # workers=16,  # Number of worker processes
        callbacks=[model_checkpoint_callback]  # model_checkpoint_callback2]
    )

    # save the last model
    model.save(model_file_last)

    # plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder_name + '/training_history.png')
    plt.show()

    if save_tflite:
        _save_tflite(model_file_best, folder_name + '/yolo_model_best.tflite')
        _save_tflite(model_file_last, folder_name + '/yolo_model_last.tflite')

    # 자동 분리 시 생성한 임시 폴더 삭제
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


#%% example usage
if __name__ == '__main__':
    #% encode data
    # encode training data
    train_data_path = 'C:/Users/somethingsomething'
    train_label_path = 'C:/Users/somethingsomething'

    # # encode validation data
    val_data_path = 'C:/Users/somethingsomething'
    val_label_path = 'C:/Users/somethingsomething'

    # fit model
    fit_yolo_model(train_data_path, train_label_path, val_data_path, val_label_path, epochs=10)


