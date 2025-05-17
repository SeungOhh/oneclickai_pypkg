#%%
import os
import tensorflow as tf
from datetime import datetime

# import modules
from . import yoloModel8
from . import createDataGenerator
from .yoloLoss import yolo_loss_tf
from .load_model import load_model


def fit_yolo_model(train_data_path, train_label_path, val_data_path, val_label_path, epochs=100, batch_size=64):

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
    model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
        filepath=folder_name + '/yolo_model_epoch_{epoch}' + ext,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=False)


    ## tensorboard callback
    # tensorboard_callback = TensorBoard(log_dir = './logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
    #                                    histogram_freq=1, profile_batch='5,10')

    # train using generator
    model.fit(
        train_gen,
        # steps_per_epoch=4000,
        validation_data=val_gen,
        epochs=epochs,
        batch_size=batch_size,
        # use_multiprocessing=True,  # Use multiple processes to run the generator in parallel
        # workers=16,  # Number of worker processes
        callbacks=[model_checkpoint_callback, model_checkpoint_callback2]
    )

    # save the last model
    model.save(model_file_last)


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


