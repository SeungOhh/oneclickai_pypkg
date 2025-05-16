import tensorflow as tf
import gdown
import os
from .yoloLoss import yolo_loss_tf



# load model
def load_model(model_path = None):

    # if model path is not given
    if model_path is None:
        raise ValueError("model_path is None, please provide a valid path or try \"YOLO\"")
    
    elif model_path == "YOLO_coco":
        download_model_from_gdrive("https://drive.google.com/file/d/1HuiUq4q1mJdlX9PKmGnL505YPTHE8DLZ/view?usp=sharing")
        model_path = 'yolo_tf.h5'


    model = tf.keras.models.load_model(model_path, custom_objects={"yolo_loss_tf": yolo_loss_tf})
    return model





def download_model_from_gdrive(url, destination='yolo_tf.h5'):
    dest = os.path.abspath(destination)

    # Skip if the file already exists and is not empty
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"[gdown] Found existing file → {dest}  (skipping download)")
        return dest

    print("[gdown] Downloading model …")
    gdown.download(url, dest, quiet=False, fuzzy=True)  # resumes if partial
    print(f"[gdown] Model saved to {dest}")
    return dest







if __name__ == '__main__':



    # model_path = '20250215_2106/yolo_model_epoch_63.h5'
    # model = load_model(model_path)
    # print(model.summary())

    # model_path = '20250215_2106/yolo_tf.h5'
    # model = load_model(model_path)
    # print(model.summary())

    # model_path = './yolo_tf.h5'
    # model = load_model(model_path)
    # print(model.summary())


    model = load_model('YOLO')
    print(model.summary())