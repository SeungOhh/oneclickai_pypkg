#%%
import numpy as np


# yolo output to bbox
def decode(high_prediction, low_prediction, high_stride, low_stride, conf_threshold, original_image=False):

    boxes = []
    
    # Each tuple consists of (prediction_array, grid_stride)
    scales = [
        (high_prediction, high_stride),
        (low_prediction, low_stride)
    ]
    
    for pred, stride in scales:
        # Assume the confidence is at index 4.
        confs = pred[..., 4]
        valid_idxs = np.where(confs >= conf_threshold)
        
        # Determine how many class scores are provided
        num_classes = pred.shape[2] - 5
        
        # Iterate over grid cells with high confidence
        for row, col in zip(*valid_idxs):
            # Unpack the parameters for this grid cell.
            # Expected layout: [tx, ty, tw, th, conf, p1, p2, ...]
            tx, ty, tw, th, conf, *class_scores = pred[row, col, :]
            
            # Decode the center coordinates relative to the whole image.
            # (tx, ty) are the offsets inside the grid cell.
            bx = (col + tx) / stride
            by = (row + ty) / stride
            
            # Decode the box dimensions.
            # If not using the "original" dimensions, assume tw and th are in log-space.
            if original_image:
                bw = tw
                bh = th
            else:
                bw = np.exp(tw)
                bh = np.exp(th)
            
            
            # Determine the predicted class by taking the argmax of the class scores.
            # If there are no class scores, default to -1.
            if num_classes > 0:
                predicted_class = int(np.argmax(class_scores))
            else:
                predicted_class = -1
            
            # Append the decoded box as integers.
            boxes.append([predicted_class,
                          bx,
                          by,
                          bw,
                          bh])
    return boxes


#%%
# ===========================
# Example Usage
# ===========================
if __name__ == "__main__":
    import yoloModel
    import createDataGenerator
    from drawImage import draw_result_imshow

    num_classes = 80  # Example number of classes
    img_size = 416
    data_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/images/val2017'
    label_path = 'C:/Users/osy04/Desktop/wok_me/project/yolo/labels_bbox/val2017'

    model = yoloModel.create_yolo_model(num_classes, img_size)
    high_stride = model.output_shape[0][1]
    low_stride = model.output_shape[1][1]
    model.summary()

    train_gen = createDataGenerator.data_generator(num_classes, high_stride, low_stride, img_size,data_path, label_path, batch_size=1, shuffle=True, prob=0.5)
    x_data, [y_data_high, y_data_low] = next(train_gen)

    # Decode predictions
    decoded_boxes = decode(y_data_high[0], y_data_low[0], high_stride,low_stride,conf_threshold=0.3,original_image=True)
    
    print("Decoded Boxes:")
    for box in decoded_boxes:
        print(box)

    draw_result_imshow(np.array(x_data[0]), decoded_boxes)
    