#%%
import numpy as np


def _iou(box1, box2):
    # box format: [x_center, y_center, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1 - w1/2, x2 - w2/2)
    inter_y1 = max(y1 - h1/2, y2 - h2/2)
    inter_x2 = min(x1 + w1/2, x2 + w2/2)
    inter_y2 = min(y1 + h1/2, y2 + h2/2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0


def _nms(boxes, iou_threshold):
    # boxes: list of [class_id, x, y, w, h, conf]
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if b[0] != best[0] or _iou(best[1:5], b[1:5]) < iou_threshold]
    return kept


# yolo output to bbox
def decode(high_prediction, low_prediction, high_stride, low_stride, conf_threshold, iou_threshold=0.5, original_image=False):

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
            conf = min(1.0, conf)
            
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
            
            boxes.append([predicted_class, bx, by, bw, bh, conf])

    boxes = _nms(boxes, iou_threshold)
    return boxes  # [class_id, x, y, w, h, conf]


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
    