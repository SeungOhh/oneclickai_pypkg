import random
import numpy as np
import cv2

def random_horizontal_flip(image, bboxes, prob=0.5):
    """
    :param image:  NumPy array (H x W x 3), dtype float or uint8
    :param bboxes: List of bounding boxes, each in [cls, x_center, y_center, w, h] (YOLO format)
    :param flip_prob: Probability of flipping
    :return: (image, bboxes) after potential horizontal flip
    """
    if random.random() < prob:
        # Flip image horizontally
        image = cv2.flip(image, 1)
        
        # Flip bboxes: x_center -> 1 - x_center
        # y_center, w, h stay the same
        # cls stays the same
        new_bboxes = []

                # if no bounding boxes, return
        if bboxes == None:
            return image, bboxes
        
        for box in bboxes:
            cls_id, x, y, w, h = box
            x_new = 1.0 - x
            new_bboxes.append([cls_id, x_new, y, w, h])
        bboxes = new_bboxes

    return image, bboxes


def random_color_jitter(image, brightness_range, contrast_range, prob):
    """
    Randomly adjusts brightness and contrast of the image.
    :param image: NumPy array in range [0,1] or [0,255]
    :param brightness_range: float, max fraction for brightness shift
    :param contrast_range: float, max fraction for contrast shift
    :param p: probability to apply this augmentation
    :return: augmented image
    """
    if random.random() < prob:
        # Convert to float32 for robust operations
        img_float = image.astype(np.float32)        

        # Random brightness
        brightness_factor = random.uniform(-brightness_range, brightness_range)
        # Random contrast
        contrast_factor = random.uniform(-contrast_range, contrast_range)

        # Apply adjustments
        img_float = img_float + (img_float-127.5) * contrast_factor + brightness_factor * 255.0

        # Clip to valid range
        img_float = np.clip(img_float, 0, 255)
        
        
        # Convert back to original dtype
        image = img_float.astype(image.dtype)

    return image


def random_crop(image, bboxes, scale_range=(0.8, 1.0), prob=0.5):
    """
    Randomly crops a region from the image, updates YOLO-format bounding boxes accordingly.
    
    :param image:   NumPy array (H x W x 3), BGR or RGB, in [0,255] or [0,1].
    :param bboxes:  List of bounding boxes in YOLO format: [cls, x_center, y_center, w, h],
                    with x_center, y_center, w, h normalized to [0, 1].
    :param scale_range:  (min_scale, max_scale) for the crop dimension as a fraction of the original size.
    :param p:       Probability of applying the crop.
    :return:        (cropped_image, new_bboxes)
    """
    if random.random() > prob:
        # No crop
        return image, bboxes

    h, w = image.shape[:2]
    
    # Choose random crop size
    min_scale, max_scale = scale_range
    crop_scale = random.uniform(min_scale, max_scale)
    crop_h = int(h * crop_scale)
    crop_w = int(w * crop_scale)

    # Choose top-left corner for the crop
    # We can place the crop anywhere in the original image
    y0 = random.randint(0, h - crop_h)
    x0 = random.randint(0, w - crop_w)

    # Crop the image
    cropped_image = image[y0:y0+crop_h, x0:x0+crop_w]
    resized_image = cv2.resize(cropped_image, (w, h))

    # Adjust bounding boxes
    new_bboxes = []

    # if no bounding boxes, return
    if bboxes == None:
        return resized_image, bboxes
    for box in bboxes:
        cls_id, x_rel, y_rel, w_rel, h_rel = box
        x_center_abs = x_rel * w
        y_center_abs = y_rel * h
        bw_abs = w_rel * w
        bh_abs = h_rel * h
        
        # Convert to [x1, y1, x2, y2] in absolute coords
        x1 = x_center_abs - bw_abs/2
        y1 = y_center_abs - bh_abs/2
        x2 = x_center_abs + bw_abs/2
        y2 = y_center_abs + bh_abs/2
            
        # Clip coords to the crop region
        # Shift them by (x0, y0)
        x1_c = x1 - x0 # if x1 is outside the crop, it will be negative
        y1_c = y1 - y0 # if y1 is outside the crop, it will be negative
        x2_c = x2 - x0 # if x2 is outside the crop, it will be larger than crop_w
        y2_c = y2 - y0 # if y2 is outside the crop, it will be larger than crop_h
        
        # If the box is completely outside the crop, skip it
        # or you could partially clip it (below).
        # We'll partially clip:
        x1_c = max(0, x1_c)
        y1_c = max(0, y1_c)
        x2_c = min(crop_w, x2_c)
        y2_c = min(crop_h, y2_c)
        
        new_w = x2_c - x1_c
        new_h = y2_c - y1_c
        
        # Discard if no area
        if new_w <= 1 or new_h <= 1:
            continue
        
        # Convert back to YOLO [x_c, y_c, w, h] in normalized coords
        x_c_rel = (x1_c + new_w/2) / crop_w
        y_c_rel = (y1_c + new_h/2) / crop_h
        w_rel = new_w / crop_w
        h_rel = new_h / crop_h
        
        # Ensure coords are within [0, 1]
        x_c_rel = min(max(x_c_rel, 0), 1)
        y_c_rel = min(max(y_c_rel, 0), 1)
        w_rel = min(w_rel, 1)
        h_rel = min(h_rel, 1)
        
        new_bboxes.append([cls_id, x_c_rel, y_c_rel, w_rel, h_rel])
        # print('cls', cls_id)
        # print('x1_c, y1_c, x2_c, y2_c:', x1_c, y1_c, x2_c, y2_c)


    

    return resized_image, new_bboxes



def letterbox_and_pad(image, bboxes, final_size=320, color=(114, 114, 114), prob = 0.5):
    
    orig_h, orig_w = image.shape[:2]

    aspect_ratio = orig_w / orig_h

    if aspect_ratio < 1.6 and aspect_ratio > 0.6:
        if random.random() < prob:
            # resize image to make it square
            image = cv2.resize(image, (final_size, final_size))
            return image, bboxes
    


    # Calculate scale to fit the image in a square of size final_size
    scale = min(final_size / orig_w, final_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create the padded canvas
    # final_size x final_size, fill with "color"
    padded_image = np.full((final_size, final_size, 3), color, dtype=resized.dtype)

    # Compute top-left corner to place the resized image (we'll center it)
    dw = (final_size - new_w) // 2
    dh = (final_size - new_h) // 2

    # Place resized image onto padded canvas
    padded_image[dh:dh+new_h, dw:dw+new_w] = resized

    # Update bounding boxes: scale + shift
    new_bboxes = []

    # if no bounding boxes, return
    if bboxes == None:
        return padded_image, bboxes
    
    for box in bboxes:
        # Convert YOLO -> absolute
        cls_id, x_rel, y_rel, w_rel, h_rel = box
        x_center_abs = x_rel * orig_w
        y_center_abs = y_rel * orig_h
        w_abs = w_rel * orig_w
        h_abs = h_rel * orig_h
        x1 = x_center_abs - w_abs / 2
        x2 = x_center_abs + w_abs / 2
        y1 = y_center_abs - h_abs / 2
        y2 = y_center_abs + h_abs / 2

        # Scale coords
        x1s = x1 * scale
        x2s = x2 * scale
        y1s = y1 * scale
        y2s = y2 * scale
        # Shift by (dw, dh)
        x1p = x1s + dw
        x2p = x2s + dw
        y1p = y1s + dh
        y2p = y2s + dh

        # Ensure coords are within [0..final_size]
        x1p = max(0, min(final_size, x1p))
        x2p = max(0, min(final_size, x2p))
        y1p = max(0, min(final_size, y1p))
        y2p = max(0, min(final_size, y2p))

        bw = x2p - x1p
        bh = y2p - y1p
        if bw < 1 or bh < 1:
            # Discard boxes too small after scaling/padding
            continue

        # Convert back to YOLO normalized
        x_c = (x1p + x2p) / 2.0
        y_c = (y1p + y2p) / 2.0
        w_box = bw
        h_box = bh
        x_rel = x_c / final_size
        y_rel = y_c / final_size
        w_rel = w_box / final_size
        h_rel = h_box / final_size

        new_bboxes.append([cls_id, x_rel, y_rel, w_rel, h_rel])

    return padded_image, new_bboxes