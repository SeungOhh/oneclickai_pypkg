#%%
import tensorflow as tf


def yolo_loss_tf(target, pred):

    # Loss weights
    lambda_coord = 5.0
    lambda_noobj = 0.5

    # -----------------------------
    # Separate out predictions
    # -----------------------------
    pred_x = pred[..., 0]
    pred_y = pred[..., 1]
    raw_w = pred[..., 2]
    raw_h = pred[..., 3]
    pred_w = tf.exp(raw_w)  # ensures width > 0
    pred_h = tf.exp(raw_h)  # ensures height > 0
    pred_conf = pred[..., 4]
    pred_logits = pred[..., 5:]  # raw logits for classes

    # -----------------------------
    # Separate out targets
    # -----------------------------
    tgt_x = target[..., 0]
    tgt_y = target[..., 1]
    tgt_w = target[..., 2]
    tgt_h = target[..., 3]
    tgt_conf = target[..., 4]      # 1 if object present, 0 otherwise
    tgt_classes = target[..., 5:]    # one-hot encoded classes

    # -----------------------------
    # Create object/no-object masks
    # -----------------------------
    obj_mask = tgt_conf           # mask for cells with objects
    noobj_mask = 1.0 - tgt_conf   # mask for cells without objects

    # -----------------------------
    # 1) Localization Loss (x, y, w, h)
    # -----------------------------
    # Loss for (x, y) coordinates using sum-of-squares (applied only where objects exist)
    loss_xy = obj_mask * ((tgt_x - pred_x) ** 2 + (tgt_y - pred_y) ** 2)
    loss_xy = tf.reduce_sum(loss_xy)

    # Loss for (w, h): using the squared difference of square roots
    eps = 1e-6  # small constant to avoid sqrt(0)
    loss_wh = obj_mask * (
        (tf.sqrt(tgt_w + eps) - tf.sqrt(pred_w + eps)) ** 2 +
        (tf.sqrt(tgt_h + eps) - tf.sqrt(pred_h + eps)) ** 2
    )
    loss_wh = tf.reduce_sum(loss_wh)

    loss_coord = lambda_coord * (loss_xy + loss_wh)

    # -----------------------------
    # 2) Objectness (Confidence) Loss
    # -----------------------------
    # Mean Squared Error (MSE) for confidence scores:
    loss_conf_obj = tf.reduce_sum(obj_mask * (tgt_conf - pred_conf) ** 2)
    loss_conf_noobj = tf.reduce_sum(noobj_mask * (tgt_conf - pred_conf) ** 2)
    loss_conf = loss_conf_obj + lambda_noobj * loss_conf_noobj

    # -----------------------------
    # 3) Classification Loss
    # -----------------------------
    # Compute binary cross-entropy loss (per class) for cells with objects
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tgt_classes, logits=pred_logits)
    bce = tf.reduce_sum(bce, axis=-1)  # sum over class probabilities
    loss_class = tf.reduce_sum(bce * obj_mask)

    # -----------------------------
    # Total Loss
    # -----------------------------
    total_loss = loss_coord + loss_conf + loss_class

    # tf.print('lossBox:', loss_coord, 'lossCnf:', loss_conf, 'lossCls:', loss_class)
    return total_loss




if __name__ == "__main__":
    pass

# %%
