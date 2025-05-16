import tensorflow as tf
from tensorflow.keras import layers, Model

# ------------------------------------------------------------------
# Utility Blocks
# ------------------------------------------------------------------

def ConvBlock(x, filters, kernel_size=3, strides=1):
    """
    Convolution -> BatchNorm -> Activation (SiLU)
    """
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)  # SiLU (swish) activation
    return x

def Focus(x, filters, kernel_size=3, strides=1):
    """
    Optimized Focus module using tf.nn.space_to_depth.
    This operator rearranges blocks of spatial data into depth.
    """
    # Replace manual slicing with the highly optimized space_to_depth op.
    x = tf.nn.space_to_depth(x, block_size=2)  # equivalent to slicing and concatenating four parts.
    x = ConvBlock(x, filters, kernel_size, strides)
    return x

def Bottleneck(x, filters, shortcut=True):
    """
    A simple bottleneck block.
    """
    y = ConvBlock(x, filters, kernel_size=1)
    y = ConvBlock(y, filters, kernel_size=3)
    if shortcut and x.shape[-1] == filters:
        y = layers.Add()([x, y])
    return y

def CSPBlock(x, filters, n=1):
    """
    A simple CSP (Cross Stage Partial) block.
    Splits the input into two parts, processes one part with bottleneck(s),
    then concatenates with the other part and fuses them.
    """
    route = ConvBlock(x, filters, kernel_size=1)
    x = ConvBlock(x, filters, kernel_size=1)
    for _ in range(n):
        x = Bottleneck(x, filters, shortcut=True)
    x = layers.Concatenate()([x, route])
    x = ConvBlock(x, filters, kernel_size=1)
    return x

def SPP(x, filters, pool_sizes=(5, 9, 13)):
    """
    Spatial Pyramid Pooling (SPP) block.
    Pools the features at different scales and concatenates them.
    """
    x = ConvBlock(x, filters // 2, kernel_size=1)
    pooled_outputs = [x]
    for pool_size in pool_sizes:
        pooled = layers.MaxPooling2D(pool_size, strides=1, padding='same')(x)
        pooled_outputs.append(pooled)
    x = layers.Concatenate()(pooled_outputs)
    x = ConvBlock(x, filters, kernel_size=1)
    return x

# ------------------------------------------------------------------
# Backbone (only two scales: small & large)
# ------------------------------------------------------------------

def build_backbone(inputs):
    """
    Backbone network (YOLOv8-style) that produces two scales:
      - feature_small: High-resolution features (for small object detection)
      - feature_large: Low-resolution features (for large object detection)
    """
    # Focus: reduce spatial dims and enrich channels.
    # x = Focus(inputs, filters=32, kernel_size=3) 
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)  # 360→180
    # Stage 1: High-resolution features.
    x = ConvBlock(x, 32, kernel_size=3, strides=2)  # 160 -> 80
    x = ConvBlock(x, 32, kernel_size=3, strides=2)  # 80 -> 40
    x = ConvBlock(x, 32, kernel_size=3, strides=2)  # 40 -> 20
    x = CSPBlock(x, filters=64, n=2)
    feature_small = x  # resolution ~160x160

    # Stage 2: Downsample to obtain coarse features.
    x = ConvBlock(x, 32, kernel_size=3, strides=2)  # 20 -> 10
    x = ConvBlock(x, 32, kernel_size=3, strides=2)  # 10 -> 5
    x = CSPBlock(x, filters=64, n=2)
    x = SPP(x, filters=64)
    feature_large = x  # resolution ~40x40

    return feature_small, feature_large

# ------------------------------------------------------------------
# Neck: Two-Scale Feature Fusion (Top-Down & Bottom-Up)
# ------------------------------------------------------------------

def build_neck_two_scales(feature_small, feature_large):
    """
    Fuses the high-resolution (feature_small) and low-resolution (feature_large) features.
    - Top-down: Upsample feature_large to fuse with feature_small.
    - Bottom-up: Downsample the fused feature to refine the coarse feature.
    Returns two refined features for detection.
    """
    # Top-down pathway: upsample feature_large and fuse with feature_small.
    up_feature = ConvBlock(feature_large, 64, kernel_size=1)
    up_feature = layers.UpSampling2D(size=4, interpolation='nearest')(up_feature) # 5 -> 20
    p_small = layers.Concatenate()([feature_small, up_feature])
    p_small = CSPBlock(p_small, filters=128, n=2)

    # Bottom-up pathway: downsample p_small back to coarse resolution.
    x = ConvBlock(p_small, 64, kernel_size=3, strides=2)  # 20 -> 10
    down_feature = ConvBlock(x, 64, kernel_size=3, strides=2)  # 10 -> 5
    p_large = layers.Concatenate()([feature_large, down_feature])
    p_large = CSPBlock(p_large, filters=128, n=2)

    return p_small, p_large

# ------------------------------------------------------------------
# YOLO Head
# ------------------------------------------------------------------

def build_yolo_head(x, num_classes):
    """
    YOLO head that predicts bounding box coordinates, objectness score,
    and class probabilities at each grid cell.
    The number of output channels = 4 (bbox) + 1 (objectness) + num_classes.
    """
    output_filters = 4 + 1 + num_classes
    x = layers.Conv2D(output_filters, kernel_size=1, strides=1, padding='same')(x)
    return x

# ------------------------------------------------------------------
# Assemble YOLOv8-Style Model (Two-Scale Version)
# ------------------------------------------------------------------

def create_yolov8_model(num_classes, image_size=640):
    """
    Create a YOLOv8-style model with two detection heads corresponding to:
      - A high-resolution feature (for small objects)
      - A low-resolution feature (for large objects)
    """
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    
    # Backbone produces two scales.
    feature_small, feature_large = build_backbone(inputs)
    
    # Neck fuses the two scales.
    p_small, p_large = build_neck_two_scales(feature_small, feature_large)
    
    # Detection heads.
    head_small = build_yolo_head(p_small, num_classes)
    head_large = build_yolo_head(p_large, num_classes)
    
    model = Model(inputs, [head_small, head_large])
    return model

# Example: create the model for 80 classes (e.g., COCO) with 640x640 input images.
if __name__ == '__main__':
    num_classes = 80
    model = create_yolov8_model(num_classes, image_size=320)
    model.summary()

    # save model
    model.save('yolo_model8.h5')