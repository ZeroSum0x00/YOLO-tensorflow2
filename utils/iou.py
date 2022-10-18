import math
import tensorflow as tf
from tensorflow.keras import backend as K


def box_iou(b1, b2):

    b1          = K.expand_dims(b1, -2)
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    b2          = K.expand_dims(b2, 0)
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def iou_loss(b1, b2, iou_method='ciou'):
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / K.maximum(union_area, K.epsilon())

    enclose_mins    = K.minimum(b1_mins, b2_mins)
    enclose_maxes   = K.maximum(b1_maxes, b2_maxes)
    enclose_wh      = K.maximum(enclose_maxes - enclose_mins, 0.0)
    center_wh       = b1_xy - b2_xy
    
    if iou_method.lower() == 'ciou':
        center_distance = K.sum(K.square(center_wh), axis=-1)

        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
        ciou    = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())
        
        v       = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
        alpha   = v / K.maximum((1.0 - iou + v), K.epsilon())
        out     = ciou - alpha * v

    elif iou_method.lower() == 'siou':
        sigma       = tf.pow(center_wh[..., 0] ** 2 + center_wh[..., 1] ** 2, 0.5)

        sin_alpha_1 = tf.abs(center_wh[..., 0]) / K.maximum(sigma, K.epsilon())
        sin_alpha_2 = tf.abs(center_wh[..., 1]) / K.maximum(sigma, K.epsilon())

        threshold   = pow(2, 0.5) / 2
        sin_alpha   = tf.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)

        angle_cost  = tf.cos(tf.asin(sin_alpha) * 2 - math.pi / 2)
        gamma       = 2 - angle_cost

        rho_x           = (center_wh[..., 0] / K.maximum(enclose_wh[..., 0], K.epsilon())) ** 2
        rho_y           = (center_wh[..., 1] / K.maximum(enclose_wh[..., 1], K.epsilon())) ** 2
        distance_cost   = 2 - tf.exp(-gamma * rho_x) - tf.exp(-gamma * rho_y)

        omiga_w     = tf.abs(b1_wh[..., 0] - b2_wh[..., 0]) / K.maximum(tf.maximum(b1_wh[..., 0], b2_wh[..., 0]), K.epsilon())
        omiga_h     = tf.abs(b1_wh[..., 1] - b2_wh[..., 1]) / K.maximum(tf.maximum(b1_wh[..., 1], b2_wh[..., 1]), K.epsilon())
        shape_cost  = tf.pow(1 - tf.exp(-1 * omiga_w), 4) + tf.pow(1 - tf.exp(-1 * omiga_h), 4)
        out         = iou - 0.5 * (distance_cost + shape_cost)

    return K.expand_dims(out, -1)


# def box_ciou(b1, b2):

#     b1_xy = b1[..., :2]
#     b1_wh = b1[..., 2:4]
#     b1_wh_half = b1_wh/2.
#     b1_mins = b1_xy - b1_wh_half
#     b1_maxes = b1_xy + b1_wh_half

#     b2_xy = b2[..., :2]
#     b2_wh = b2[..., 2:4]
#     b2_wh_half = b2_wh/2.
#     b2_mins = b2_xy - b2_wh_half
#     b2_maxes = b2_xy + b2_wh_half

#     intersect_mins = K.maximum(b1_mins, b2_mins)
#     intersect_maxes = K.minimum(b1_maxes, b2_maxes)
#     intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
#     intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#     b1_area = b1_wh[..., 0] * b1_wh[..., 1]
#     b2_area = b2_wh[..., 0] * b2_wh[..., 1]
#     union_area = b1_area + b2_area - intersect_area
#     iou = intersect_area / K.maximum(union_area, K.epsilon())

#     center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
#     enclose_mins = K.minimum(b1_mins, b2_mins)
#     enclose_maxes = K.maximum(b1_maxes, b2_maxes)
#     enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)

#     enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
#     ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())
    
#     v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
#     alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
#     ciou = ciou - alpha * v

#     ciou = K.expand_dims(ciou, -1)
#     return ciou
