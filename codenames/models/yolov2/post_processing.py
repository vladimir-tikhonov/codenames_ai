"""
Process [GRID x GRID x BOXES x (4 + 1 + CLASSES)]. Filter low confidence
boxes, apply NMS and return boxes, scores, classes.
"""

import numpy as np
import tensorflow as tf
from keras import backend as keras

import codenames.config.yolo as yolo_config


def generate_yolo_grid(g: np.ndarray, num_bb: int) -> np.ndarray:  # pylint: disable=invalid-name
    c_x = keras.cast(keras.reshape(keras.tile(keras.arange(g), [g]), (1, g, g, 1, 1)), keras.floatx())
    c_y = keras.permute_dimensions(c_x, (0, 2, 1, 3, 4))
    return keras.tile(keras.concatenate([c_x, c_y], -1), [1, 1, 1, num_bb, 1])


def process_outs(b: np.ndarray, s: np.ndarray, c: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    b_p = b
    # Expand dims of scores and classes so we can concat them
    # with the boxes and have the output of NMS as an added layer of YOLO.
    # Have to do another expand_dims this time on the first dim of the result
    # since NMS doesn't know about BATCH_SIZE (operates on 2D, see
    # https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
    # but Keras needs this dimension in the output.
    s_p = keras.expand_dims(s, axis=-1)
    c_p = keras.expand_dims(c, axis=-1)

    output_stack = keras.concatenate([b_p, s_p, c_p], axis=1)
    return keras.expand_dims(output_stack, axis=0)


class PostProcessing:
    def __call__(self, y_sing_pred: np.ndarray) -> np.ndarray:
        num_bounding_boxes = len(yolo_config.ANCHORS) // 2
        c_grid = generate_yolo_grid(yolo_config.GRID_SIZE, num_bounding_boxes)
        anchors = np.reshape(yolo_config.ANCHORS, [1, 1, 1, num_bounding_boxes, 2])

        # need to convert b's from gridSize units into IMG coords. Divide by grid here.
        b_xy = (keras.sigmoid(y_sing_pred[..., 0:2]) + c_grid[0]) / yolo_config.GRID_SIZE
        b_wh = (keras.exp(y_sing_pred[..., 2:4]) * anchors[0]) / yolo_config.GRID_SIZE
        b_xy1 = b_xy - b_wh / 2.
        b_xy2 = b_xy + b_wh / 2.
        boxes = keras.concatenate([b_xy1, b_xy2], axis=-1)

        # filter out scores below detection threshold
        scores_all = keras.sigmoid(y_sing_pred[..., 4:5]) * keras.softmax(y_sing_pred[..., 5:])
        indicator_detection = scores_all > yolo_config.DETECTION_THRESHOLD
        scores_all = scores_all * keras.cast(indicator_detection, np.float32)

        # compute detected classes and scores
        classes = keras.argmax(scores_all, axis=-1)
        scores = keras.max(scores_all, axis=-1)

        # flattened tensor length
        s2b = (yolo_config.GRID_SIZE ** 2) * num_bounding_boxes

        # flatten boxes, scores for NMS
        flatten_boxes = keras.reshape(boxes, shape=(s2b, 4))
        flatten_scores = keras.reshape(scores, shape=(s2b,))
        flatten_classes = keras.reshape(classes, shape=(s2b,))

        ids = []

        # only include boxes of the current class, with > 0 confidence
        class_mask = keras.cast(keras.equal(flatten_classes, 0), np.float32)
        score_mask = keras.cast(flatten_scores > 0, np.float32)
        mask = class_mask * score_mask

        # compute class NMS
        nms_ids = tf.image.non_max_suppression(
            flatten_boxes,
            flatten_scores * mask,
            max_output_size=yolo_config.MAX_BOXES,
            iou_threshold=yolo_config.IOU_THRESHOLD,
            score_threshold=yolo_config.SCORE_THRESHOLD
        )

        ids.append(nms_ids)

        # combine winning box indices of all classes
        selected_indices = keras.concatenate(ids, axis=-1)

        # gather corresponding boxes, scores, class indices
        selected_boxes = keras.gather(flatten_boxes, selected_indices)
        selected_scores = keras.gather(flatten_scores, selected_indices)
        selected_classes = keras.gather(flatten_classes, selected_indices)

        return process_outs(selected_boxes, selected_scores, keras.cast(selected_classes, np.float32))
