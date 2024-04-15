from code_loader.helpers.detection.utils import xyxy_to_xywh_format
from code_loader.helpers.detection.yolo.utils import jaccard, xywh_to_xyxy_format

from typing import Union
from numpy._typing import NDArray
from armbench_segmentation.yolo_helpers.yolo_utils import DECODER
from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue
import tensorflow as tf
import numpy as np
from armbench_segmentation.config import CONFIG


def transpose_bbox_coor(boxes: Union[NDArray[np.float32], tf.Tensor]) -> Union[NDArray[np.float32], tf.Tensor]:
    if isinstance(boxes, tf.Tensor):
        result = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)
    else:
        result = boxes[:, :, [0, 1, 3, 2]]
    return result


def confusion_matrix_metric(bb_gt: tf.Tensor, y_pred_bb: tf.Tensor):
    # assumes we get predictions in xyxy format in gt AND reg
    # assumes gt is in xywh form

    reg = y_pred_bb[:, :, :4]
    class_pred = y_pred_bb[:, :, 4:]
    cls = class_pred[..., :CONFIG['CLASSES']]

    #yolo match?

    # reg, cls = tf.transpose(reg, (0, 2, 1)), tf.transpose(cls, (0, 2, 1))
    id_to_name = {0: 'Object', 1: 'Tote'}
    threshold = CONFIG['CM_IOU_THRESH']
    # image_shape = CONFIG['IMAGE_SIZE']
    # reg_fixed = xyxy_to_xywh_format(reg)
    gt_boxes = bb_gt[..., :-1]
    gt_labels = bb_gt[..., -1]
    # gt_boxes = xyxy_to_xywh_format(gt_boxes)
    # gt_boxes = np.concatenate([
    #     gt_boxes[:, :, :2] / image_shape,  # Normalized (x1, y1)
    #     gt_boxes[:, :, 2:] / image_shape  # Normalized (w, h)
    # ], axis=2)
    # reg_normalized = tf.concat([
    #     reg_fixed[:, :, :2] / image_shape,  # Normalized (x1, y1)
    #     reg_fixed[:, :, 2:] / image_shape  # Normalized (w, h)
    # ], axis=2)
    outputs = DECODER(loc_data=[reg], conf_data=[cls], prior_data=[None],
                      from_logits=False, decoded=True)
    ret = []
    for batch_i in range(len(outputs)):
        confusion_matrix_elements = []
        if len(outputs[batch_i]) != 0:
            ious = jaccard(outputs[batch_i][:, 1:5],
                           tf.cast(gt_boxes[batch_i, :], tf.double)).numpy()  # (#bb_predicted,#gt)
            prediction_detected = np.any((ious > threshold), axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(gt_labels[batch_i, max_iou_ind[i]])
                class_name = id_to_name.get(gt_idx)
                gt_label = f"{class_name}"
                confidence = outputs[batch_i][i, 0]
                if prediction:  # TP
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(gt_label),
                        ConfusionMatrixValue.Positive,
                        float(confidence)
                    ))
                else:  # FP
                    class_name = id_to_name.get(int(outputs[batch_i][i, -1]))
                    pred_label = f"{class_name}"
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(pred_label),
                        ConfusionMatrixValue.Negative,
                        float(confidence)
                    ))
        else:  # No prediction
            ious = np.zeros((1, gt_boxes[batch_i, ...].shape[0]))
        gts_detected = np.any((ious > threshold), axis=0)
        for k, gt_detection in enumerate(gts_detected):
            label_idx = gt_labels[batch_i, k]
            if not gt_detection and label_idx != CONFIG['BACKGROUND_LABEL']:  # FN
                class_name = id_to_name.get(int(gt_labels[batch_i, k]))
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    f"{class_name}",
                    ConfusionMatrixValue.Positive,
                    float(0)
                ))
        if all(~ gts_detected):
            confusion_matrix_elements.append(ConfusionMatrixElement(
                "background",
                ConfusionMatrixValue.Positive,
                float(0)
            ))
        ret.append(confusion_matrix_elements)
    return ret
