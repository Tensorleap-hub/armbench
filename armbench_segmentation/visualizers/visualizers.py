import numpy as np
from code_loader.contract.visualizer_classes import LeapImageWithBBox

from armbench_segmentation.utils.general_utils import get_mask_list, remove_label_from_bbs
from armbench_segmentation.utils.ioa_utils import get_ioa_array

from code_loader.inner_leap_binder.leapbinder_decorators import *


# Visualizers
@tensorleap_custom_visualizer('bb_decoder', LeapDataType.ImageWithBBox)
def bb_decoder(image, bb_prediction):
    """
    Overlays the BB predictions on the image
    """
    bb_object, _ = get_mask_list(bb_prediction, None, is_gt=False)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


@tensorleap_custom_visualizer('bb_gt_decoder', LeapDataType.ImageWithBBox)
def gt_bb_decoder(image, bb_gt) -> LeapImageWithBBox:
    bb_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)

@tensorleap_custom_visualizer('under segment', LeapDataType.ImageWithBBox)
def under_segmented_bb_visualizer(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8
    ioas = get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='pred')
    th_arr = ioas > th
    matches_count = th_arr.astype(int).sum(axis=-1)
    relevant_bbs = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
    relevant_gts = np.where(np.any((th_arr)[relevant_bbs], axis=0))[0]  # [Indices of gts]
    bb_pred_object, _ = get_mask_list(y_pred_bb, None, is_gt=False)
    new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
    bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
    new_bb_array = [new_gt_objects[i] for i in relevant_gts] + [new_bb_pred_object[i] for i in relevant_bbs]
    return LeapImageWithBBox((image * 255).astype(np.float32), new_bb_array)


@tensorleap_custom_visualizer('over segment', LeapDataType.ImageWithBBox)
def over_segmented_bb_visualizer(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8
    ioas = get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='gt')
    th_arr = ioas > th
    matches_count = th_arr.astype(int).sum(axis=0)
    relevant_gts = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
    relevant_bbs = np.where(np.any(th_arr[..., relevant_gts], axis=1))[0]  # [Indices of gts]
    bb_pred_object, _ = get_mask_list(y_pred_bb, None, is_gt=False)
    new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
    bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
    new_bb_array = [new_gt_objects[i] for i in relevant_gts] + [new_bb_pred_object[i] for i in relevant_bbs]
    return LeapImageWithBBox((image * 255).astype(np.float32), new_bb_array)
