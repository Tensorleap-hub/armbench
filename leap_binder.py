import os
from typing import Union, List, Dict, Optional
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from code_loader import leap_binder
from code_loader.contract.enums import LeapDataType

from code_loader.contract.datasetclasses import PreprocessResponse
from pycocotools.coco import COCO

from armbench_segmentation.config import CONFIG, local_filepath
from armbench_segmentation.data.preprocessing import load_set
from armbench_segmentation.utils.confusion_matrix import confusion_matrix_metric
from armbench_segmentation.utils.general_utils import count_obj_masks_occlusions, \
    count_obj_bbox_occlusions, extract_and_cache_bboxes
from armbench_segmentation.metrics import instance_seg_loss, compute_losses, dummy_loss
from armbench_segmentation.visualizers.visualizers import gt_bb_decoder, bb_decoder, \
    under_segmented_bb_visualizer, over_segmented_bb_visualizer
from armbench_segmentation.visualizers.visualizers_getters import mask_visualizer_gt, mask_visualizer_prediction
from armbench_segmentation.utils.general_utils import get_mask_list, get_argmax_map_and_separate_masks
from armbench_segmentation.utils.ioa_utils import ioa_mask
from armbench_segmentation.metrics import over_under_segmented_metrics
from armbench_segmentation.utils.gcs_utils import _download
from code_loader.inner_leap_binder.leapbinder_decorators import *


# ----------------------------------------------------data processing--------------------------------------------------

@tensorleap_preprocess()
def subset_images() -> List[PreprocessResponse]:
    """
    This function returns the training and validation datasets in the format expected by tensorleap
    """
    np.random.seed(0)
    local_paths = []
    subset_names = ["train", "val"]
    res = []
    if CONFIG['LOCAL_FLAG'] is False:
        for sub in subset_names:
            remote_filepath = os.path.join(CONFIG['remote_dir'], f"{sub}.json")
            local_paths.append(_download(remote_filepath))
    else:
        for sub in subset_names:
            local_paths.append(os.path.join(local_filepath, f'{sub}.json'))

    for path, sub in zip(local_paths, subset_names):
        # initialize COCO api for instance annotations
        coco = COCO(path)
        x_raw = load_set(coco=coco, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'], local_filepath=local_filepath)
        # sub_size = min(len(x_raw), CONFIG[f'{sub.upper()}_SIZE'])
        idx = np.random.choice(len(x_raw), len(x_raw), replace=False)
        res.append(PreprocessResponse(length=len(x_raw), data={'cocofile': coco,
                                                               'samples': np.take(x_raw, idx),
                                                               'subdir': f'{sub}'}))
    return res


@tensorleap_unlabeled_preprocess()
def unlabeled_preprocessing_func() -> PreprocessResponse:
    """
    This function returns the unlabeled data split in the format expected by tensorleap
    """
    if CONFIG['LOCAL_FLAG'] is False:
        remote_filepath = os.path.join(CONFIG['remote_dir'], f"test.json")
        local_path = _download(remote_filepath)
    else:
        local_path = os.path.join(local_filepath, f'test.json')

    test = COCO(local_path)
    x_test_raw = load_set(coco=test, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'], local_filepath=local_filepath)
    # test_size = min(len(x_test_raw), CONFIG['UL_SIZE'])
    np.random.seed(0)
    test_idx = np.random.choice(len(x_test_raw), len(x_test_raw), replace=False)
    return PreprocessResponse(length=len(x_test_raw), data={'cocofile': test,
                                                            'samples': np.take(x_test_raw, test_idx),
                                                            'subdir': 'test'})


@tensorleap_input_encoder('images')
def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    """
    Returns a BGR image normalized and padded
    """
    data = data.data
    x = data['samples'][idx]

    if CONFIG['LOCAL_FLAG'] is False:
        remote_filepath = os.path.join(CONFIG['remote_dir'], "images", x['file_name'])

        local_path = _download(remote_filepath)
    else:
        local_path = os.path.join(local_filepath, "images", x['file_name'])
    # rescale
    image = np.array(
        Image.open(local_path).resize((CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]), Image.BILINEAR)) / 255.
    return image.astype(np.float32)


def get_annotation_coco(idx: int, data: PreprocessResponse) -> np.ndarray:
    x = data.data['samples'][idx]
    coco = data.data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    return anns


@tensorleap_gt_encoder('masks')
def get_masks(idx: int, data: PreprocessResponse) -> np.ndarray:
    MASK_SIZE = (160, 160)
    coco = data.data['cocofile']
    anns = get_annotation_coco(idx, data)
    masks = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], *MASK_SIZE], dtype=np.uint8)
    for i in range(min(len(anns), CONFIG['MAX_BB_PER_IMAGE'])):
        ann = anns[i]
        mask = coco.annToMask(ann)
        mask = cv2.resize(mask, (MASK_SIZE[0], MASK_SIZE[1]), cv2.INTER_NEAREST)
        masks[i, ...] = mask
    return masks.astype(np.float32)


@tensorleap_gt_encoder('bbs')
def get_bbs(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    bboxes = extract_and_cache_bboxes(idx, data)
    return bboxes.astype(np.float32)


# ----------------------------------------------------------metadata----------------------------------------------------
def get_cat_instances_seg_lst(img: np.ndarray, masks: np.ndarray) -> Union[List[np.ma.array], None]:
    if masks is None:
        return None
    if masks[0, ...].shape != CONFIG['IMAGE_SIZE']:
        masks = tf.image.resize(masks[..., None], CONFIG['IMAGE_SIZE'], tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        masks = masks.numpy()
    instances = []
    for mask in masks:
        mask = np.broadcast_to(mask[..., np.newaxis], img.shape)
        masked_arr = np.ma.masked_array(img, mask)
        instances.append(masked_arr)
    return instances


def get_fname(index: int, subset: PreprocessResponse) -> str:
    data = subset.data
    x = data['samples'][index]
    return x['file_name']


def get_original_width(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return x['width']


def get_original_height(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return x['height']


def bbox_num(bbs: np.ndarray) -> int:
    number_of_bb = np.count_nonzero(bbs[..., -1] != CONFIG['BACKGROUND_LABEL'])
    return number_of_bb


def get_avg_bb_area(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()


def get_avg_bb_aspect_ratio(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()


def get_instances_num(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    return float(valid_bbs.shape[0])


def get_object_instances_num(bbs: np.ndarray) -> float:
    label = CONFIG['CATEGORIES'].index('Object')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def get_tote_instances_num(bbs: np.ndarray) -> float:
    label = CONFIG['CATEGORIES'].index('Tote')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def get_avg_instance_percent(mask: np.ndarray, bboxs: np.ndarray) -> float:
    valid_masks = mask[bboxs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    if valid_masks.size == 0:
        return 0.
    res = np.sum(valid_masks, axis=(1, 2))
    size = valid_masks[0, :, :].size
    return np.mean(np.divide(res, size))


def get_tote_instances_masks(mask: np.ndarray, bboxs: np.ndarray) -> Optional[float]:
    label = CONFIG['CATEGORIES'].index('Tote')
    valid_masks = mask[bboxs[..., -1] == label]
    if valid_masks.size == 0:
        return None
    return valid_masks


def get_tote_instances_sizes(tote_instances_masks: Optional[float]) -> float:
    masks = tote_instances_masks
    if masks is None:
        return 0
    res = np.sum(masks, axis=(1, 2))
    size = masks[0, :, :].size
    return np.divide(res, size)


def get_tote_avg_instance_percent(tote_instances_sizes) -> float:
    return float(np.mean(tote_instances_sizes))


def get_tote_std_instance_percent(tote_instances_sizes) -> float:
    return float(np.std(tote_instances_sizes))


def get_tote_instances_mean(cat_instances_seg_lst_tote) -> float:
    if cat_instances_seg_lst_tote is None:
        return -1
    return np.array([i.mean() for i in cat_instances_seg_lst_tote]).mean()


def get_tote_instances_std(cat_instances_seg_lst_tote) -> float:
    if cat_instances_seg_lst_tote is None:
        return -1
    return np.array([i.std() for i in cat_instances_seg_lst_tote]).std()


def get_object_instances_mean(cat_instances_seg_lst_object) -> float:
    if cat_instances_seg_lst_object is None:
        return -1
    return np.array([i.mean() for i in cat_instances_seg_lst_object]).mean()


def get_object_instances_std(cat_instances_seg_lst_object) -> float:
    if cat_instances_seg_lst_object is None:
        return -1
    return np.array([i.std() for i in cat_instances_seg_lst_object]).std()


def get_object_instances_masks(mask: np.ndarray, bboxs: np.ndarray) -> Union[np.ndarray, None]:
    label = CONFIG['CATEGORIES'].index('Object')
    valid_masks = mask[bboxs[..., -1] == label]
    if valid_masks.size == 0:
        return None
    return valid_masks


def get_object_instances_sizes(object_instances_masks) -> float:
    masks = object_instances_masks
    if masks is None:
        return 0
    res = np.sum(masks, axis=(1, 2))
    size = masks[0, :, :].size
    return np.divide(res, size)


def get_object_avg_instance_percent(object_instances_sizes) -> float:
    return float(np.mean(object_instances_sizes))


def get_object_std_instance_percent(object_instances_sizes) -> float:
    return float(np.std(object_instances_sizes))


def get_background_percent(masks: np.ndarray, bboxs: np.ndarray) -> float:
    valid_masks = masks[bboxs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    if valid_masks.size == 0:
        return 1.0
    res = np.sum(valid_masks, axis=0)
    size = valid_masks[0, :, :].size
    return float(np.round(np.divide(res[res == 0].size, size), 3))


def get_obj_bbox_occlusions_count(img: np.ndarray, bboxes: np.ndarray, calc_avg_flag=False) -> float:
    occlusion_threshold = 0.2  # Example threshold value
    occlusions_count = count_obj_bbox_occlusions(img, bboxes, occlusion_threshold, calc_avg_flag)
    return occlusions_count


def get_obj_bbox_occlusions_avg(img: np.ndarray, bboxes: np.ndarray) -> float:
    return get_obj_bbox_occlusions_count(img, bboxes, calc_avg_flag=True)


def get_obj_mask_occlusions_count(object_instances_masks) -> int:
    occlusion_threshold = 0.1  # Example threshold value
    masks = object_instances_masks
    occlusion_count = count_obj_masks_occlusions(masks, occlusion_threshold)
    return occlusion_count


def count_duplicate_bbs(bbs_gt: np.ndarray) -> int:
    real_gt = bbs_gt[bbs_gt[..., 4] != CONFIG['BACKGROUND_LABEL']]
    return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])


def count_small_bbs(bboxes: np.ndarray) -> float:
    obj_boxes = bboxes[bboxes[..., -1] == 0]
    areas = obj_boxes[..., 2] * obj_boxes[..., 3]
    return float(len(areas[areas < CONFIG['SMALL_BBS_TH']]))


@tensorleap_metadata('metadata')
def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    bbs = get_bbs(idx, data)
    masks = get_masks(idx, data)
    img = input_image(idx, data)

    tote_instances_masks = get_tote_instances_masks(masks, bbs)
    object_instances_masks = get_object_instances_masks(masks, bbs)
    cat_instances_seg_lst_object = get_cat_instances_seg_lst(img, object_instances_masks)
    cat_instances_seg_lst_tote = get_cat_instances_seg_lst(img, tote_instances_masks)
    tote_instances_sizes = get_tote_instances_sizes(tote_instances_masks)
    object_instances_sizes = get_object_instances_sizes(object_instances_masks)

    metadatas = {
        "idx": idx,
        "fname": get_fname(idx, data),
        "origin_width": get_original_width(idx, data),
        "origin_height": get_original_height(idx, data),
        "instances_number": get_instances_num(bbs),
        "tote_number": get_tote_instances_num(bbs),
        "object_number": get_object_instances_num(bbs),
        "avg_instance_size": get_avg_instance_percent(masks, bbs),
        "tote_instances_mean": get_tote_instances_mean(cat_instances_seg_lst_tote),
        "tote_instances_std": get_tote_instances_std(cat_instances_seg_lst_tote),
        "object_instances_mean": get_object_instances_mean(cat_instances_seg_lst_object),
        "object_instances_std": get_object_instances_std(cat_instances_seg_lst_object),
        "tote_avg_instance_size": get_tote_avg_instance_percent(tote_instances_sizes),
        "tote_std_instance_size": get_tote_std_instance_percent(tote_instances_sizes),
        "object_avg_instance_size": get_object_avg_instance_percent(object_instances_sizes),
        "object_std_instance_size": get_object_std_instance_percent(object_instances_sizes),
        "bbox_number": bbox_num(bbs),
        "bbox_area": get_avg_bb_area(bbs),
        "bbox_aspect_ratio": get_avg_bb_aspect_ratio(bbs),
        "background_percent": get_background_percent(masks, bbs),
        "duplicate_bb": count_duplicate_bbs(bbs),
        "small_bbs_number": count_small_bbs(bbs),
        "count_total_obj_bbox_occlusions": get_obj_bbox_occlusions_count(img, bbs),
        "avg_obj_bbox_occlusions": get_obj_bbox_occlusions_avg(img, bbs),
        "count_obj_mask_occlusions": get_obj_mask_occlusions_count(object_instances_masks)
    }

    return metadatas


@tensorleap_custom_metric('general_metrics')
def general_metrics_dict(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                         mask_gt: tf.Tensor, segmentation_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
    try:
        reg_met, class_met, obj_met, mask_met = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    except Exception as e:
        batch_dim = bb_gt.shape[0]
        fault_res_tensor = [tf.convert_to_tensor(-np.ones((batch_dim, 1))) for _ in range(3)]
        reg_met, class_met, obj_met, mask_met = (fault_res_tensor, fault_res_tensor, fault_res_tensor, fault_res_tensor)
    res = {
        "Regression_metric": tf.reduce_sum(reg_met, axis=0)[:, 0].numpy().astype(np.float32),
        "Classification_metric": tf.reduce_sum(class_met, axis=0)[:, 0].numpy().astype(np.float32),
        "Objectness_metric": tf.reduce_sum(obj_met, axis=0)[:, 0].numpy().astype(np.float32),
        "Mask_metric": tf.reduce_sum(mask_met, axis=0)[:, 0].numpy().astype(np.float32),
    }
    return res


@tensorleap_custom_metric('segmentation_metrics')
def segmentation_metrics_dict(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                              mask_gt: tf.Tensor) -> Dict[str, Union[int, float]]:
    bs = bb_gt.shape[0]
    bb_mask_gt = [get_mask_list(bb_gt[i, ...], mask_gt[i, ...], is_gt=True) for i in range(bs)]
    bb_mask_pred = [get_mask_list(y_pred_bb[i, ...], y_pred_mask[i, ...], is_gt=False) for i in range(bs)]
    sep_mask_pred = [get_argmax_map_and_separate_masks(image[i, ...], bb_mask_pred[i][0],
                                                       bb_mask_pred[i][1])['separate_masks'] for i in range(bs)]
    sep_mask_gt = [get_argmax_map_and_separate_masks(image[i, ...], bb_mask_gt[i][0],
                                                     bb_mask_gt[i][1])['separate_masks'] for i in range(bs)]
    pred_gt_ioas = [np.array([[ioa_mask(pred_mask, gt_mask) for gt_mask in sep_mask_gt[i]]
                              for pred_mask in sep_mask_pred[i]]) for i in range(bs)]
    gt_pred_ioas = [np.array([[ioa_mask(gt_mask, pred_mask) for gt_mask in sep_mask_gt[i]]
                              for pred_mask in sep_mask_pred[i]]) for i in range(bs)]
    gt_pred_ioas_t = [arr.transpose() for arr in gt_pred_ioas]
    over_seg_bool, over_seg_count, avg_segments_over, _, over_conf = \
        over_under_segmented_metrics(gt_pred_ioas_t, get_avg_confidence=True, bb_mask_object_list=bb_mask_pred)
    under_seg_bool, under_seg_count, avg_segments_under, under_small_bb, _ = \
        over_under_segmented_metrics(pred_gt_ioas, count_small_bbs=True, bb_mask_object_list=bb_mask_gt)
    res = {
        "Over_Segmented_metric": over_seg_bool.numpy().astype(np.float32),
        "Under_Segmented_metric": under_seg_bool.numpy().astype(np.float32),
        "Small_BB_Under_Segmtented": under_small_bb.numpy().astype(np.float32),
        "Over_Segmented_Instances_count": over_seg_count.numpy().astype(np.float32),
        "Under_Segmented_Instances_count": under_seg_count.numpy().astype(np.float32),
        "Average_segments_num_Over_Segmented": avg_segments_over.numpy().astype(np.float32),
        "Average_segments_num_Under_Segmented": avg_segments_under.numpy().astype(np.float32),
        "Over_Segment_confidences": over_conf.numpy().astype(np.float32)
    }
    return res


# ---------------------------------------------------------binding------------------------------------------------------
# set prediction (object)
leap_binder.add_prediction('object detection',
                           ["x", "y", "w", "h", "obj"] +
                           [f"class_{i}" for i in range(CONFIG['CLASSES'])] +
                           [f"mask_coeff_{i}" for i in range(32)])

# set prediction (segmentation)
leap_binder.add_prediction('segementation masks', [f"mask_{i}" for i in range(32)])


if __name__ == '__main__':
    leap_binder.check()
