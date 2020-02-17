import numpy as np
import cv2

from mmdet.apis import init_detector, inference_detector
import mmcv
import os

config_file = './experiments/exp2020_01_28_retinanet/retinanet_r50_fpn_1x.py'
checkpoint_file = './experiments/exp2020_01_28_retinanet/epoch_12.pth'
classes_det = ['Text']
image_path = '3.png'

model_detection = init_detector(config_file , checkpoint_file , device='cuda:0')

def decode_detection(bboxes, labels, class_names, score_thr=0):
    """Decode result
    Args:
        bboxes (ndarray): Bounding boxes (with scores), (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
    Return:
        will return json file. Example of json format is given below.
        {0:{'bbox': [], 'label': }, 1:{'bbox': [], 'label': }, ...}
        bbox format = [xmin, ymin, xmax, ymax]
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5

    all_res = {}
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    detection_cnt = 0
    for bbox, label in zip(bboxes, labels):
        res = {}
        bbox_int = bbox.astype(np.int32)
        bbox_int = bbox_int.tolist()
        label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)

        res['label'] = label_text
        res['score'] = str(bbox[-1])
        res['bbox'] = bbox_int[:-1]
        res['label_txt'] = label_text + '|{:.02f}'.format(bbox[-1])

        all_res[str(detection_cnt)] = res
        detection_cnt = detection_cnt + 1

    return all_res

img = cv2.imread(image_path)
result, img_meta = inference_detector(model_detection, img)
bboxes = np.vstack(result)
labels = [ np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
labels = np.concatenate(labels)
mmcv.imshow_det_bboxes(img, bboxes , labels, score_thr=0.3, show=True, out_file="result.jpg")
Detections = decode_detection(bboxes, labels, classes_det, score_thr=0.3)