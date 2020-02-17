import torch
import mmcv
import cv2
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import collate, scatter

#----------------------  Input arguments  --------------------------------

checkpoint_path = 'epoch_1.pth'
image_path = './mmdetectionABP/3.png'
classes_det = ['Text']

#----------------------   Model loading  ---------------------------------

checkpoint = torch.load(checkpoint_path)
model = checkpoint['network']
model.load_state_dict(checkpoint['state_dict'])
for parameter in model.parameters():
	parameter.requires_grad = False
model.eval()
device = next(model.parameters()).device

#---------------------  image pre process  --------------------------------

def img_pre_process(img):
	''' To pre process the image for given network. Values used in this function can be found in config file (test pipeline)
	Input argument:
	img: Image read by openCv or mmcv
	Return:
	data: dictionary of preprocessed data.
	''' 
	data = {}
	result = {}
	result['ori_shape'] = img.shape
	img, scale_factor = mmcv.imrescale(img, (1333,800), return_scale=True)
	result['img_shape'] = img.shape
	result['scale_factor'] = scale_factor
	mean = np.array([123.675, 116.28 , 103.53] ,dtype=np.float32)
	std = np.array([58.395, 57.12 , 57.375],dtype=np.float32)
	img = mmcv.imnormalize(img, mean, std, True)
	img = mmcv.impad_to_multiple(img, 32, pad_val=0)
	result['pad_shape'] = img.shape
	img = img.transpose(2, 0, 1)
	img = torch.from_numpy(img)
	result['filename'] = None
	result['flip'] = False
	result['img_norm_cfg'] = {'mean' : mean, 'std' : std, 'to_rgb' : True}
	data['img_meta'] = [DC(result, cpu_only=True)]
	data['img'] = [img]
	data = scatter(collate([data], samples_per_gpu=1), ['cuda:0'])[0]
	return data

#-------------------- Result decoding -------------------------------

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

#----------------- main --------------------------------------

def main(image_path):
	image = cv2.imread(image_path)
	data = img_pre_process(image)
	with torch.no_grad():
		res = model(return_loss=False, rescale=True, **data)
	bboxes = np.vstack(res)
	labels = [ np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
	labels = np.concatenate(labels)
	mmcv.imshow_det_bboxes(image, bboxes , labels, score_thr=0.3, show=True, out_file="result.jpg")
	Detections = decode_detection(bboxes, labels, classes_det, score_thr=0.3)

main(image_path)