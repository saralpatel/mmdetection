import torch
from collections import OrderedDict

custom_checkpoint = OrderedDict()

#Get Backbone
pretrained_weights  = torch.load("/home/rlussier/pytorch_object_detection/mmdetectionABP/experiments/exp20191009CascadeRPNGCNet/cascade_mask_rcnn_r16_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b86027a6.pth")

for key, value in pretrained_weights['state_dict'].items():
    if "backbone" in key:
        custom_checkpoint[key] = value

#Get Neck

for key, value in pretrained_weights['state_dict'].items():
    if "neck" in key:
        custom_checkpoint[key] = value

#Get rpn

pretrained_weights  = torch.load("/home/rlussier/pytorch_object_detection/mmdetectionABP/experiments/exp20191008CascadeRPN/crpn_faster_rcnn_fpn_1x_20191008-cb1e5335.pth")

for key, value in pretrained_weights['state_dict'].items():
    if "rpn" in key or "offset" in key:
        custom_checkpoint[key] = value

#Get ROI_extractor

for key, value in pretrained_weights['state_dict'].items():
    if "bbox_roi_extractor" in key:
        custom_checkpoint[key] = value

#Get head #

pretrained_weights  = torch.load("/home/rlussier/pytorch_object_detection/mmdetectionABP/experiments/exp20191009CascadeRPNGCNet/cascade_mask_rcnn_r16_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b86027a6.pth")
for key, value in pretrained_weights['state_dict'].items():
    if "head" in key and not 'rpn' in key:
        custom_checkpoint[key] = value

torch.save(custom_checkpoint, '/home/rlussier/pytorch_object_detection/mmdetectionABP/experiments/exp20191009CascadeRPNGCNet/custom_checkpoint.pth')








