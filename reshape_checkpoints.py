import torch
pretrained_weights  = torch.load("./experiments/try/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth")

num_class = 2
'''
pretrained_weights['state_dict']['bbox_head.0.fc_cls.weight'].resize_([num_class, 1024])
pretrained_weights['state_dict']['bbox_head.0.fc_cls.bias'].resize_([num_class])
pretrained_weights['state_dict']['bbox_head.1.fc_cls.weight'].resize_([num_class, 1024])
pretrained_weights['state_dict']['bbox_head.1.fc_cls.bias'].resize_([num_class])
pretrained_weights['state_dict']['bbox_head.2.fc_cls.weight'].resize_([num_class, 1024])
pretrained_weights['state_dict']['bbox_head.2.fc_cls.bias'].resize_([num_class])

pretrained_weights['state_dict']['bbox_head.fc_cls.weight'].resize_([num_class, 1024])
pretrained_weights['state_dict']['bbox_head.fc_cls.bias'].resize_([num_class])
pretrained_weights['state_dict']['bbox_head.fc_reg.weight'].resize_([8, 1024])
pretrained_weights['state_dict']['bbox_head.fc_reg.bias'].resize_([8])'''

pretrained_weights['state_dict']['bbox_head.cls_convs.0.weight'].resize_([8, 512, 3, 3])
pretrained_weights['state_dict']['bbox_head.cls_convs.0.bias'].resize_([8])
pretrained_weights['state_dict']['bbox_head.cls_convs.1.weight'].resize_([12, 1024, 3, 3])
pretrained_weights['state_dict']['bbox_head.cls_convs.1.bias'].resize_([12])
pretrained_weights['state_dict']['bbox_head.cls_convs.2.weight'].resize_([12, 512, 3, 3])
pretrained_weights['state_dict']['bbox_head.cls_convs.2.bias'].resize_([12])
pretrained_weights['state_dict']['bbox_head.cls_convs.3.weight'].resize_([12, 256, 3, 3])
pretrained_weights['state_dict']['bbox_head.cls_convs.3.bias'].resize_([12])
pretrained_weights['state_dict']['bbox_head.cls_convs.4.weight'].resize_([12, 256, 3, 3])
pretrained_weights['state_dict']['bbox_head.cls_convs.4.bias'].resize_([12])
pretrained_weights['state_dict']['bbox_head.cls_convs.5.weight'].resize_([8, 256, 3, 3])
pretrained_weights['state_dict']['bbox_head.cls_convs.5.bias'].resize_([8])
pretrained_weights['state_dict']['bbox_head.cls_convs.6.weight'].resize_([8, 256, 3, 3])
pretrained_weights['state_dict']['bbox_head.cls_convs.6.bias'].resize_([8])

torch.save(pretrained_weights, "./experiments/try/Ssd_512_1class.pth")