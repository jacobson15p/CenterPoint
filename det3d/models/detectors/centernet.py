from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import L

import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from collections import defaultdict
from matplotlib import pyplot as plt


from ..registry import DETECTORS
from .. import builder
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss
from det3d.torchie.trainer import load_checkpoint




def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
    
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

class BaseHead(nn.Module):
  def __init__(self, tasks, backbone, pretrained=None, train_cfg=None, test_cfg=None):
    
    super(BaseHead, self).__init__()
    print('Creating model...')
    self.model = builder.build_backbone(backbone)
    #self.model = load_model(self.model,'pretrained_weights/nuScenes_3Ddetection_e140.pth')
    #self.model.to(torch.device('cuda'))

    self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = [len(t["class_names"]) for t in tasks]
    self.bbox_head = Classhead(tasks)
    self.scales = [1]
    self.pause = True
    self.flip_test = False
    self.down_ratio = 4
    self.crit = FastFocalLoss()
    self.crit_reg = RegLoss()
    self.pretrained= pretrained

    self.init_weights(pretrained=pretrained)

  def init_weights(self, pretrained=None):
    if pretrained is None:
        return 
    try:
        load_checkpoint(self, pretrained, strict=False)
        print("init weight from {}".format(pretrained))
    except:
        print("no pretrained model at {}".format(pretrained))

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def forward(self, x, return_loss = True, meta=None):
    images = x['images']
    #images = images.to(torch.device('cuda'))
    
    output = self.process(images)
    
    if return_loss:
      pred_dict = x
      rets = []
      hm_loss = self.crit(output['hm'], pred_dict['hm_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['mask_cam'][0], pred_dict['cat_cam'][0])
      dep_loss = self.crit_reg(output['dep'], pred_dict['mask_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['dep'][0].unsqueeze(-1))

      loss = hm_loss #+ 0.1*dep_loss

      losses = {'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'dep_loss': 0.1*dep_loss.detach().cpu(),'num_positive': pred_dict['mask_cam'][0].float().sum(),}
      rets.append(losses)

      """
      convert batch-key to key-batch
      """
      losses_merged = defaultdict(list)
      for ret in rets:
          for k, v in ret.items():
              losses_merged[k].append(v)

      return losses_merged

    else:
      return {'results': output, }



  def forward_two_stage(self, x, return_loss = True, meta=None):
    images = x['images']
    
    output = self.process(images)
    
    if return_loss:
      pred_dict = x
      rets = []
      hm_loss = self.crit(output['hm'], pred_dict['hm_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['mask_cam'][0], pred_dict['cat_cam'][0])
      dep_loss = self.crit_reg(output['dep'], pred_dict['mask_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['dep'][0].unsqueeze(-1))

      loss = hm_loss + 0.1*dep_loss

      losses = {'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'dep_loss': 0.1*dep_loss.detach().cpu(),'num_positive': pred_dict['mask_cam'][0].float().sum(),}
      rets.append(losses)

      """
      convert batch-key to key-batch
      """
      losses_merged = defaultdict(list)
      for ret in rets:
          for k, v in ret.items():
              losses_merged[k].append(v)

      return {'results': output, 'loss': losses_merged}

    else:
      return {'results': output, }



@DETECTORS.register_module
class DddHead(BaseHead):
  def __init__(self, tasks, backbone, pretrained=None, train_cfg=None, test_cfg=None):
    super(DddHead, self).__init__(tasks, backbone, pretrained)
    self.calib = np.array([[707.0493, 0, 604.0814, 45.75831],
                           [0, 707.0493, 180.5066, -0.3454157],
                           [0, 0, 1., 0.004981016]], dtype=np.float32)
  
  def process(self, images):

    output = self.model(images)[-1]
    output['hm'] = output['hm'].sigmoid_()
    #output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    #wh = output['wh'] #if self.opt.reg_bbox else None
    #reg = output['reg'] #if self.opt.reg_offset else None
    #      
    return output


  def merge_outputs(self, detections):
    results = detections[0]
    for j in range(1, self.num_classes + 1):
      if len(results[j] > 0):
        keep_inds = (results[j][:, -1] > 0.2)
        results[j] = results[j][keep_inds]
    return results

  def freeze(self):
    for p in self.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(self)
    return self

class Classhead(object):
  def __init__(self, tasks):
    
    super(Classhead, self).__init__()
    self.class_names = [t["class_names"] for t in tasks]