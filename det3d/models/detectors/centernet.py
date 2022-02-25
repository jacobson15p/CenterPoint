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


from ..registry import DETECTORS
from .. import builder
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

class BaseHead(nn.Module):
  def __init__(self, tasks, backbone, pretrained=None, train_cfg=None, test_cfg=None):
    
    super(BaseHead, self).__init__()
    print('Creating model...')
    self.module = builder.build_backbone(backbone)

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
    
    output = self.process(images)
    
    if return_loss:
      pred_dict = x
      rets = []
      hm_loss = self.crit(output['hm'], pred_dict['hm_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['mask_cam'][0], pred_dict['cat_cam'][0])
      dep_loss = self.crit_reg(output['dep'], pred_dict['mask_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['dep'][0].unsqueeze(-1))

      loss = hm_loss + dep_loss

      losses = {'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'dep_loss': dep_loss.detach().cpu(),'num_positive': pred_dict['mask_cam'][0].float().sum(),}
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

      loss = hm_loss + dep_loss

      losses = {'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'dep_loss': dep_loss.detach().cpu(),'num_positive': pred_dict['mask_cam'][0].float().sum(),}
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
    super(DddHead, self).__init__(tasks, backbone)
    self.calib = np.array([[707.0493, 0, 604.0814, 45.75831],
                           [0, 707.0493, 180.5066, -0.3454157],
                           [0, 0, 1., 0.004981016]], dtype=np.float32)
  
  def process(self, images):
    #with torch.no_grad():
    output = self.module(images)[-1]
    output['hm'] = output['hm'].sigmoid_()
    #output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    #wh = output['wh'] #if self.opt.reg_bbox else None
    #reg = output['reg'] #if self.opt.reg_offset else None

    
    #dets = ddd_decode(output['hm'], output['dep'], wh=wh, reg=reg)
    #       
    return output


  def merge_outputs(self, detections):
    results = detections[0]
    for j in range(1, self.num_classes + 1):
      if len(results[j] > 0):
        keep_inds = (results[j][:, -1] > 0.2)
        results[j] = results[j][keep_inds]
    return results

class Classhead(object):
  def __init__(self, tasks):
    
    super(Classhead, self).__init__()
    self.class_names = [t["class_names"] for t in tasks]