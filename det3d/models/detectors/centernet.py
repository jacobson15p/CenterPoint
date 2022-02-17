from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch
import torch.nn as nn

from ..registry import DETECTORS
from .. import builder
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss




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
  def __init__(self, tasks, backbone, train_cfg=None, test_cfg=None):
    
    super(BaseHead, self).__init__()
    print('Creating model...')
    self.model = builder.build_backbone(backbone)

    self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = [len(t["class_names"]) for t in tasks]
    self.scales = [1]
    self.pause = True
    self.flip_test = False
    self.down_ratio = 4
    self.crit = FastFocalLoss()
    self.crit_reg = RegLoss()

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    #if self.fix_res:
    #  inp_height, inp_width = self.opt.input_h, self.opt.input_w
    #  c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    #  s = max(height, width) * 1.0
    #else:
    inp_height = (new_height | 31) + 1
    inp_width = (new_width | 31) + 1
    c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
    s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.down_ratio, 
            'out_width': inp_width // self.down_ratio}
    return images, meta

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

  def forward(self, image_or_path_or_tensor, return_loss = True, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    elif isinstance(image_or_path_or_tensor, dict):
      image = image_or_path_or_tensor['images'][0]
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      #images = images.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, forward_time = self.process(images, return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      
      #dets = self.post_process(dets, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
    
    #results = self.merge_outputs(detections)
    results = output
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if return_loss:
      pred_dict = image_or_path_or_tensor
      hm_loss = self.crit(output['hm'], pred_dict['hm_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['mask_cam'][0], pred_dict['cat_cam'][0])
      dep_loss = self.crit_reg(output['dep'], pred_dict['mask_cam'][0], pred_dict['ind_cam'][0],
        pred_dict['dep'][0].unsqueeze(-1))

      loss = hm_loss + dep_loss

      return {'loss': loss.detach().cpu(), 'hm_loss': hm_loss.detach().cpu(), 'dep_loss': dep_loss.detach().cpu(),}

    else:
      return {'results': results, }


@DETECTORS.register_module
class DddHead(BaseHead):
  def __init__(self, tasks, backbone, train_cfg=None, test_cfg=None):
    super(DddHead, self).__init__(tasks, backbone)
    self.calib = np.array([[707.0493, 0, 604.0814, 45.75831],
                           [0, 707.0493, 180.5066, -0.3454157],
                           [0, 0, 1., 0.004981016]], dtype=np.float32)


  def pre_process(self, image, scale, calib=None):
    height, width = image.shape[0:2]
    
    inp_height, inp_width = height, width
    c = np.array([width / 2, height / 2], dtype=np.float32)
    #if self.opt.keep_res:
    s = np.array([inp_width, inp_height], dtype=np.int32)
    #else:
    #  s = np.array([width, height], dtype=np.int32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    images = image #cv2.resize(image, (width, height))
    #inp_image = cv2.warpAffine(
    #  resized_image, trans_input, (inp_width, inp_height),
    #  flags=cv2.INTER_LINEAR)
    #inp_image = (inp_image / 255.)
    #inp_image = (inp_image - self.mean) / self.std
    #images = inp_image.transpose(2, 0, 1).unsqueeze(0)
    calib = np.array(calib, dtype=np.float32) if calib is not None \
            else self.calib
    #images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.down_ratio, 
            'out_width': inp_width // self.down_ratio,
            'calib': calib}
    return images, meta
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      #wh = output['wh'] #if self.opt.reg_bbox else None
      #reg = output['reg'] #if self.opt.reg_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      #dets = ddd_decode(output['hm'], output['dep'], wh=wh, reg=reg)
      dets = [0]
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def merge_outputs(self, detections):
    results = detections[0]
    for j in range(1, self.num_classes + 1):
      if len(results[j] > 0):
        keep_inds = (results[j][:, -1] > 0.2)
        results[j] = results[j][keep_inds]
    return results