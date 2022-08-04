# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------

import logging
from collections import defaultdict
from det3d.core import box_torch_ops
import torch
from det3d.torchie.cnn import kaiming_init
from torch import double, nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss
from det3d.models.utils import Sequential
from ..registry import HEADS
import copy 
try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

from det3d.core.utils.circle_nms_jit import circle_nms

import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x

class SepHead(nn.Module):
    def __init__(
        self,
        in_channels, #share_channels means the last layer before they seperate, YZ Notes 
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads 
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv-1): #Twice, 0, 1 
                fc.add(nn.Conv2d(in_channels, head_conv,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))    

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)
        

    def forward(self, x):
        ret_dict = dict()        
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class DCNSepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_cls,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4) 
        
        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)  

        # heatmap prediction head 
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                kernel_size=3, stride=1, 
                padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target 
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)


    def forward(self, x):    
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret

@HEADS.register_module
class CenterFusionHead(nn.Module):
    def __init__(
        self,
        in_channels=[128,],
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
    ):
        super(CenterFusionHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights 
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        
        # YZ Notes 
        # common_heads is the 
        # common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel':(2,2)}, # (output_channel, num_conv)
        self.box_n_dim = 9 if 'vel' in common_heads else 7  
        self.use_direction_classifier = False 

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList() # YZ Notes, different from the task list, this is a module list for training purposes. 
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads) # Creating a head for each class 
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv))) #Here, added the head map 
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x)) #ret_dict is a dictionary of each head. 
        return ret_dicts, x

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            #print(preds_dict['hm'].shape,example['hm'][task_id].shape, example['ind'][task_id].shape, example['mask'][task_id].shape, example['cat'][task_id].shape)

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])
            #hm_loss = torch.tensor([0]).cuda()
            #ind (batch x max_objects) get the object location, and only work with the category 

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['vel'], preds_dict['rot']), dim=1)  
                else:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['rot']), dim=1)   
                    target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target                       
            else:
                raise NotImplementedError()

            ret = {}
 
            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip 
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                if double_flip:
                    # flip vy
                    batch_vel[:, 1, ..., 1] *= -1
                    # flip vx
                    batch_vel[:, 2, ..., 0] *= -1

                    batch_vel[:, 3] *= -1
                    
                    batch_vel = batch_vel.mean(dim=1)

                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                pass 
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, task_id)) 

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 


    def Unproject(points, Z, intrinsic, distortion):
        f_x = intrinsic[0, 0]
        f_y = intrinsic[1, 1]
        c_x = intrinsic[0, 2]
        c_y = intrinsic[1, 2]
        # This was an error before
        # c_x = intrinsic[0, 3]
        # c_y = intrinsic[1, 3]

        # Step 1. Undistort.
        points_undistorted = np.array([])
        if len(points) > 0:
            points_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=1), intrinsic, distortion, P=intrinsic)
        points_undistorted = np.squeeze(points_undistorted, axis=1)

        # Step 2. Reproject.
        result = []
        for idx in range(points_undistorted.shape[0]):
            z = Z[0] if len(Z) == 1 else Z[idx]
            x = (points_undistorted[idx, 0] - c_x) / f_x * z
            y = (points_undistorted[idx, 1] - c_y) / f_y * z
            result.append([x, y, z])
        return result


    @torch.no_grad()
    def projection_forward(self, example, image_out, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing
        Added image_output as well for fusion in heat map (hm) and possibly other features  
        """

        front_intrinsics= example['calib']['FRONT_INTRINSIC']
        front_extrinsics= example['calib']['FRONT_EXTRINSIC']

        # USE THIS FOR RESULT 
        cam_bev_feats = []
        bev_pos = []
        batch_idx=0
        for cam_hm, cam_dep, cam_features, front_intrinsic, front_extrinsic, viewrangemeters, feature_map_size, hm_pixel_size in zip(image_out['results']['hm'].cpu(), 
            example['dep_map'][:,np.newaxis,...].cpu().numpy(),image_out['results']['feature_map'].cpu(), front_intrinsics.cpu().numpy(), front_extrinsics.cpu().numpy(),
            example['range'], example['feature_map_size'], example['hm_pixel_size']):

            # USE THIS FOR EXAMPLE TESTING 
            #for cam_hm, cam_dep, front_intrinsic, front_extrinsic, viewrangemeters, feature_map_size, hm_pixel_size in zip(example['hm_cam'][0].cpu().numpy(), 
            #        example['dep_map'][np.newaxis,...], front_intrinsics.cpu().numpy(), front_extrinsics.cpu().numpy(),
            #        example['range'], example['feature_map_size'], example['hm_pixel_size']):
            
            # Assuming the down ratio is 4. 
            front_intrinsic/=4
            # USE THIS FOR RESULT 
            
            pixel_unproject= np.array(np.meshgrid(np.linspace(1,cam_dep.shape[2],cam_dep.shape[2]),
                                        np.linspace(1,cam_dep.shape[1],cam_dep.shape[1]))).T.reshape(-1, 2)
            # USE THIS FOR EXAMPLE TESTING 
            # pixel_unproject= np.array(np.meshgrid(np.linspace(1,cam_dep.shape[1],cam_dep.shape[1]),
            #                             np.linspace(1,cam_dep.shape[0],cam_dep.shape[0]))).T.reshape(-1, 2)

            
            pixel_depth= cam_dep.T.reshape(-1,1)
            pixel_unproject[:,0]= (pixel_unproject[:,0]- front_intrinsic[2]) / front_intrinsic[0]
            pixel_unproject[:,1]= (pixel_unproject[:,1]- front_intrinsic[3]) / front_intrinsic[1]
            pixel_unproject= pixel_unproject*pixel_depth[:,0:1]
            pixel_unproject= np.hstack((pixel_unproject,pixel_depth))

            nonzero_mask = cam_hm[0].T.reshape(-1) > 1
            class_thresholds = [0.9,0.8,0.1]
            for channel_idx in range(cam_hm.shape[0]):
                nonzero_mask = (cam_hm[channel_idx].T.reshape(-1) > class_thresholds[channel_idx]) | nonzero_mask
            pixel_unproject_subset = pixel_unproject[nonzero_mask,:]
            
            rotation_axis= np.array([[0,0,1],
                                    [1,0,0],
                                    [0,1,0]])
            pixel_unproject_subset= (rotation_axis @ pixel_unproject_subset.T)
            pixel_unproject_subset= ((front_extrinsic[0:3,0:3] @ pixel_unproject_subset) +  np.expand_dims(front_extrinsic[0:3,3], axis=1)).T    
            pixel_unproject[nonzero_mask,:] = pixel_unproject_subset      
            
            # Now x is pointing towards forward, y to left, and z up 

            pixel_unproject/=hm_pixel_size
            
            hm_new= np.zeros((cam_hm.shape[0], feature_map_size[0], feature_map_size[1]))
            cam_feats_new = np.zeros((cam_features.shape[0], feature_map_size[0], feature_map_size[1]))

            valid_mask = (pixel_unproject[:,0]< feature_map_size[0]/2) & (pixel_unproject[:,0]> 0) & (pixel_unproject[:,1]< feature_map_size[1]/2) & (pixel_unproject[:,1]> -feature_map_size[1]/2)
            valid_mask = nonzero_mask.numpy() & valid_mask
            
            valid_length = np.sum(valid_mask)
            pixel_with_taskHM = np.zeros((cam_hm.shape[0],valid_length))
            for channel_idx in range(cam_hm.shape[0]):
                pixel_with_taskHM[channel_idx] = cam_hm[channel_idx].T.reshape(-1)[valid_mask]

            pixel_with_taskfeat = np.zeros((cam_features.shape[0],valid_length))
            for channel_idx in range(cam_features.shape[0]):
                pixel_with_taskfeat[channel_idx] = cam_features[channel_idx].T.reshape(-1)[valid_mask]
            
            chans_hm = np.arange(0,cam_hm.shape[0],1)
            chans_feats = np.arange(0,cam_features.shape[0],1)
            pixel_unproject = pixel_unproject[valid_mask,:]
            pixel_unproject[:,0] = (-1*pixel_unproject[:,0] + feature_map_size[0]/2)
            pixel_unproject[:,1] = (pixel_unproject[:,1] + feature_map_size[1]/2)
            unique_pixels = np.unique(pixel_unproject.astype(int)[:,:2],axis=0)
            for i in range(unique_pixels.shape[0]):
                pixel_with_taskfeat[:,(pixel_unproject.astype(int)[:,:2] == unique_pixels[i]).all(axis=1)] = np.amax(pixel_with_taskfeat[:,(pixel_unproject.astype(int)[:,:2] == unique_pixels[i]).all(axis=1)],axis=1)[...,np.newaxis]

            hm_new[:,pixel_unproject[:,0].astype(int),pixel_unproject[:,1].astype(int)] = pixel_with_taskHM[chans_hm,:]
            cam_feats_new[:,pixel_unproject[:,0].astype(int),pixel_unproject[:,1].astype(int)] = pixel_with_taskfeat[chans_feats,:]

            hm_new = np.flip(np.rot90(hm_new,k=3,axes=(1,2)),axis=1)
            cam_feats_new = np.flip(np.rot90(cam_feats_new,k=3,axes=(1,2)),axis=1)
            bev_p = np.unique(np.array(np.where(hm_new != 0)[1:]),axis=1).T
            bev_p = torch.tensor(bev_p[:,[1,0]])

            #preds_dicts[0]['hm'][batch_idx]= torch.add(torch.sigmoid(preds_dicts[0]['hm'][batch_idx]),
            #                                            torch.tensor(hm_new.copy(), dtype=preds_dicts[0]['hm'].dtype, device=preds_dicts[0]['hm'].device))
            #preds_dicts[0]['hm'][batch_idx] = preds_dicts[0]['hm'][batch_idx]/torch.max(preds_dicts[0]['hm'][batch_idx])

            cam_bev_feats.append(cam_feats_new[np.newaxis,...])
            bev_pos.append(bev_p)
            batch_idx+=1
        
        cam_bev_feats = np.concatenate(cam_bev_feats,axis=0)
        bev_pos = torch.cat(bev_pos,axis=0)

        return torch.tensor(cam_bev_feats), bev_pos

    @torch.no_grad()
    def fusion_predict(self, example, preds_dicts, bev_feat, test_cfg, **kwargs):
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip 
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
            bev_pos = torch.cat((xs,ys),dim=2)

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                if double_flip:
                    # flip vy
                    batch_vel[:, 1, ..., 1] *= -1
                    # flip vx
                    batch_vel[:, 2, ..., 0] *= -1

                    batch_vel[:, 3] *= -1
                    
                    batch_vel = batch_vel.mean(dim=1)

                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                pass 
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, task_id, bev_pos, bev_feat)) 

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores", "query_pos", "bev_pos","query_feat","indices"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, task_id, bev_pos, bev_feat):
        batch_size = len(batch_hm)

        prediction_dicts = []
        bev_feat = bev_feat.view(bev_feat.shape[0],bev_feat.shape[1],-1)

        for i in range(batch_size):
            query_pos = torch.zeros((1,500,2))
            query_feat = torch.zeros((1,bev_feat.shape[1],500))
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            max_indices = mask.nonzero().flatten()
            query_pos[0,:len(max_indices),:] = bev_pos[i,mask,:]
            query_feat[0,:,:len(max_indices)] = bev_feat[i,:,mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.get('circular_nms', False):
                centers = boxes_for_nms[:, [0, 1]] 
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(boxes, min_radius=test_cfg.min_radius[task_id], post_max_size=test_cfg.nms.nms_post_max_size)  
            else:
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(), 
                                    thresh=test_cfg.nms.nms_iou_threshold,
                                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                    post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels,
                'query_pos': query_pos,
                'bev_pos': bev_pos[i].unsqueeze(0),
                'query_feat': query_feat,
                'indices': max_indices, 
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 

import numpy as np 
def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep  

def create_2D_grid(x_size, y_size):
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
    batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
    batch_x = batch_x + 0.5
    batch_y = batch_y + 0.5
    coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
    coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
    return coord_base