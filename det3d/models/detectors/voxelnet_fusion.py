from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
from .. import builder
import torch 
from copy import deepcopy 
from matplotlib import pyplot as plt

@DETECTORS.register_module
class VoxelNetFusion(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        image_head_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        freeze=True,
    ):
        super(VoxelNetFusion, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.image_head= builder.build_detector(image_head_cfg)

        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.freeze()
        self.image_head.freeze()
        for p in self.bbox_head.parameters():
            p.requires_grad = True

        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)
        bev_feature = x
        image_out= self.image_head.forward_two_stage(example,return_loss,**kwargs)
        cam_bev_feats = self.bbox_head.projection_forward(example ,image_out, self.test_cfg)
        bev_feature = self.bbox_head.fusion_forward(bev_feature,cam_bev_feats)
        preds, _ = self.bbox_head(bev_feature)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.fusion_predict(example, preds , self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 



    def forward_with_2Dfusion(self, example, image_out, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        # Preds only come with the HM and stuff, not the full prediction, considering adding something here 

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach() # will not have the gradient again. 
                new_preds.append(new_pred)

            #boxes, cam_bev_feats = self.bbox_head.predict_with_fusion(example, new_preds,image_out, self.test_cfg)
            #bev_feature = torch.cat((bev_feature,torch.tensor(cam_bev_feats,device=bev_feature.get_device())),axis=1).float()
            cam_bev_feats = self.bbox_head.projection_forward(example, new_preds ,image_out, self.test_cfg)
            bev_feature, fusion_hm = self.bbox_head.fusion_forward(bev_feature,cam_bev_feats)
            new_preds[0]['hm'] = fusion_hm
            boxes = self.bbox_head.fusion_predict(example, new_preds , self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            #boxes, cam_bev_feats = self.bbox_head.predict_with_fusion(example, preds, image_out, self.test_cfg)
            #bev_feature = torch.cat((bev_feature,torch.tensor(cam_bev_feats,device=bev_feature.get_device())),axis=1).float()
            cam_bev_feats = self.bbox_head.projection_forward(example, preds ,image_out, self.test_cfg)
            bev_feature, fusion_hm = self.bbox_head.fusion_forward(bev_feature,cam_bev_feats)
            preds[0]['hm'] = fusion_hm
            boxes = self.bbox_head.fusion_predict(example, preds , self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 