---
title: Paper note 6 - PointPillars - Fast Encoders for Object Detection from Point Clouds
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-13 11:11:14 +0700
categories: [Computer Vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

[PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

# 1. Problem

- VoxelNet and PointNet is to slow.
- SECOND was improved but still slow.

=> <b> PointPillars: </b> a method for object detection in 3D that enables end-to-end learning with
only 2D convolutional layers. PointPillars uses <b> a novel encoder </b> that learn features on pillars (vertical columns) of the
point cloud to predict 3D oriented boxes for objects. There
are several advantages of this approach. 

- First, by learning
features instead of relying on fixed encoders, PointPillars
can leverage the full information represented by the point
cloud. 

- Further, by operating on pillars instead of voxels
there is no need to tune the binning of the vertical direction by hand. 

- Finally, pillars are highly efficient because all
key operations can be formulated as 2D convolutions which
are extremely efficient to compute on a GPU. 

An additional benefit of learning features is that PointPillars requires no
hand-tuning to use different point cloud configurations. For
example, it can easily incorporate multiple lidar scans, or
even radar point clouds.

# 2. PointPillars Network

PointPillars accepts point clouds as input and estimates
oriented 3D boxes for cars, pedestrians and cyclists. It consists of three main stages (Figure 2): 

(1) A feature encoder
network that converts a point cloud to a sparse pseudoimage; 

(2) A 2D convolutional backbone to process the
pseudo-image into high-level representation;

(3) A detection head that detects and regresses 3D boxes.

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-1.png?raw=true)

## 2.1. Pointcloud to Pseudo-Image

To apply a 2D convolutional architecture, we first convert the point cloud to a pseudo-image.
We denote by $l$ a point in a point cloud with coordinates
$x$, $y$, $z$ and reflectance $r$. As a first step the point cloud
is discretized into an evenly spaced grid in the x-y plane,
creating a set of pillars $P$ with $|P| = B$. Note that there is
no need for a hyper parameter to control the binning in the
z dimension. The points in each pillar are then augmented
with $x_c$, $y_c$, $z_c$, $x_p$ and $y_p$ where the $c$ subscript denotes
distance to the arithmetic mean of all points in the pillar and
the $p$ subscript denotes the offset from the pillar $x$, $y$ center.
The augmented lidar point $l$ is now $D = 9$ dimensional.
The set of pillars will be mostly empty due to sparsity
of the point cloud, and the non-empty pillars will in general
have few points in them. For example, at $0.162 m^2$ bins
the point cloud from an HDL-64E Velodyne lidar has 6k-9k
non-empty pillars in the range typically used in KITTI for
∼ 97% sparsity. This sparsity is exploited by imposing a
limit both on the number of non-empty pillars per sample
$(P)$ and on the number of points per pillar (N) to create a
dense tensor of size $(D, P, N)$. If a sample or pillar holds
too much data to fit in this tensor the data is randomly sampled. Conversely, if a sample or pillar has too little data to
populate the tensor, zero padding is applied.

Next, we use a simplified version of PointNet where,
for each point, a linear layer is applied followed by BatchNorm and ReLU to generate a $(C, P, N)$ sized
tensor. This is followed by a max operation over the channels to create an output tensor of size $(C, P)$. Note that the
linear layer can be formulated as a 1x1 convolution across
the tensor resulting in very efficient computation.
Once encoded, the features are scattered back to the
original pillar locations to create a pseudo-image of size
$(C, H, W)$ where $H$ and $W$ indicate the height and width
of the canvas.

## 2.2. Backbone

We use a similar backbone as VoxelNet and the structure is
shown in Figure 2. The backbone has two sub-networks:
one top-down network that produces features at increasingly small spatial resolution and a second network that performs upsampling and concatenation of the top-down features. The top-down backbone can be characterized by a series of blocks $Block(S, L, F)$. Each block operates at stride
$S$ (measured relative to the original input pseudo-image).
A block has $L$ 3x3 2D conv-layers with $F$ output channels,
each followed by BatchNorm and a ReLU. The first convolution inside the layer has stride $\frac{S}{S_{in}}$
to ensure the block
operates on stride $S$ after receiving an input blob of stride
$S_{in}$. All subsequent convolutions in a block have stride 1.
The final features from each top-down block are combined through upsampling and concatenation as follows.
First, the features are upsampled, $Up(S_{in}, S_{out}, F)$ from an
initial stride $S_{in}$ to a final stride Sout (both again measured
wrt. original pseudo-image) using a transposed 2D convolution with $F$ final features. Next, BatchNorm and ReLU
is applied to the upsampled features. The final output features are a concatenation of all features that originated from
different strides.

## 2.3. Detection Head

In this paper, we use the Single Shot Detector (SSD)
setup to perform 3D object detection. Similar to SSD, we
match the priorboxes to the ground truth using 2D Intersection over Union (IoU). Bounding box height and
elevation were not used for matching; instead given a 2D
match, the height and elevation become additional regression targets.

# 3. Implementation Details

## 3.1. Network

Instead of pre-training our networks, all weights were
initialized randomly using a uniform distribution.

The encoder network has C = 64 output features. The
car and pedestrian/cyclist backbones are the same except
for the stride of the first block (S = 2 for car, S = 1 for
pedestrian/cyclist). Both network consists of three blocks,
Block1(S, 4, C), Block2(2S, 6, 2C), and Block3(4S, 6,
4C). Each block is upsampled by the following upsampling
steps: Up1(S, S, 2C), Up2(2S, S, 2C) and Up3(4S, S, 2C).
Then the features of Up1, Up2 and Up3 are concatenated
together to create 6C features for the detection head.

## 3.2. Loss

We use the same loss function as SECOND.  Ground truth boxes and anchors are defined by
$(x, y, z, w, l, h, θ)$. The localization regression residuals between ground truth and anchors are defined by:

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-2.png?raw=true)

where $x^{gt}$ and $x^a$ are respectively the ground truth and anchor boxes and $d^a = \sqrt{ (w^a)^2 + (l^a)^2}$. The total localization loss is

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-3.png?raw=true)


Since the angle localization loss cannot distinguish
flipped boxes, we use a softmax classification loss on the
discretized directions , $L_{dir}$, which enables the network
to learn the heading.

For the object classification loss, we use thye focal loss:

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-5.png?raw=true)


where $p^a$
is the class probability of an anchor. We use the
original paper settings of $α = 0.25$ and $γ = 2$. The total
loss is therefore:

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-6.png?raw=true)


where Npos is the number of positive anchors and $β_{loc} = 2$, $β_{cls} = 1$, and $β_{dir} = 0.2$.
To optimize the loss function we use the Adam optimizer
with an initial learning rate of 2 ∗ 10−4
and decay the learning rate by a factor of 0.8 every 15 epochs and train for 160
epochs. We use a batch size of 2 for validation set and 4 for
our test submission.

# 4. Experiments

- [Code](https://github.com/nutonomy/second.pytorch)

## Code Explanation

### Pointcloud to Pseudo-Image

```python
class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas
```

### Backbone

```python
class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs
```

### Detection Head

```python
class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
```

### Full Network

```python
class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=3, 
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.nclasses = nclasses
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                        point_cloud_range=point_cloud_range, 
                                        max_num_points=max_num_points, 
                                        max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                            point_cloud_range=point_cloud_range, 
                                            in_channel=9, 
                                            out_channel=64)
        self.backbone = Backbone(in_channel=64, 
                                 out_channels=[64, 128, 256], 
                                 layer_nums=[3, 5, 5])
        self.neck = Neck(in_channels=[64, 128, 256], 
                         upsample_strides=[1, 2, 4], 
                         out_channels=[128, 128, 128])
        self.head = Head(in_channel=384, n_anchors=2*nclasses, n_classes=nclasses)
        
        # anchors
        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations=[0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, 
                                         sizes=sizes, 
                                         rotations=rotations)
        
        # train
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)

        # bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        # bbox_pred: (bs, n_anchors*7, 248, 216)
        # bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   
```

