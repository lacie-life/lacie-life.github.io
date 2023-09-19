---
title: 3D Object Detection for Autonomous Driving - A Comprehensive Survey [Paper Lists]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-19 11:11:14 +0700
categories: [Computer Vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

[3D Object Detection for Autonomous Driving: A Comprehensive Survey](https://arxiv.org/abs/2206.09474)

![overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/overview.JPG?raw=true)

# 1. Data Source for 3D Object Detection

## 1.1. Datasets for 3D Object Detection

### 2022

- DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative
3D Object Detection [(CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_DAIR-V2X_A_Large-Scale_Dataset_for_Vehicle-Infrastructure_Cooperative_3D_Object_Detection_CVPR_2022_paper.pdf)

### 2021s

- One Million Scenes for Autonomous Driving: ONCE Dataset [(NeurIPS 21)](https://arxiv.org/pdf/2106.11037.pdf)

- Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting [(NeurIPS 21)](https://openreview.net/pdf?id=vKQGe36av4k)

- Cirrus: A Long-range Bi-pattern LiDAR Dataset [(ICRA 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561267)

- RADIATE: A Radar Dataset for Automotive Perception in Bad Weather [(ICRA 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9562089)

- PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving [(ITSC 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9565009)

- KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D [(arXiv 21)](https://arxiv.org/pdf/2109.13410.pdf)

- All-In-One Drive: A Large-Scale Comprehensive Perception Dataset with High-Density Long-Range Point Clouds [(arXiv 21)](https://www.researchgate.net/profile/Xinshuo-Weng/publication/347112693_All-In-One_Drive_A_Large-Scale_Comprehensive_Perception_Dataset_with_High-Density_Long-Range_Point_Clouds/links/5fd8156492851c13fe8925e8/All-In-One-Drive-A-Large-Scale-Comprehensive-Perception-Dataset-with-High-Density-Long-Range-Point-Clouds.pdf)

### 2020

- nuScenes: A Multimodal Dataset for Autonomous Driving [(CVPR 20)](https://arxiv.org/pdf/1903.11027.pdf)

- Scalability in Perception for Autonomous Driving: Waymo Open Dataset [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf)

- Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bijelic_Seeing_Through_Fog_Without_Seeing_Fog_Deep_Multimodal_Sensor_Fusion_CVPR_2020_paper.pdf)

- Cityscapes 3D: Dataset and Benchmark for 9 DoF Vehicle Detection [(CVPRW 20)](https://arxiv.org/pdf/2006.07864.pdf)

- The ApolloScape Open Dataset for Autonomous Driving and its Application [(T-PAMI 20)](https://arxiv.org/pdf/1803.06184.pdf)

- EU Long-term Dataset with Multiple Sensors for Autonomous Driving [(IROS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341406)

- LIBRE: The Multiple 3D LiDAR Dataset [(IV 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9304681)

- A2D2: Audi Autonomous Driving Dataset [(arXiv 20)](https://arxiv.org/pdf/2004.06320.pdf)

- Canadian Adverse Driving Conditions Dataset [(arXiv 20)](https://arxiv.org/pdf/2001.10117.pdf)

### 2019

- TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents [(AAAI 19)](https://aaai.org/ojs/index.php/AAAI/article/view/4569/4447)

- Lyft Level 5 AV Dataset [(Website)](https://level-5.global/data/)

- The H3D Dataset for Full-Surround 3D Multi-Object Detection and Tracking in Crowded Urban Scenes [(ICRA 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793925)

- Precise Synthetic Image and LiDAR (PreSIL) Dataset for Autonomous Vehicle Perception [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8813809)

- A*3D Dataset: Towards Autonomous Driving in Challenging Environments [(arXiv 19)](https://arxiv.org/pdf/1909.07541.pdf)

### 2017 or earlier

- Vision meets Robotics: The KITTI Dataset [(IJRR 13)](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)

- Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite [(CVPR 12)](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf)

## 1.2. Evaluation Metrics

### 2021

- Revisiting 3D Object Detection From an Egocentric Perspective [(NeurIPS 21)](https://papers.nips.cc/paper/2021/file/db182d2552835bec774847e06406bfa2-Paper.pdf)

### 2020

- Learning to Evaluate Perception Models using Planner-Centric Metrics [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Philion_Learning_to_Evaluate_Perception_Models_Using_Planner-Centric_Metrics_CVPR_2020_paper.pdf)

- The efficacy of Neural Planning Metrics: A meta-analysis of PKL on nuScenes [(IROSW 20)](https://arxiv.org/pdf/2010.09350.pdf)

## 1.3. Loss Functions

### 2021

- Object DGCNN: 3D Object Detection using Dynamic Graphs [(NeurIPS 21)](https://arxiv.org/pdf/2110.06923.pdf)

- Center-based 3D Object Detection and Tracking [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf)

- Accurate 3D Object Detection using Energy-Based Models [(CVPRW 21)](https://openaccess.thecvf.com/content/CVPR2021W/WAD/papers/Gustafsson_Accurate_3D_Object_Detection_Using_Energy-Based_Models_CVPRW_2021_paper.pdf)

### 2020

- Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660069.pdf)

- Rotation-robust Intersection over Union for 3D Object Detection [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650460.pdf)

- Improving 3D Object Detection via Joint Attribute-oriented 3D Loss [(IV 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9304680)

### 2019

- IoU Loss for 2D/3D Object Detection [(3DV 19)](https://arxiv.org/pdf/1908.03851.pdf)

- Focal Loss in 3D Object Detection [(RA-L 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8624385)

# 2. Sensor-based 3D Object Detection

## 2.1. LiDAR-based 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/lidarmap.JPG)

A chronological overview of the most prestigious LiDAR-based 3D object detection methods.

### 2.1.1. Point-based 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/point.JPG?raw=true)

A general point-based detection framework contains a point-based backbone network and a prediction head. The point-based backbone consists of several blocks for point cloud
sampling and feature learning, and the prediction head directly estimates 3D bounding boxes from the candidate points.

#### 2022

- SASA: Semantics-Augmented Set Abstraction for Point-based 3D Object Detection [(AAAI 22)](https://arxiv.org/pdf/2201.01976.pdf)

#### 2021

- 3D Object Detection with Pointformer [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_3D_Object_Detection_With_Pointformer_CVPR_2021_paper.pdf)

- Relation Graph Network for 3D Object Detection in Point Clouds [(T-IP 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9234727)

- 3D-CenterNet: 3D object detection network for point clouds with center estimation priority [(PR 21)](https://www.sciencedirect.com/science/article/pii/S0031320321000716)

#### 2020

- 3DSSD: Point-based 3D Single Stage Object Detector [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_3DSSD_Point-Based_3D_Single_Stage_Object_Detector_CVPR_2020_paper.pdf)

- Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.pdf)

- Joint 3D Instance Segmentation and Object Detection for Autonomous Driving [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Joint_3D_Instance_Segmentation_and_Object_Detection_for_Autonomous_Driving_CVPR_2020_paper.pdf)

- Improving 3D Object Detection through Progressive Population Based Augmentation [(ECCV 20)](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660273.pdf)

- False Positive Removal for 3D Vehicle Detection with Penetrated Point Classifier [(ICIP 20)](https://arxiv.org/pdf/2005.13153.pdf)

#### 2019

- PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf)

- Attentional PointNet for 3D-Object Detection in Point Clouds [(CVPRW 19)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Paigwar_Attentional_PointNet_for_3D-Object_Detection_in_Point_Clouds_CVPRW_2019_paper.pdf)

- STD: Sparse-to-Dense 3D Object Detector for Point Cloud [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_STD_Sparse-to-Dense_3D_Object_Detector_for_Point_Cloud_ICCV_2019_paper.pdf)

- StarNet: Targeted Computation for Object Detection in Point Clouds [(arXiv 19)](https://arxiv.org/pdf/1908.11069.pdf)

- PointRGCN: Graph Convolution Networks for 3D Vehicles Detection Refinement [(arXiv 19)](https://arxiv.org/pdf/1911.12236.pdf)

#### 2018

- IPOD: Intensive Point-based Object Detector for Point Cloud [(arXiv 18)](https://arxiv.org/pdf/1812.05276.pdf)

### 2.1.2. Grid-based 3D Object Detection (Voxel and Pillars)

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/grid.JPG?raw=true)

The grid-based approaches rasterize point cloud into
3 grid representations: voxels, pillars, and bird’s-eye view (BEV) feature maps. 2D convolutional neural networks or 3D
sparse neural networks are applied on grids for feature extraction. 3D objects are finally predicted from BEV grid cells.

#### 2021

- Object DGCNN: 3D Object Detection using Dynamic Graphs [(NeurIPS 21)](https://arxiv.org/pdf/2110.06923.pdf)

- Center-based 3D Object Detection and Tracking [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf)

- Voxel Transformer for 3D Object Detection [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Mao_Voxel_Transformer_for_3D_Object_Detection_ICCV_2021_paper.pdf)

- LiDAR-Aug: A General Rendering-based Augmentation Framework for 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_LiDAR-Aug_A_General_Rendering-Based_Augmentation_Framework_for_3D_Object_Detection_CVPR_2021_paper.pdf)

- RAD: Realtime and Accurate 3D Object Detection on Embedded Systems [(CVPRW 21)](https://openaccess.thecvf.com/content/CVPR2021W/WAD/papers/Aghdam_RAD_Realtime_and_Accurate_3D_Object_Detection_on_Embedded_Systems_CVPRW_2021_paper.pdf)

- AGO-Net: Association-Guided 3D Point Cloud Object Detection Network [(T-PAMI 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9511841)

- CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud [(AAAI 21)](https://arxiv.org/pdf/2012.03015.pdf)

- Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection [(AAAI 21)](https://www.aaai.org/AAAI21Papers/AAAI-3337.DengJ.pdf)

- Anchor-free 3D Single Stage Detector with Mask-Guided Attention for Point Cloud [(ACM MM 21)](https://dl.acm.org/doi/pdf/10.1145/3474085.3475208)

- Integration of Coordinate and Geometric Surface Normal for 3D Point Cloud Object Detection [(IJCNN 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9534281)

- PSANet: Pyramid Splitting and Aggregation Network for 3D Object Detection in Point Cloud [(Sensors 21)](https://www.mdpi.com/1424-8220/21/1/136/pdf)

#### 2020

- Every View Counts: Cross-View Consistency in 3D Object Detection with Hybrid-Cylindrical-Spherical Voxelization [(NeurIPS 20)](https://drive.google.com/file/d/1oXLz0SwJVn7HM85g2LUiJh6ydvvnxMqS/view)

- HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_HVNet_Hybrid_Voxel_Network_for_LiDAR_Based_3D_Object_Detection_CVPR_2020_paper.pdf)

- Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Du_Associate-3Ddet_Perceptual-to-Conceptual_Association_for_3D_Point_Cloud_Object_Detection_CVPR_2020_paper.pdf)

- DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Najibi_DOPS_Learning_to_Detect_3D_Objects_and_Predict_Their_3D_CVPR_2020_paper.pdf)

- Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660069.pdf)

- SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700579.pdf)

- Pillar-based Object Detection for Autonomous Driving [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670018.pdf)

- From Points to Parts: 3D Object Detection From Point Cloud With Part-Aware and Part-Aggregation Network [(T-PAMI 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9018080)

- Reconfigurable Voxels: A New Representation for LiDAR-Based Point Clouds [(CoRL 20)](https://arxiv.org/pdf/2004.02724.pdf)

- SegVoxelNet: Exploring Semantic Context and Depth-aware Features for 3D Vehicle Detection from Point Cloud [(ICRA 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9196556)

- TANet: Robust 3D Object Detection from Point Clouds with Triple Attention [(AAAI 20)](https://ojs.aaai.org/index.php/AAAI/article/view/6837/6691)

- SARPNET: Shape attention regional proposal network for liDAR-based 3D object detection [(NeuroComputing 20)](https://www.sciencedirect.com/science/article/pii/S0925231219313827)

- Voxel-FPN: Multi-Scale Voxel Feature Aggregation for 3D Object Detection from LIDAR Point Clouds [(Sensors 20)](https://www.mdpi.com/1424-8220/20/3/704/pdf)

- BirdNet+: End-to-End 3D Object Detection in LiDAR Bird’s Eye View [(ITSC 20)](https://arxiv.org/pdf/2003.04188.pdf)

- 1st Place Solution for Waymo Open Dataset Challenge - 3D Detection and Domain Adaptation [(arXiv 20)](https://arxiv.org/pdf/2006.15505.pdf)

- AFDet: Anchor Free One Stage 3D Object Detection [(arXiv 20)](https://arxiv.org/pdf/2006.12671.pdf)

#### 2019

- PointPillars: Fast Encoders for Object Detection from Point Clouds [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)

- End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds [(CoRL 19)](http://proceedings.mlr.press/v100/zhou20a/zhou20a.pdf)

- IoU Loss for 2D/3D Object Detection [(3DV 19)](https://arxiv.org/pdf/1908.03851.pdf)

- Accurate and Real-time Object Detection based on Bird’s Eye View on 3D Point Clouds [(3DV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8885850)

- Focal Loss in 3D Object Detection [(RA-L 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8624385)

- 3D-GIoU: 3D Generalized Intersection over Union for Object Detection in Point Cloud [(Sensors 19)](https://www.mdpi.com/1424-8220/19/19/4093/pdf)

- FVNet: 3D Front-View Proposal Generation for Real-Time Object Detection from Point Clouds [(CISP 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8965844)

- Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection [(arXiv 19)](https://arxiv.org/pdf/1908.09492.pdf)

- Patch Refinement - Localized 3D Object Detection [(arXiv 19)](https://arxiv.org/pdf/1910.04093.pdf)

#### 2018

- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf)

- PIXOR: Real-time 3D Object Detection from Point Clouds [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf)

- SECOND: Sparsely Embedded Convolutional Detection [(Sensors 18)](https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf)

- RT3D: Real-Time 3-D Vehicle Detection in LiDAR Point Cloud for Autonomous Driving [(RA-L 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8403277)

- BirdNet: a 3D Object Detection Framework from LiDAR Information [(ITSC 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8569311)

- YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud [(ECCVW 18)](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Ali_YOLO3D_End-to-end_real-time_3D_Oriented_Object_Bounding_Box_Detection_from_ECCVW_2018_paper.pdf)

- Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection on Point Clouds [(ECCVW 28)](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11129/Simony_Complex-YOLO_An_Euler-Region-Proposal_for_Real-time_3D_Object_Detection_on_Point_ECCVW_2018_paper.pdf)

#### 2017 or earlier

- 3D Fully Convolutional Network for Vehicle Detection in Point Cloud [(IROS 17)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8205955)

- Vote3Deep: Fast Object Detection in 3D Point Clouds Using Efficient
Convolutional Neural Networks [(ICRA 17)](http://www.cvlibs.net/projects/autonomous_vision_survey/literature/Engelcke2016ARXIV.pdf)

- Vehicle Detection from 3D Lidar Using Fully Convolutional Network [(RSS 16)](https://arxiv.org/pdf/1608.07916.pdf)

- Voting for Voting in Online Point Cloud Object Detection [(RSS 15)](http://roboticsproceedings.org/rss11/p35.pdf)

### 2.1.3. Point-voxel based 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/pv.JPG?raw=true)

Single-stage point-voxel detection framework fuses
point and voxel features in the backbone network. Two-stage point-voxel detection framework first generates 3D object
proposals with a voxel-based 3D detector, and then refines these proposals using keypoints sampled from point cloud. 

#### 2022

- Behind the Curtain: Learning Occluded Shapes for 3D Object Detection [(AAAI 22)](https://arxiv.org/pdf/2112.02205.pdf)

#### 2021

- LiDAR R-CNN: An Efficient and Universal 3D Object Detector [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_LiDAR_R-CNN_An_Efficient_and_Universal_3D_Object_Detector_CVPR_2021_paper.pdf)

- PVGNet: A Bottom-Up One-Stage 3D Object Detector with Integrated Multi-Level Features [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Miao_PVGNet_A_Bottom-Up_One-Stage_3D_Object_Detector_With_Integrated_Multi-Level_CVPR_2021_paper.pdf)

- HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Noh_HVPR_Hybrid_Voxel-Point_Representation_for_Single-Stage_3D_Object_Detection_CVPR_2021_paper.pdf)

- Pyramid R-CNN: Towards Better Performance and Adaptability for 3D Object Detection [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Mao_Pyramid_R-CNN_Towards_Better_Performance_and_Adaptability_for_3D_Object_ICCV_2021_paper.pdf)

- Improving 3D Object Detection with Channel-wise Transformer [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Sheng_Improving_3D_Object_Detection_With_Channel-Wise_Transformer_ICCV_2021_paper.pdf)

- SA-Det3D: Self-Attention Based Context-Aware 3D Object Detection [(ICCVW 21)](https://arxiv.org/pdf/2101.02672.pdf)

- From Voxel to Point: IoU-guided 3D Object Detection for Point Cloud with Voxel-to-Point Decoder [(ACM MM 21)](https://dl.acm.org/doi/pdf/10.1145/3474085.3475314)

- RV-FuseNet: Range View Based Fusion of Time-Series LiDAR Data for Joint 3D Object Detection and Motion Forecasting [(IROS 21)](https://arxiv.org/pdf/2005.10863.pdf)

- Pattern-Aware Data Augmentation for LiDAR 3D Object Detection [(ITSC 21)](https://arxiv.org/pdf/2112.00050.pdf)

- From Multi-View to Hollow-3D: Hallucinated Hollow-3D R-CNN for 3D Object Detection [(T-CSVT 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9500203)

- Pseudo-Image and Sparse Points: Vehicle Detection With 2D LiDAR Revisited by Deep Learning-Based Methods [(T-ITS 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9152088)

- Dual-Branch CNNs for Vehicle Detection and Tracking on LiDAR Data [(T-ITS 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9142426)

- Improved Point-Voxel Region Convolutional Neural Network: 3D Object Detectors for Autonomous Driving [(T-ITS 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9440849)

- DSP-Net: Dense-to-Sparse Proposal Generation Approach for 3D Object Detection on Point Cloud [(IJCNN 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9534412)

- P2V-RCNN: Point to Voxel Feature Learning for 3D Object Detection From Point Clouds [(IEEE Access 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9474438)

- PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2102.00463.pdf)

- M3DeTR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers [(arXiv 21)](https://arxiv.org/pdf/2104.11896.pdf)

#### 2020

- PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.pdf)

- Structure Aware Single-stage 3D Object Detection from Point Cloud [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf)

- Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution [(ECCV 20)](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730681.pdf)

- InfoFocus: 3D Object Detection for Autonomous Driving with Dynamic Information Modeling [(ECCV 20)](https://arxiv.org/pdf/2007.08556.pdf)

- SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds [(arXiv 20)](https://arxiv.org/pdf/2006.04043.pdf)

#### 2019

- Point-Voxel CNN for Efficient 3D Deep Learning [(NeurIPS 19)](https://proceedings.neurips.cc/paper/2019/file/5737034557ef5b8c02c0e46513b98f90-Paper.pdf)

- Fast Point R-CNN [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Fast_Point_R-CNN_ICCV_2019_paper.pdf)

#### 2018

- LMNet: Real-time Multiclass Object Detection on CPU Using 3D LiDAR [(ACIRS 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8467245)

### 2.1.4. Range-based 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/range.JPG?raw=true)

The first category of range-based approaches directly predicts
3D objects from pixels in range images, with standard 2D
convolutions, or specialized convolutional/graph operators
for feature extraction. The second category transforms features
from range view into bird’s-eye view or point-view,
and then detects 3D objects from the transformed view.

#### 2021

- RSN: Range Sparse Net for Efficient, Accurate LiDAR 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_RSN_Range_Sparse_Net_for_Efficient_Accurate_LiDAR_3D_Object_CVPR_2021_paper.pdf)

- RangeIoUDet: Range Image based Real-Time 3D Object Detector Optimized by Intersection over Union [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Liang_RangeIoUDet_Range_Image_Based_Real-Time_3D_Object_Detector_Optimized_by_CVPR_2021_paper.pdf)

- To the Point: Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Chai_To_the_Point_Efficient_3D_Object_Detection_in_the_Range_CVPR_2021_paper.pdf)

- RangeDet: In Defense of Range View for LiDAR-based 3D Object Detection [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_RangeDet_In_Defense_of_Range_View_for_LiDAR-Based_3D_Object_ICCV_2021_paper.pdf)

- It’s All Around You: Range-Guided Cylindrical Network for 3D Object Detection [(ICCVW 21)](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Rapoport-Lavie_Its_All_Around_You_Range-Guided_Cylindrical_Network_for_3D_Object_ICCVW_2021_paper.pdf)

- LaserFlow: Efficient and Probabilistic Object Detection and Motion Forecasting [(RA-L 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9310205)

#### 2020

- Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection [(arXiv 20)](https://arxiv.org/pdf/2005.09927.pdf)

- RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation [(arXiv 20)](https://arxiv.org/pdf/2009.00206.pdf)

#### 2019

- LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.pdf)


### 2.1.5. Anchor-based 3D object detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/anchor.JPG?raw=true)

3D anchor boxes are placed at each BEV grid cell. Those anchors
that have high IoUs with ground truths are selected as
positives. The sizes and centers of 3D objects are regressed
from the positive anchors, and the objects’ heading angles
are predicted by bin-based classification and regression.

### 2.1.6. Anchor-free 3D object detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/anchorfree.JPG?raw=true)

The anchor-free
learning targets can be assigned to diverse views, including
the bird’s-eye view, point view, and range view. Object
parameters are predicted directly from the positive samples. 

## 2.2 Camera-based 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/cameramap.JPG?raw=true)

A chronological overview of the camera-based 3D object detection methods.

### 2.2.1. Monocular-based 3D Object Detection

- <b> Image only </b>

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/imageonly.JPG?raw=true)

Single-stage anchor-based approaches
predict 3D object parameters leveraging both image features
and predefined 3D anchor boxes. Single-stage anchor-free
methods directly predict 3D object parameters from image
pixels. Two-stage approaches first generate 2D bounding
boxes from a 2D detector, and then lift up 2D detection to
the 3D space by predicting 3D object parameters from the
2D RoI features.

- <b> Depth-assisted </b> 

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/depth.JPG?raw=true)

Depth-image based approaches obtain
depth-aware image features by fusing information from both the RGB image and the depth image. Pseudo-LiDAR based
methods first transform the depth image into a 3D pseudo point cloud, and then apply LiDAR-based 3D detector on the
point cloud to detect 3D objects. Patch-based approaches transform the depth image into a 2D coordinate map, and then
apply a 2D neural network on the coordinate map for detection.

- <b> Prior-guided monocular </b>

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/prior.JPG?raw=true)

Prior-guided approaches leverage object shape priors, geometric priors, segmentation and temporal constrains to help detect 3D objects.

#### 2022

- MonoDistill: Learning Spatial Features for Monocular 3D Object Detection [(ICLR 22)](https://openreview.net/pdf?id=C54V-xTWfi)

- Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection [(AAAI 22)](https://arxiv.org/pdf/2112.04628.pdf)

- ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection [(WACV 22)](https://arxiv.org/pdf/2106.01178.pdf)

#### 2021

- Progressive Coordinate Transforms for Monocular 3D Object Detection [(NeurIPS 21)](https://papers.nips.cc/paper/2021/file/6f3ef77ac0e3619e98159e9b6febf557-Paper.pdf)

lidar, progressive refine.

- Delving into Localization Errors for Monocular 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Delving_Into_Localization_Errors_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf)

image, error analysis.

- Depth-conditioned Dynamic Message Propagation for Monocular 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Depth-Conditioned_Dynamic_Message_Propagation_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf)

image+depth, depth-conditioned graph.

- Monocular 3D Object Detection: An Extrinsic Parameter Free Approach [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Monocular_3D_Object_Detection_An_Extrinsic_Parameter_Free_Approach_CVPR_2021_paper.pdf)

image, extrinsic free.

- MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation [(CVPR 21)](https://arxiv.org/pdf/2103.12605.pdf)

lidar, 2d det, shape reconstruction.

- GrooMeD-NMS: Grouped Mathematically Differentiable NMS for Monocular 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Kumar_GrooMeD-NMS_Grouped_Mathematically_Differentiable_NMS_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf)

image, nms.

- Categorical Depth Distribution Network for Monocular 3D Object Detection [(CVPR 21)](https://arxiv.org/pdf/2103.01100.pdf)

lidar, catrgorical depth, frustum+voxel.

- Objects are Different: Flexible Monocular 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Objects_Are_Different_Flexible_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf)

image+depth, edge fusion.

- M3DSSD: Monocular 3D Single Stage Object Detector [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_M3DSSD_Monocular_3D_Single_Stage_Object_Detector_CVPR_2021_paper.pdf)

image, anchor, attention.

- Exploring Intermediate Representation for Monocular Vehicle Pose Estimation [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Exploring_intermediate_representation_for_monocular_vehicle_pose_estimation_CVPR_2021_paper.pdf)

pose only.

- AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_AutoShape_Real-Time_Shape-Aware_Monocular_3D_Object_Detection_ICCV_2021_paper.pdf)

image, shape constraint.

- Is Pseudo-Lidar needed for Monocular 3D Object detection? [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Is_Pseudo-Lidar_Needed_for_Monocular_3D_Object_Detection_ICCV_2021_paper.pdf)

image+depth, end-to-end

- The Devil is in the Task: Exploiting Reciprocal Appearance-Localization Features for Monocular 3D Object Detection [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Zou_The_Devil_Is_in_the_Task_Exploiting_Reciprocal_Appearance-Localization_Features_ICCV_2021_paper.pdf)

image+depth, feature disentangle.

- FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection [(ICCVW 21)](https://openaccess.thecvf.com/content/ICCV2021W/3DODI/papers/Wang_FCOS3D_Fully_Convolutional_One-Stage_Monocular_3D_Object_Detection_ICCVW_2021_paper.pdf)

image, end-to-end, 2d boxes, centerness.

- MonoCInIS: Camera Independent Monocular 3D Object Detection using Instance Segmentation [(ICCVW 21)](https://openaccess.thecvf.com/content/ICCV2021W/3DODI/papers/Heylen_MonoCInIS_Camera_Independent_Monocular_3D_Object_Detection_Using_Instance_Segmentation_ICCVW_2021_paper.pdf)

image, camera-independent, instance segmentation.

- Probabilistic and Geometric Depth: Detecting Objects in Perspective [(CoRL 21)](https://openreview.net/pdf?id=bEito8UUUmf)

image+depth, end-to-end, depth estimation with probability and geo-relation.

- Monocular 3D Detection With Geometric Constraint Embedding and Semi-Supervised Training [(RA-L 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9361326)

image, geo-reasoning.

- Ground-aware Monocular 3D Object Detection for Autonomous Driving [(RA-L 21)](https://arxiv.org/pdf/2102.00690.pdf)

image+depth, 3D anchors, ground estimation.

- Neighbor-Vote: Improving Monocular 3D Object Detection through Neighbor Distance Voting [(ACM MM 21)](https://dl.acm.org/doi/pdf/10.1145/3474085.3475641)

- Point-Guided Contrastive Learning for Monocular 3-D Object Detection [(T-Cybernetics 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9525165)

- Lidar Point Cloud Guided Monocular 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2104.09035.pdf)

- OCM3D: Object-Centric Monocular 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2104.06041.pdf)

lidar, voxel car.

#### 2020

- MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_MonoPair_Monocular_3D_Object_Detection_Using_Pairwise_Spatial_Relationships_CVPR_2020_paper.pdf)

image+depth, objects relations.

- Autolabeling 3D Objects with Differentiable Rendering of SDF Shape Priors [(CVPR 20)](https://arxiv.org/pdf/1911.11288.pdf)

image, SDF shape estimation.

- Learning Depth-Guided Convolutions for Monocular 3D Object Detection [(CVPRW 20)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w60/Ding_Learning_Depth-Guided_Convolutions_for_Monocular_3D_Object_Detection_CVPRW_2020_paper.pdf)

image+depth, depth-guided conv.

- SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation [(CVPRW 20)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w60/Liu_SMOKE_Single-Stage_Monocular_3D_Object_Detection_via_Keypoint_Estimation_CVPRW_2020_paper.pdf)

image, keypoints estimation.

- RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480647.pdf)

image, 9 keypoints estimation.

- Distance-Normalized Unified Representation for Monocular 3D Object Detection [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740086.pdf)

image, keypoint, center distance.

- Monocular 3D Object Detection via Feature Domain Adaptation [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540018.pdf)

lidar, real-pseudo DA.

- Monocular Differentiable Rendering for Self-Supervised 3D Object Detection [(ECCV 20)](https://arxiv.org/pdf/2009.14524.pdf)

image, differential rendering, render and compare, depth+det+seg.

- Rethinking Pseudo-LiDAR Representation [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580307.pdf)

lidar, xyz map, seg.

- Kinematic 3D Object Detection in Monocular Video [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680137.pdf)

image, video, motion, kalman filter.

- Towards Generalization Across Depth for Monocular 3D Object Detection [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670766.pdf)

image, categorical distance.

- Monocular 3D Object Detection with Decoupled Structured Polygon Estimation and Height-Guided Depth Estimation [(AAAI 20)](https://ojs.aaai.org/index.php/AAAI/article/download/6618/6472)

lidar, polygon estimation, height constraint, 3D box pool on BEV.

- Task-Aware Monocular Depth Estimation for 3D Object Detection [(AAAI 20)](https://ojs.aaai.org/index.php/AAAI/article/view/6908/6762)

lidar, foreground-background depth estimation.

- MoNet3D: Towards Accurate Monocular 3D Object Localization in Real Time [(ICML 20)](http://proceedings.mlr.press/v119/zhou20b/zhou20b.pdf)

image+depth, keypoints.

- MonoFENet: Monocular 3D Object Detection With Feature Enhancement Networks [(T-IP 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8897727)

image+depth, 2-stage, point 2nd-stage.

- Dynamic Depth Fusion and Transformation for Monocular 3D Object Detection [(ACCV 20)](https://openaccess.thecvf.com/content/ACCV2020/papers/Ouyang_Dynamic_Depth_Fusion_and_Transformation_for_Monocular_3D_Object_Detection_ACCV_2020_paper.pdf)

- IAFA: Instance-aware Feature Aggregation for 3D Object Detection from a Single Image [(ACCV 20)](https://openaccess.thecvf.com/content/ACCV2020/papers/Zhou_IAFA_Instance-Aware_Feature_Aggregation_for_3D_Object_Detection_from_a_ACCV_2020_paper.pdf)

- PerMO: Perceiving More at Once from a Single Image for Autonomous Driving [(arXiv 21)](https://arxiv.org/pdf/2007.08116.pdf)

#### 2019

- Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Pseudo-LiDAR_From_Visual_Depth_Estimation_Bridging_the_Gap_in_3D_CVPR_2019_paper.pdf)

lidar

- GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_GS3D_An_Efficient_3D_Object_Detection_Framework_for_Autonomous_Driving_CVPR_2019_paper.pdf)

image, 2d box + heading, feature extraction on surfaces.

- Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ku_Monocular_3D_Object_Detection_Leveraging_Accurate_Proposals_and_Shape_Reconstruction_CVPR_2019_paper.pdf)

image, local point cloud reconstruction, 2-stage.

- ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Manhardt_ROI-10D_Monocular_Lifting_of_2D_Detection_to_6D_Pose_and_CVPR_2019_paper.pdf)

image+depth, lifting 8 corners, fitting CAD models.

- Deep Fitting Degree Scoring Network for Monocular 3D Object Detection [(CVPR 19)](https://arxiv.org/pdf/1904.12681.pdf)

image, anchors, fitting IoU.

- Joint Monocular 3D Vehicle Detection and Tracking [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Joint_Monocular_3D_Vehicle_Detection_and_Tracking_ICCV_2019_paper.pdf)

image, video, depth-order tracking.

- M3D-RPN: Monocular 3D Region Proposal Network for Object Detection [(ICCV 19)](https://arxiv.org/pdf/1907.06038.pdf)

image, end-to-end, depth-aware convolution, anchors.

- Accurate Monocular 3D Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ma_Accurate_Monocular_3D_Object_Detection_via_Color-Embedded_3D_Reconstruction_for_ICCV_2019_paper.pdf)

lidar, rgb-aug, seg.

- Deep Optics for Monocular Depth Estimation and 3D Object Detection [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chang_Deep_Optics_for_Monocular_Depth_Estimation_and_3D_Object_Detection_ICCV_2019_paper.pdf)

lidar, sensor depth.

- MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bertoni_MonoLoco_Monocular_3D_Pedestrian_Localization_and_Uncertainty_Estimation_ICCV_2019_paper.pdf)

image, pedestrian, uncertainty.

- Disentangling Monocular 3D Object Detection [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Simonelli_Disentangling_Monocular_3D_Object_Detection_ICCV_2019_paper.pdf)

image, disentangle 2D and 3D det. (paradigm)

- Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud [(ICCVW 19)](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Weng_Monocular_3D_Object_Detection_with_Pseudo-LiDAR_Point_Cloud_ICCVW_2019_paper.pdf)

lidar, instance mask, 2d-3d box consistency constraint.

- Mono3D++: Monocular 3D Vehicle Detection with Two-Scale 3D Hypotheses and Task Priors [(AAAI 19)](https://ojs.aaai.org/index.php/AAAI/article/view/4856/4729)

image+depth, morphable wireframe model.

- MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization [(AAAI 19)](https://arxiv.org/pdf/1811.10247.pdf)

image+depth, instance depth, delta, multi-stage.

- Orthographic Feature Transform for Monocular 3D Object Detection [(BMVC 19)](https://bmvc2019.org/wp-content/uploads/papers/0328-paper.pdf)

image, feature transformed to BEV.

- Shift R-CNN: Deep Monocular 3D Object Detection with Closed-Form Geometric Constraints [(ICIP 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8803397)

image, optimize distance to max iou.

- Beyond Bounding Boxes: Using Bounding Shapes for Real-Time 3D Vehicle Detection from Monocular RGB Images [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8814036)

- Deep Learning based Vehicle Position and Orientation Estimation via Inverse Perspective Mapping Image [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8814050)

- Objects as Points [(arXiv 19)](https://arxiv.org/pdf/1904.07850.pdf)

image, centernet.

- Monocular 3D Object Detection and Box Fitting Trained End-to-End Using Intersection-over-Union Loss [(arXiv 19)](https://arxiv.org/pdf/1906.08070.pdf)

image, box fitting with optimization.

- Monocular 3D Object Detection via Geometric Reasoning on Keypoints [(arXiv 19)](https://arxiv.org/pdf/1905.05618.pdf)

- RefinedMPL: Refined Monocular PseudoLiDAR for 3D Object Detection in Autonomous Driving [(arXiv 19)](https://arxiv.org/pdf/1911.09712.pdf)

- Learning 2D to 3D Lifting for Object Detection in 3D for Autonomous Vehicles [(arXiv 19)](https://arxiv.org/pdf/1904.08494.pdf)

#### 2018

- Multi-Level Fusion based 3D Object Detection from Monocular Images [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Multi-Level_Fusion_Based_CVPR_2018_paper.pdf)

image+depth, 2-stage, point 2nd-stage.

- 3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kundu_3D-RCNN_Instance-Level_3D_CVPR_2018_paper.pdf)

image, shape reconstruction, TSDF, fitting CAD models.

- 3D Bounding Boxes for Road Vehicles: A One-Stage, Localization Prioritized Approach using Single Monocular Images [(ECCVW 18)](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Gupta_3D_Bounding_Boxes_for_Road_Vehicles_A_One-Stage_Localization_Prioritized_ECCVW_2018_paper.pdf)

- The Earth ain’t Flat: Monocular Reconstruction of Vehicles on Steep and Graded Roads from a Moving Camera [(IROS 18)](https://arxiv.org/pdf/1803.02057.pdf)

- MB-Net: MergeBoxes for Real-Time 3D Vehicles Detection [(IV 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8500395)

#### 2017 or earlier

- Deep MANTA: A Coarse-to-fine Many-Task Network for joint 2D and 3D vehicle analysis from monocular image [(CVPR 17)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chabot_Deep_MANTA_A_CVPR_2017_paper.pdf)

image, parts, shape template.

- 3D Bounding Box Estimation Using Deep Learning and Geometry [(CVPR 17)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mousavian_3D_Bounding_Box_CVPR_2017_paper.pdf)

image, basic

- Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection [(WACV 17)](http://www.cvlibs.net/projects/autonomous_vision_survey/literature/Xiang2016ARXIV.pdf)

image, sub-category.

- Monocular 3D Object Detection for Autonomous Driving [(CVPR 16)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Chen_Monocular_3D_Object_CVPR_2016_paper.pdf)

image, ground constraint, anchors, energy minimization, seg.

- Data-Driven 3D Voxel Patterns for Object Category Recognition [(CVPR 15)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiang_Data-Driven_3D_Voxel_2015_CVPR_paper.pdf)

image, seg, voxel pattern, shape reconstruction.

- Are Cars Just 3D Boxes? – Jointly Estimating the 3D Shape of Multiple Objects [(CVPR 14)](https://openaccess.thecvf.com/content_cvpr_2014/papers/Zia_Are_Cars_Just_2014_CVPR_paper.pdf)

image, wireframe 3D shape, ground.

### 2.2.2. Stereo-based 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/stereo.JPG?raw=true)

2D-detection based methods first generate a pair of 2D
proposals from the left and right image respectively, and then estimate 3D object parameters from the paired proposals.
Pseudo-LiDAR based approaches predict a disparity map by stereo matching, and then transform the disparity estimation
into depth and 3D point cloud subsequently, followed by a LiDAR-based detector for 3D detection. Volume-based methods
construct a 3D feature volume by view transform, and then a grid-based 3D object detector is applied on the 3D volume
for detection. 

#### 2022

- SIDE: Center-based Stereo 3D Detector with Structure-aware Instance Depth Estimation [(WACV 22)](https://arxiv.org/pdf/2108.09663.pdf)

#### 2021

- LIGA-Stereo: Learning LiDAR Geometry Aware Representations for Stereo-based 3D Detector [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_LIGA-Stereo_Learning_LiDAR_Geometry_Aware_Representations_for_Stereo-Based_3D_Detector_ICCV_2021_paper.pdf)

- YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection [(ICRA 21)](https://arxiv.org/pdf/2103.09422.pdf)

- PLUMENet: Efficient 3D Object Detection from Stereo Images [(IROS 21)](https://arxiv.org/pdf/2101.06594.pdf)

- Shape Prior Guided Instance Disparity Estimation for 3D Object Detection [(T-PAMI 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9419782)

#### 2020

- End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qian_End-to-End_Pseudo-LiDAR_for_Image-Based_3D_Object_Detection_CVPR_2020_paper.pdf)

- DSGN: Deep Stereo Geometry Network for 3D Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_DSGN_Deep_Stereo_Geometry_Network_for_3D_Object_Detection_CVPR_2020_paper.pdf)

- IDA-3D: Instance-Depth-Aware 3D Object Detection from Stereo Vision for Autonomous Driving [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Peng_IDA-3D_Instance-Depth-Aware_3D_Object_Detection_From_Stereo_Vision_for_Autonomous_CVPR_2020_paper.pdf)

- Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Disp_R-CNN_Stereo_3D_Object_Detection_via_Shape_Prior_Guided_CVPR_2020_paper.pdf)

- Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving [(ICLR 20)](https://openreview.net/pdf?id=BJedHRVtPB)

- ZoomNet: Part-Aware Adaptive Zooming Neural Network for 3D Object Detection [(AAAI 20)](https://arxiv.org/pdf/2003.00529.pdf)

- Object-Centric Stereo Matching for 3D Object Detection [ICRA 20](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9196660)

- Confidence Guided Stereo 3D Object Detection with Split Depth Estimation [(IROS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341188)

#### 2019

- Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Pseudo-LiDAR_From_Visual_Depth_Estimation_Bridging_the_Gap_in_3D_CVPR_2019_paper.pdf)

- Stereo R-CNN based 3D Object Detection for Autonomous Driving [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Stereo_R-CNN_Based_3D_Object_Detection_for_Autonomous_Driving_CVPR_2019_paper.pdf)

- Triangulation Learning Network: From Monocular to Stereo 3D Object Detection [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_Triangulation_Learning_Network_From_Monocular_to_Stereo_3D_Object_Detection_CVPR_2019_paper.pdf)

- Realtime 3D Object Detection for Automated Driving Using Stereo Vision and Semantic Information [(ITSC 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8917330)

#### 2017 or earlier

- 3D Object Proposals using Stereo Imagery for Accurate Object Class Detection [(T-PAMI 17)](https://arxiv.org/pdf/1608.07711.pdf)

- 3D Object Proposals for Accurate Object Class Detection [(NIPS 15)](https://proceedings.neurips.cc/paper/2015/file/6da37dd3139aa4d9aa55b8d237ec5d4a-Paper.pdf)

### 2.2.3. Multi-view-based 3D Object Detection

#### 2022

- ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection [(WACV 22)](https://arxiv.org/pdf/2106.01178.pdf)

#### 2021

- DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries [(CoRL 21)](https://arxiv.org/pdf/2110.06922.pdf)

#### 2020

- siaNMS: Non-Maximum Suppression with Siamese Networks for Multi-Camera 3D Object Detection [(IV 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9304685)

#### 2017

- 3D Object Localisation from Multi-View Image Detections [(T-PAMI 17)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7919240)

## 2.3 Multi-modal 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/fusionmap.JPG?raw=true)

A chronological overview of the most prestigious multi-modal 3D object detection methods.

### 2.3.1. LiDAR & Camera Fusion for 3D Object Detection

<b> Early Fusion </b>

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/early.JPG?raw=true)

Early-fusion approaches enhance point cloud
features with image information before they are passed through a LiDAR-based 3D object detector. In region-level
knowledge fusion, 2D detection is firstly employed on images to generate 2D bounding boxes. Then 2D boxes are
extruded into viewing frustums to select proper point cloud regions for the subsequent LiDAR-based 3D object detection.
In point-level knowledge fusion, semantic segmentation is firstly applied on images, and then the segmentation results are
transferred from the image pixels to points and used as an additional feature attached to each point. The augmented point
cloud is finally passed through a LiDAR detector for 3D object detection.

<b> Intermediate Fusion </b>

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/inter.JPG?raw=true)

Intermediate fusion approaches aim to
conduct multi-modal fusion at the intermediate steps of a 3D object detection pipeline. In backbone networks, pixel-to-point
correspondences are firstly established by camera-to-LiDAR transform, and then with the correspondences, LiDAR features
are fused with image features through diverse fusion operators. The fusion can be conducted either at the intermediate
layers or only at the output feature maps. In the proposal generation and refinement stage, 3D object proposals are first
generated and then projected into the camera and LiDAR views to crop features of different modalities. The multi-view
features are finally fused to refine the 3D object proposals for detection.

<b> Late Fusion </b>

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/late.JPG?raw=true)

Late-fusion based approaches operate on the
outputs, i.e. 3D and 2D bounding boxes, generated from a LiDAR-based 3D object detector and an image-based 2D object
detector respectively. 3D boxes and 2D boxes are combined together and fused to obtain the final detection results.

#### 2022

- AutoAlign: Pixel-Instance Feature Aggregation for Multi-Modal 3D Object Detection [(arXiv 22)](https://arxiv.org/pdf/2201.06493.pdf)

- Fast-CLOCs: Fast Camera-LiDAR Object Candidates Fusion for 3D Object Detection [(WACV 22)](https://openaccess.thecvf.com/content/WACV2022/papers/Pang_Fast-CLOCs_Fast_Camera-LiDAR_Object_Candidates_Fusion_for_3D_Object_Detection_WACV_2022_paper.pdf)

#### 2021

- Multimodal Virtual Point 3D Detection [(NeurIPS 21)](https://proceedings.neurips.cc/paper/2021/file/895daa408f494ad58006c47a30f51c1f-Paper.pdf)

- PointAugmenting: Cross-Modal Augmentation for 3D Object Detection [(CVPR 21)](https://vision.sjtu.edu.cn/files/cvpr21_pointaugmenting.pdf)

- Frustum-PointPillars: A Multi-Stage Approach for 3D Object Detection using RGB Camera and LiDAR [(ICCVW 21)](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Paigwar_Frustum-PointPillars_A_Multi-Stage_Approach_for_3D_Object_Detection_Using_RGB_ICCVW_2021_paper.pdf)

- Multi-Stage Fusion for Multi-Class 3D Lidar Detection [(ICCVW 21)](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Wang_Multi-Stage_Fusion_for_Multi-Class_3D_Lidar_Detection_ICCVW_2021_paper.pdf)

- Cross-Modality 3D Object Detection [(WACV 21)](https://openaccess.thecvf.com/content/WACV2021/papers/Zhu_Cross-Modality_3D_Object_Detection_WACV_2021_paper.pdf)

- Sparse-PointNet: See Further in Autonomous Vehicles [(RA-L 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9483647)

- FusionPainting: Multimodal Fusion with Adaptive Attention for 3D Object Detection [(ITSC 21)](https://arxiv.org/pdf/2106.12449.pdf)

- MF-Net: Meta Fusion Network for 3D object detection [(IJCNN 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9534374)

- Multi-Scale Spatial Transformer Network for LiDAR-Camera 3D Object Detection [(IJCNN 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9533588)

- Boost 3-D Object Detection via Point Clouds Segmentation and Fused 3-D GIoU-L1 Loss [(T-NNLS)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9241806)

- RangeLVDet: Boosting 3D Object Detection in LIDAR with Range Image and RGB Image [(Sensors Journal 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9612185)

- LiDAR Cluster First and Camera Inference Later: A New Perspective Towards Autonomous Driving [(arXiv 21)](https://arxiv.org/pdf/2111.09799.pdf)

- Exploring Data Augmentation for Multi-Modality 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2012.12741.pdf)

#### 2020

- PointPainting: Sequential Fusion for 3D Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.pdf)

- 3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720715.pdf)

- EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600035.pdf)

- PI-RCNN: An Efficient Multi-Sensor 3D Object Detector with Point-Based Attentive Cont-Conv Fusion Module [(AAAI 20)](https://ojs.aaai.org/index.php/AAAI/article/view/6933/6787)

- CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection [IROS 20](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341791)

- LRPD: Long Range 3D Pedestrian Detection Leveraging Specific Strengths of LiDAR and RGB [(ITSC 20)](https://arxiv.org/pdf/2006.09738.pdf)

- Fusion of 3D LIDAR and Camera Data for Object Detection in Autonomous Vehicle Applications [(Sensors Journal 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8957313)

- SemanticVoxels: Sequential Fusion for 3D Pedestrian Detection using LiDAR Point Cloud and Semantic Segmentation [(MFI 20)](https://arxiv.org/pdf/2009.12276.pdf)

#### 2019

- Multi-Task Multi-Sensor Fusion for 3D Object Detection [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Multi-Task_Multi-Sensor_Fusion_for_3D_Object_Detection_CVPR_2019_paper.pdf)

- Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds [(CVPRW 19)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Simon_Complexer-YOLO_Real-Time_3D_Object_Detection_and_Tracking_on_Semantic_Point_CVPRW_2019_paper.pdf)

- Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation [(CVPRW 19)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Meyer_Sensor_Fusion_for_Joint_3D_Object_Detection_and_Semantic_Segmentation_CVPRW_2019_paper.pdf)

- MVX-Net: Multimodal VoxelNet for 3D Object Detection [(ICRA 19)](https://arxiv.org/pdf/1904.01649.pdf)

- SEG-VoxelNet for 3D Vehicle Detection from RGB and LiDAR Data [(ICRA 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793492)

- 3D Object Detection Using Scale Invariant and Feature Reweighting Networks [(AAAI 19)](https://ojs.aaai.org/index.php/AAAI/article/download/4963/4836)

- Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection [(IROS 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8968513)

- Deep End-to-end 3D Person Detection from Camera and Lidar [(ITSC 19)](http://pure.tudelft.nl/ws/portalfiles/portal/68940754/roth2019itsc_lidar_person_detection.pdf)

- RoarNet: A Robust 3D Object Detection based on RegiOn Approximation Refinement [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8813895)

- SCANet: Spatial-channel attention network for 3D object detection [(ICASSP 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682746)

- One-Stage Multi-Sensor Data Fusion Convolutional Neural Network for 3D Object Detection [(Sensors 19)](https://www.mdpi.com/1424-8220/19/6/1434/pdf)

#### 2018

- Frustum PointNets for 3D Object Detection from RGB-D Data [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)

- PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_PointFusion_Deep_Sensor_CVPR_2018_paper.pdf)

- Deep Continuous Fusion for Multi-Sensor 3D Object Detection [(ECCV 18)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf)

- Joint 3D Proposal Generation and Object Detection from View Aggregation [(IROS 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8594049)

- A General Pipeline for 3D Detection of Vehicles [(ICRA 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8461232)

- Fusing Bird’s Eye View LIDAR Point Cloud and Front View Camera Image for 3D Object Detection [(IV 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8500387)

- Robust Camera Lidar Sensor Fusion Via Deep Gated Information Fusion Network [(IV 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8500711)

#### 2017 or earlier

- Multi-View 3D Object Detection Network for Autonomous Driving [(CVPR 17)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf)

### 2.3.2. LiDAR & Other Sensors Fusion for 3D Object Detection

#### 2021

- Robust Multimodal Vehicle Detection in Foggy Weather Using Complementary Lidar and Radar Signals [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_Robust_Multimodal_Vehicle_Detection_in_Foggy_Weather_Using_Complementary_Lidar_CVPR_2021_paper.pdf)

- CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection [(WACV 21)](https://openaccess.thecvf.com/content/WACV2021/papers/Nabati_CenterFusion_Center-Based_Radar_and_Camera_Fusion_for_3D_Object_Detection_WACV_2021_paper.pdf)

- Graph Convolutional Networks for 3D Object Detection on Radar Data [(ICCVW 21)](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Meyer_Graph_Convolutional_Networks_for_3D_Object_Detection_on_Radar_Data_ICCVW_2021_paper.pdf)

- 3D for Free: Crossmodal Transfer Learning using HD Maps [(arXiv 21)](https://arxiv.org/pdf/2008.10592.pdf)

- MapFusion: A General Framework for 3D Object Detection with HDMaps [(arXiv 21)](https://arxiv.org/pdf/2103.05929.pdf)

- Monocular 3D Vehicle Detection Using Uncalibrated Traffic Cameras through Homography [(arXiv 21)](https://arxiv.org/pdf/2103.15293.pdf)

#### 2020

- What You See is What You Get: Exploiting Visibility for 3D Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.pdf)

- RadarNet: Exploiting Radar for Robust Perception of Dynamic Objects [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630477.pdf)

- High Dimensional Frustum PointNet for 3D Object Detection from Camera, LiDAR, and Radar [(IV 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9304655)

- Radar-Camera Sensor Fusion for Joint Object Detection and Distance Estimation in Autonomous Vehicles [(IROSW 20)](https://arxiv.org/pdf/2009.08428.pdf)

#### 2019

- Vehicle Detection With Automotive Radar Using Deep Learning on Range-Azimuth-Doppler Tensors [(ICCVW 19)](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Major_Vehicle_Detection_With_Automotive_Radar_Using_Deep_Learning_on_Range-Azimuth-Doppler_ICCVW_2019_paper.pdf)

#### 2018

- HDNET: Exploiting HD Maps for 3D Object Detection [(CoRL 18)](http://proceedings.mlr.press/v87/yang18b/yang18b.pdf)

- Sensors and Sensor Fusion in Autonomous Vehicles [(TELFOR 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8612054)

#### 2017 or earlier

- Deep Learning Based 3D Object Detection for Automotive Radar and Camera [(ERC 16)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8904867)

# 3. Temporal 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/temporalmap.JPG?raw=true)

A chronological overview of the most prestigious temporal 3D object detection methods.

<b> 3D object detection from LiDAR sequences </b>

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/lidarseq.JPG?raw=true)

In temporal 3D object detection from LiDAR sequences, diverse temporal aggregation modules
are employed to fuse features and object proposals from
multi-frame point clouds.

<b> 3D object detection from streaming data </b>

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/stream.JPG?raw=true)

Detection from streaming data is conducted on each LiDAR
packet before the scanner produces a complete sweep. 

## 2022

- Joint 3D Object Detection and Tracking Using Spatio-Temporal Representation of Camera Image and LiDAR Point Clouds [(AAAI 22)](https://arxiv.org/pdf/2112.07116.pdf)

## 2021

- PolarStream: Streaming Lidar Object Detection and Segmentation with Polar Pillars [(NeurIPS 21)](https://arxiv.org/pdf/2106.07545.pdf)

- Offboard 3D Object Detection from Point Cloud Sequencess [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Qi_Offboard_3D_Object_Detection_From_Point_Cloud_Sequences_CVPR_2021_paper.pdf)

- 3D-MAN: 3D Multi-frame Attention Network for Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_3D-MAN_3D_Multi-Frame_Attention_Network_for_Object_Detection_CVPR_2021_paper.pdf)

- 4D-Net for Learned Multi-Modal Alignment [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Piergiovanni_4D-Net_for_Learned_Multi-Modal_Alignment_ICCV_2021_paper.pdf)

- Graph Neural Network and Spatiotemporal Transformer Attention for 3D Video Object Detection from Point Clouds [(T-PAMI 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9609569)

- LaserFlow: Efficient and Probabilistic Object Detection and Motion Forecasting [(RA-L 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9310205)

- VelocityNet: Motion-Driven Feature Aggregation for 3D Object Detection in Point Cloud Sequences [(ICRA 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561644)

- RV-FuseNet: Range View Based Fusion of Time-Series LiDAR Data for Joint 3D Object Detection and Motion Forecasting [(IROS 21)](https://arxiv.org/pdf/2005.10863.pdf)

- LiDAR-based 3D Video Object Detection with Foreground Context Modeling and Spatiotemporal Graph Reasoning [(ITSC 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9565058)

- Temporal-Channel Transformer for 3D Lidar-Based Video Object Detection for Autonomous Driving [(T-CSVT 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9438625)

- Auto4D: Learning to Label 4D Objects from Sequential Point Clouds [(arXiv 21)](https://arxiv.org/pdf/2101.06586.pdf)

## 2020

- STINet: Spatio-Temporal-Interactive Network for Pedestrian Detection and Trajectory Prediction [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_STINet_Spatio-Temporal-Interactive_Network_for_Pedestrian_Detection_and_Trajectory_Prediction_CVPR_2020_paper.pdf)

- LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_LiDAR-Based_Online_3D_Video_Object_Detection_With_Graph-Based_Message_Passing_CVPR_2020_paper.pdf)

- An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds [(ECCV 20)](https://arxiv.org/pdf/2007.12392.pdf)

- Streaming Object Detection for 3-D Point Clouds [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630409.pdf)

- Kinematic 3D Object Detection in Monocular Video [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680137.pdf)

- STROBE: Streaming Object Detection from LiDAR Packets [(CoRL 20)](https://arxiv.org/pdf/2011.06425.pdf)

- 3D Object Detection and Tracking Based on Streaming Data [(ICRA 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9197183)

- 3D Object Detection For Autonomous Driving Using Temporal Lidar Data [(ICIP 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9191134)

- Deep SCNN-Based Real-Time Object Detection for Self-Driving Vehicles Using LiDAR Temporal Data [(IEEE Access 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9078792)

## 2019

- 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks [(CVPR 19)](https://arxiv.org/pdf/1904.08755.pdf)

- Joint Monocular 3D Vehicle Detection and Tracking [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Joint_Monocular_3D_Vehicle_Detection_and_Tracking_ICCV_2019_paper.pdf)

## 2018

- Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)

- YOLO4D: A Spatio-temporal Approach for Real-time Multi-object Detection and Classification from LiDAR Point Clouds [(NIPSW 18)](https://openreview.net/pdf?id=B1xWZic29m)

# 4. Label-Efficient 3D Object Detection

## 4.1. Domain Adaptation for 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/da.JPG?raw=true)

In real-world applications, 3D object detectors suffer from severe domain gaps across different datasets, sensors, and weather conditions.

### 2021

- Learning Transferable Features for Point Cloud Detection via 3D Contrastive Co-training [(NeurIPS 21)](https://papers.nips.cc/paper/2021/file/b3b25a26a0828ea5d48d8f8aa0d6f9af-Paper.pdf)

- ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_ST3D_Self-Training_for_Unsupervised_Domain_Adaptation_on_3D_Object_Detection_CVPR_2021_paper.pdf)

- SRDAN: Scale-aware and Range-aware Domain Adaptation Network for Cross-dataset 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_SRDAN_Scale-Aware_and_Range-Aware_Domain_Adaptation_Network_for_Cross-Dataset_3D_CVPR_2021_paper.pdf)

- SPG: Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_SPG_Unsupervised_Domain_Adaptation_for_3D_Object_Detection_via_Semantic_ICCV_2021_paper.pdf)

- Unsupervised Domain Adaptive 3D Detection with Multi-Level Consistency [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Luo_Unsupervised_Domain_Adaptive_3D_Detection_With_Multi-Level_Consistency_ICCV_2021_paper.pdf)

- PIT: Position-Invariant Transform for Cross-FoV Domain Adaptation [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_PIT_Position-Invariant_Transform_for_Cross-FoV_Domain_Adaptation_ICCV_2021_paper.pdf)

- FAST3D: Flow-Aware Self-Training for 3D Object Detectors [(BMVC 21)](https://arxiv.org/pdf/2110.09355.pdf)

- What My Motion tells me about Your Pose: A Self-Supervised Monocular 3D Vehicle Detector [(ICRA 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9562086)

- Adversarial Training on Point Clouds for Sim-to-Real 3D Object Detection [(RA-L 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9468980)

- 3D for Free: Crossmodal Transfer Learning using HD Maps [(arXiv 21)](https://arxiv.org/pdf/2008.10592.pdf)

- Uncertainty-aware Mean Teacher for Source-free Unsupervised Domain Adaptive 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2109.14651.pdf)

- Exploiting Playbacks in Unsupervised Domain Adaptation for 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2103.14198.pdf)

- See Eye to Eye: A Lidar-Agnostic 3D Detection Framework for Unsupervised Multi-Target Domain Adaptation [(arXiv 21)](https://arxiv.org/pdf/2111.09450.pdf)

- Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2111.15656.pdf)

- Cycle and Semantic Consistent Adversarial Domain Adaptation for Reducing Simulation-to-Real Domain Shift in LiDAR Bird’s Eye View [(arXiv 21)](https://arxiv.org/pdf/2104.11021.pdf)

### 2020

- Train in Germany, Test in The USA: Making 3D Object Detectors Generalize [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Train_in_Germany_Test_in_the_USA_Making_3D_Object_CVPR_2020_paper.pdf)

- SF-UDA3D: Source-Free Unsupervised Domain Adaptation for LiDAR-Based 3D Object Detection [(3DV 20)](https://arxiv.org/pdf/2010.08243.pdf)

### 2019

- Transferable Semi-Supervised 3D Object Detection From RGB-D Data [(ICCV 19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tang_Transferable_Semi-Supervised_3D_Object_Detection_From_RGB-D_Data_ICCV_2019_paper.pdf)

- Range Adaptation for 3D Object Detection in LiDAR [(ICCVW 19)](https://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Wang_Range_Adaptation_for_3D_Object_Detection_in_LiDAR_ICCVW_2019_paper.pdf)

- Domain Adaptation for Vehicle Detection from Bird’s Eye View LiDAR Point Cloud Data [(ICCVW 19)](https://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Saleh_Domain_Adaptation_for_Vehicle_Detection_from_Birds_Eye_View_LiDAR_ICCVW_2019_paper.pdf)

- Cross-Sensor Deep Domain Adaptation for LiDAR Detection and Segmentation [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8814047)

## 4.2. Weakly-supervised 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/weak.JPG?raw=true)

Weakly-supervised approaches learn to
detect 3D objects with weak supervisory signals. 

### 2022

- WeakM3D: Towards Weakly Supervised Monocular 3D Object Detection [(ICLR 22)](https://openreview.net/pdf?id=ahi2XSHpAUZ)

### 2021

- Towards A Weakly Supervised Framework for 3D Point Cloud Object Detection and Annotation [(T-PAMI 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9369074)

- FGR: Frustum-Aware Geometric Reasoning for Weakly Supervised 3D Vehicle Detection [(ICRA 21)](https://arxiv.org/pdf/2105.07647.pdf)

- Open-set 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2112.01135.pdf)

- Lifting 2D Object Locations to 3D by Discounting LiDAR Outliers across Objects and Views [(arXiv 21)](https://arxiv.org/pdf/2109.07945.pdf)

### 2020

- Weakly Supervised 3D Object Detection from Lidar Point Cloud [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580511.pdf)

- Weakly Supervised 3D Object Detection from Point Clouds [(ACM MM 20)](https://dl.acm.org/doi/pdf/10.1145/3394171.3413805)

### 2019

- Deep Active Learning for Efficient Training of a LiDAR 3D Object Detector [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8814236)

- LATTE: Accelerating LiDAR Point Cloud Annotation via Sensor Fusion, One-Click Annotation, and Tracking [(ITSC 19)](https://arxiv.org/pdf/1904.09085.pdf)

### 2018

- Leveraging Pre-Trained 3D Object Detection Models For Fast Ground Truth Generation [(ITSC 18)](https://arxiv.org/pdf/1807.06072.pdf)

## 4.3. Semi-supervised 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/semi.JPG?raw=true)

Semi-supervised approaches first pretrain
a 3D detector on the labeled data, and then use the
pre-trained detector to produce pseudo labels or leverage
teacher-student models for training on the unlabeled data
to further boost the detection performance. 

### 2021

- 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_3DIoUMatch_Leveraging_IoU_Prediction_for_Semi-Supervised_3D_Object_Detection_CVPR_2021_paper.pdf)

- SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_SE-SSD_Self-Ensembling_Single-Stage_Object_Detector_From_Point_Cloud_CVPR_2021_paper.pdf)

- Semi-supervised 3D Object Detection via Adaptive Pseudo-Labeling [(ICIP 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9506421)

- Pseudo-labeling for Scalable 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2103.02093.pdf)

## 4.4. Self-supervised 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/self.JPG?raw=true)

Self-supervised approaches first pre-train
a 3D detector on the unlabeled data in a self-supervised
manner, and then fine-tune the detector on the labeled data. 

### 2022

- SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-Training for Spatial-Aware Visual Representations [(AAAI 22)](https://arxiv.org/pdf/2112.04680.pdf)

### 2021

- Exploring Geometry-aware Contrast and Clustering Harmonization for Self-supervised 3D Object Detection [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Exploring_Geometry-Aware_Contrast_and_Clustering_Harmonization_for_Self-Supervised_3D_Object_ICCV_2021_paper.pdf)

- Self-Supervised Pretraining of 3D Features on any Point-Cloud [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Self-Supervised_Pretraining_of_3D_Features_on_Any_Point-Cloud_ICCV_2021_paper.pdf)

### 2020

- PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding [(ECCV 20)](https://arxiv.org/pdf/2007.10985.pdf)

# 5. 3D Object Detection in Driving Systems

## 5.1. End-to-end Learning for Autonomous Driving

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/end.JPG?raw=true)

End-to-end autonomous driving aims to integrate all
tasks in autonomous driving, e.g. perception, prediction, planning, control, mapping, localization, into a unified framework
and learn these tasks in an end-to-end manner. 

### 2022

- Hindsight is 20/20: Leveraging Past Traversals to Aid 3D Perception [(ICLR 22)](https://openreview.net/pdf?id=qsZoGvFiJn1)

### 2021

- MP3: A Unified Model to Map, Perceive, Predict and Plan [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Casas_MP3_A_Unified_Model_To_Map_Perceive_Predict_and_Plan_CVPR_2021_paper.pdf)

- Deep Multi-Task Learning for Joint Localization, Perception, and Prediction [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Phillips_Deep_Multi-Task_Learning_for_Joint_Localization_Perception_and_Prediction_CVPR_2021_paper.pdf)

- LookOut: Diverse Multi-Future Prediction and Planning for Self-Driving [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_LookOut_Diverse_Multi-Future_Prediction_and_Planning_for_Self-Driving_ICCV_2021_paper.pdf)

- LaserFlow: Efficient and Probabilistic Object Detection and Motion Forecasting [(RA-L 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9310205)

- Perceive, Attend, and Drive: Learning Spatial Attention for Safe Self-Driving [(ICRA 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561904)

### 2020

- PnPNet: End-to-End Perception and Prediction with Tracking in the Loop [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liang_PnPNet_End-to-End_Perception_and_Prediction_With_Tracking_in_the_Loop_CVPR_2020_paper.pdf)

- MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird’s Eye View Maps [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_MotionNet_Joint_Perception_and_Motion_Prediction_for_Autonomous_Driving_Based_CVPR_2020_paper.pdf)

- STINet: Spatio-Temporal-Interactive Network for Pedestrian Detection and Trajectory Prediction [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_STINet_Spatio-Temporal-Interactive_Network_for_Pedestrian_Detection_and_Trajectory_Prediction_CVPR_2020_paper.pdf)

- DSDNet: Deep Structured self-Driving Network [(ECCV 20)](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660154.pdf)

- Testing the Safety of Self-driving Vehicles by Simulating Perception and Prediction [(ECCV 20)](https://arxiv.org/pdf/2008.06020.pdf)

- Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations [(ECCV 20)](https://arxiv.org/pdf/2008.05930.pdf)

- End-to-end Contextual Perception and Prediction with Interaction Transformer [(IROS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341392)

- PointTrackNet: An End-to-End Network For 3-D Object Detection and Tracking From Point Clouds [(RA-L 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9000527)

- Multimodal End-to-End Autonomous Driving [(T-ITS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9165167)

- Tracking to Improve Detection Quality in Lidar For Autonomous Driving [(ICASSP 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9053041)

### 2019

- Monocular Plan View Networks for Autonomous Driving [(IROS 19)](https://arxiv.org/pdf/1905.06937.pdf)

### 2018

- Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net [(CVPR 18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)

- IntentNet: Learning to Predict Intention from Raw Sensor Data [(CoRL 18)](http://www.cs.toronto.edu/~wenjie/papers/intentnet_corl18.pdf)

- End-to-end Driving via Conditional Imitation Learning [(IROS 18)](https://arxiv.org/pdf/1710.02410.pdf)

- Learning to Drive in a Day [(arXiv 18)](https://arxiv.org/pdf/1807.00412.pdf)

### 2017 or earlier

- End to End Learning for Self-Driving Cars [(arXiv 16)](https://arxiv.org/pdf/1604.07316.pdf)

## 5.2. Simulation for Autonomous Driving

### 2021

- GeoSim: Realistic Video Simulation via Geometry-Aware Composition for Self-Driving [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_GeoSim_Realistic_Video_Simulation_via_Geometry-Aware_Composition_for_Self-Driving_CVPR_2021_paper.pdf)

- TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Suo_TrafficSim_Learning_To_Simulate_Realistic_Multi-Agent_Behaviors_CVPR_2021_paper.pdf)

- AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.pdf)

- DriveGAN: Towards a Controllable High-Quality Neural Simulation [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_DriveGAN_Towards_a_Controllable_High-Quality_Neural_Simulation_CVPR_2021_paper.pdf)

- SceneGen: Learning to Generate Realistic Traffic Scenes [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Tan_SceneGen_Learning_To_Generate_Realistic_Traffic_Scenes_CVPR_2021_paper.pdf)

- LiDAR-Aug: A General Rendering-based Augmentation Framework for 3D Object Detection [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_LiDAR-Aug_A_General_Rendering-Based_Augmentation_Framework_for_3D_Object_Detection_CVPR_2021_paper.pdf)

- Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Hahner_Fog_Simulation_on_Real_LiDAR_Point_Clouds_for_3D_Object_ICCV_2021_paper.pdf)

- There and Back Again: Learning to Simulate Radar Data for Real-World Applications [(ICRA 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9562111)

- Learning to Drop Points for LiDAR Scan Synthesis [(IROS 21)](https://arxiv.org/pdf/2102.11952.pdf)

- VISTA 2.0: An Open, Data-driven Simulator for Multimodal Sensing and Policy Learning for Autonomous Vehicles [(arXiv 21)](https://arxiv.org/pdf/2111.12083.pdf)

- Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior [(arXiv 21)](https://arxiv.org/pdf/2112.05077.pdf)

### 2020

- LiDARsim: Realistic LiDAR Simulation by Leveraging the Real World [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Manivasagam_LiDARsim_Realistic_LiDAR_Simulation_by_Leveraging_the_Real_World_CVPR_2020_paper.pdf)

- SurfelGAN: Synthesizing Realistic Sensor Data for Autonomous Driving [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_SurfelGAN_Synthesizing_Realistic_Sensor_Data_for_Autonomous_Driving_CVPR_2020_paper.pdf)

- Testing the Safety of Self-driving Vehicles by Simulating Perception and Prediction [(ECCV 20)](https://arxiv.org/pdf/2008.06020.pdf)

- Learning Robust Control Policies for End-to-End Autonomous Driving from Data-Driven Simulation [(RA-L 20)](https://www.mit.edu/~amini/pubs/pdf/learning-in-simulation-vista.pdf)

- Augmented LiDAR Simulator for Autonomous Driving [(RA-L 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8972449)

### 2019

- Deep Generative Modeling of LiDAR Data [(IROS 19)](https://arxiv.org/pdf/1812.01180.pdf)

- AADS: Augmented Autonomous Driving Simulation using Data-driven Algorithms [(Science Robotics 19)](https://arxiv.org/pdf/1901.07849.pdf)

- Precise Synthetic Image and LiDAR (PreSIL) Dataset for Autonomous Vehicle Perception [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8813809)

### 2018

- Gibson Env: Real-World Perception for Embodied Agents [(CVPR 18)](https://arxiv.org/pdf/1808.10654.pdf)

- Off-Road Lidar Simulation with Data-Driven Terrain Primitives [(ICRA 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8461198)

- Interaction-Aware Probabilistic Behavior Prediction in Urban Environments [(IROS 18)](https://arxiv.org/pdf/1804.10467.pdf)

### 2017 or earlier

- CARLA: An Open Urban Driving Simulator [(CoRL 17)](https://arxiv.org/pdf/1711.03938.pdf)

- AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles [(FSR 17)](https://arxiv.org/pdf/1705.05065.pdf)

- The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes [(CVPR 16)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.pdf)

- Augmented Reality Meets Computer Vision : Efficient Data Generation for Urban Driving Scenes [(arXiv 17)](https://arxiv.org/pdf/1708.01566.pdf)

## 5.3. Reliablity & Robustness for 3D Object Detection

### 2021

- AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles [(CVPR 21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.pdf)

- Fooling LiDAR Perception via Adversarial Trajectory Perturbation [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Fooling_LiDAR_Perception_via_Adversarial_Trajectory_Perturbation_ICCV_2021_paper.pdf)

- Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather [(ICCV 21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Hahner_Fog_Simulation_on_Real_LiDAR_Point_Clouds_for_3D_Object_ICCV_2021_paper.pdf)

- Invisible for both Camera and LiDAR: Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical-World Attacks [(S&P 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9519442)

- Can We Use Arbitrary Objects to Attack LiDAR Perception in Autonomous Driving? [(CCS 21)](https://dl.acm.org/doi/pdf/10.1145/3460120.3485377)

- Exploring Adversarial Robustness of Multi-sensor Perception Systems in Self Driving [(CoRL 21)](https://arxiv.org/pdf/2101.06784.pdf)

- Lidar Light Scattering Augmentation (LISA): Physics-based Simulation of Adverse Weather Conditions for 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2107.07004.pdf)

- 3D-VField: Learning to Adversarially Deform Point Clouds for Robust 3D Object Detection [(arXiv 21)](https://arxiv.org/pdf/2112.04764.pdf)

### 2020

- Physically Realizable Adversarial Examples for LiDAR Object Detection [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tu_Physically_Realizable_Adversarial_Examples_for_LiDAR_Object_Detection_CVPR_2020_paper.pdf)

- Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bijelic_Seeing_Through_Fog_Without_Seeing_Fog_Deep_Multimodal_Sensor_Fusion_CVPR_2020_paper.pdf)

- Learning an Uncertainty-Aware Object Detector for Autonomous Driving [(IROS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341623)

- Inferring Spatial Uncertainty in Object Detection [(IROS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9340798S)

- Towards Better Performance and More Explainable Uncertainty for 3D Object Detection of Autonomous Vehicles [(ITSC 20)](https://arxiv.org/pdf/2006.12015.pdf)

- Towards Robust LiDAR-based Perception in Autonomous Driving: General Black-box Adversarial Sensor Attack and Countermeasures [(USENIX Security 20)](https://www.usenix.org/system/files/sec20-sun.pdf)

### 2019

- Robustness of 3D Deep Learning in an Adversarial Setting [(CVPR 19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wicker_Robustness_of_3D_Deep_Learning_in_an_Adversarial_Setting_CVPR_2019_paper.pdf)

- Identifying Unknown Instances for Autonomous Driving [(CoRL 19)](https://arxiv.org/pdf/1910.11296.pdf)

- Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection [(IV 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8814046)

- LiDAR Data Integrity Verification for Autonomous Vehicle [(IEEE Access 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8846700)

### 2018

- Towards Safe Autonomous Driving: Capture Uncertainty in the Deep Neural Network For Lidar 3D Vehicle Detection [(ITSC 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8569814)

## 5.4. Cooperative 3D Object Detection

![](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/3d-detection/coopr.JPG?raw=true)

In collaborative 3D object detection, different
vehicles can communicate with each other to obtain a more
reliable detection results. 

### 2021

- Learning to Communicate and Correct Pose Errors [(CoRL 21)](https://arxiv.org/pdf/2011.05289.pdf)

- Learning Distilled Collaboration Graph for Multi-Agent Perception [(NeurIPS 21)](https://openreview.net/pdf?id=ZRcjSOmYraB)

- Data Fusion with Split Covariance Intersection for Cooperative Perception [(ITSC 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9564963)

- CoFF: Cooperative Spatial Feature Fusion for 3-D Object Detection on Autonomous Vehicles [(IoT-J 21)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9330564)

- EMP: Edge-assisted Multi-vehicle Perception [(MobiCom 21)](https://dl.acm.org/doi/pdf/10.1145/3447993.3483242)

- OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication [(arXiv 21)](https://arxiv.org/pdf/2109.07644.pdf)

### 2020

- When2com: Multi-Agent Perception via Communication Graph Grouping [(CVPR 20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_When2com_Multi-Agent_Perception_via_Communication_Graph_Grouping_CVPR_2020_paper.pdf)

- V2VNet: Vehicle-to-Vehicle Communication for Joint Perception and Prediction [(ECCV 20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470596.pdf)

- Who2com: Collaborative Perception via Learnable Handshake Communication [(ICRA 20)](https://arxiv.org/pdf/2003.09575.pdf)

- MLOD: Awareness of Extrinsic Perturbation in Multi-LiDAR 3D Object Detection for Autonomous Driving [(IROS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341254)

- Cooperative Perception for 3D Object Detection in Driving Scenarios Using Infrastructure Sensors [(T-ITS 20)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9228884)

### 2019

- Cooper: Cooperative Perception for Connected Autonomous Vehicles based on 3D Point Clouds [(ICDCS 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8885377)

- F-Cooper: Feature based Cooperative Perception for Autonomous Vehicle Edge Computing System Using 3D Point Clouds [(SEC 19)](https://dl.acm.org/doi/pdf/10.1145/3318216.3363300)

- Automatic Vehicle Tracking With Roadside LiDAR Data for the Connected-Vehicles System [(IS 19)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8721124)

- Detection and tracking of pedestrians and vehicles using roadside LiDAR sensors [(Transport 19)](https://www.sciencedirect.com/science/article/pii/S0968090X19300282)

### 2018

- Collaborative Automated Driving: A Machine Learning-based Method to Enhance the Accuracy of Shared Information [(ITSC 18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8569832)

### 2017 or earlier

- Multivehicle Cooperative Driving Using Cooperative Perception: Design and Experimental Validation [(T-ITS 15)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6866903)

- Car2X-Based Perception in a High-Level Fusion Architecture for Cooperative Perception Systems [(IV 12)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6232130)

- V2V Communications in Automotive Multi-sensor Multi-target Tracking [(VTC 08)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4657272)

# 6. Continue Reading

## CVPR 2022

- Point2Seq: Detecting 3D Objects as Sequences [(CVPR 22)](https://arxiv.org/pdf/2203.13394.pdf)

- HyperDet3D: Learning a Scene-Conditioned 3D Object Detector [(CVPR 22)](https://arxiv.org/pdf/2204.05599.pdf)

- Exploring Geometry Consistency for monocular 3D object detection [(CVPR 22)](https://arxiv.org/pdf/2104.05858.pdf)

- MonoJSG: Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2203.08563.pdf)

- LiDAR Snowfall Simulation for Robust 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2203.15118.pdf)

- Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2203.08195.pdf)

- Leveraging Object-Level Rotation Equivariance for 3D Object Detection (CVPR 22)

- Rope3D: Take A New Look from the 3D Roadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task [(CVPR 22)](https://arxiv.org/pdf/2203.13608.pdf)

- Diversity Matters: Fully Exploiting Depth Clues for Reliable Monocular 3D Object Detection (CVPR 22)

- OccAM's Laser: Occlusion-based Attribution Maps for 3D Object Detectors on LiDAR Data [(CVPR 22)](https://arxiv.org/pdf/2204.06577.pdf)

- RBGNet: Ray-based Grouping for 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2204.02251.pdf)

- MonoGround: Detecting Monocular 3D Objects from the Ground [(CVPR 22)](https://arxiv.org/pdf/2102.00690.pdf)

- Voxel Field Fusion for 3D Object Detection [(CVPR 22)](https://jiaya.me/papers/cvpr22_yanwei.pdf)

- Dimension Embeddings for Monocular 3D Object Detection (CVPR 22)

- Embracing Single Stride 3D Object Detector with Sparse Transformer [(CVPR 22)](https://arxiv.org/pdf/2112.06375.pdf)

- Focal Sparse Convolutional Networks for 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2204.12463.pdf)

- TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers [(CVPR 22)](https://arxiv.org/pdf/2203.11496.pdf)

- VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention [(CVPR 22)](https://arxiv.org/pdf/2203.09704.pdf)

- MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer [(CVPR 22)](https://arxiv.org/pdf/2203.10981.pdf)

- Homography Loss for Monocular 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2204.00754.pdf)

- Point2Cyl: Reverse Engineering 3D Objects -- from Point Clouds to Extrusion Cylinders [(CVPR 22)](https://arxiv.org/pdf/2112.09329.pdf)

- SS3D: Sparsely-Supervised 3D Object Detection from Point Cloud (CVPR 22)

- LIFT: Learning 4D LiDAR Image Fusion Transformer for 3D Object Detection (CVPR 22)

- Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving [(CVPR 22)](https://arxiv.org/pdf/2203.02112.pdf)

- Bridged Transformer for Vision and Point Cloud 3D Object Detection [(CVPR 22)](https://fengxianghe.github.io/paper/wang2022bridged.pdf)

- Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection from Point Clouds [(CVPR 22)](https://arxiv.org/pdf/2203.10314.pdf)

- Boosting 3D Object Detection by Simulating Multimodality on Point Clouds (CVPR 22)

- Time3D: End-to-End Joint Monocular 3D Object Detection and Tracking for Autonomous Driving (CVPR 22)

- A Versatile Multi-View Framework for LiDAR-based 3D Object Detection with Guidance from Panoptic Segmentation [(CVPR 22)](https://arxiv.org/pdf/2203.02133.pdf)

- 3D-VField: Learning to Adversarially Deform Point Clouds for Robust 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2112.04764.pdf)

- Point Density-Aware Voxels for LiDAR 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2203.05662.pdf)

- D*-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2204.05575.pdf)

- CAT-Det: Contrastively Augmented Transformer for Multi-modal 3D Object Detection [(CVPR 22)](https://arxiv.org/pdf/2204.00325.pdf)

## 2021

- WeakM3D: Towards Weakly Supervised Monocular 3D Object Detection [(ICLR 22)](https://openreview.net/pdf?id=ahi2XSHpAUZ)

- Hindsight is 20/20: Leveraging Past Traversals to Aid 3D Perception [(ICLR 22)](https://openreview.net/pdf?id=qsZoGvFiJn1)

- MonoDistill: Learning Spatial Features for Monocular 3D Object Detection [(ICLR 22)](https://openreview.net/pdf?id=C54V-xTWfi)

- Behind the Curtain: Learning Occluded Shapes for 3D Object Detection [(AAAI 22)](https://arxiv.org/pdf/2112.02205.pdf)

- SASA: Semantics-Augmented Set Abstraction for Point-based 3D Object Detection [(AAAI 22)](https://arxiv.org/pdf/2201.01976.pdf)

- Joint 3D Object Detection and Tracking Using Spatio-Temporal Representation of Camera Image and LiDAR Point Clouds [(AAAI 22)](https://arxiv.org/pdf/2112.07116.pdf)

- SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-Training for Spatial-Aware Visual Representations [(AAAI 22)](https://arxiv.org/pdf/2112.04680.pdf)

- Embracing Single Stride 3D Object Detector with Sparse Transformer [(arXiv 21)](https://arxiv.org/pdf/2112.06375.pdf)

- AFDetV2: Rethinking the Necessity of the Second Stage for Object Detection from Point Clouds [(arXiv 21)](https://arxiv.org/pdf/2112.09205.pdfs)

- AutoAlign: Pixel-Instance Feature Aggregation for Multi-Modal 3D Object Detection [(arXiv 22)](https://arxiv.org/pdf/2201.06493.pdf)

- Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection [(AAAI 22)](https://arxiv.org/pdf/2112.04628.pdf)









