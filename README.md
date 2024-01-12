## Green AI project Introduction :leaves:
This repository stores code and project files related to the "Project Module" of THWS. The project name is "Green AI". Our group has tried to prune the YOLOv5 model and replace the backbone network to improve the reasoning efficiency of YOLOv5 and reduce the model size.

The original YOLOv5 code:[YOLOv5](https://github.com/ultralytics/yolov5/)(V6)

The dataset used for model training is the campus landscape image dataset provided by THWS. We selected 565 images from it as the training data for YOLOv5 transfer learning, which contains a total of 14 categories. The specific categories are: ['chair', 'person', 'tree', 'robot', 'door', 'fire extinguisher', 'screen', 'clock', 'podium', 'projector', 'backpack', 'laptop', ' bottle', 'window']

### 1.Replacement of the YOLOv5 backbone network
Replacing the backbone network in YOLOv5 involves substituting the default backbone with an alternative neural network architecture to potentially improve performance and efficiency.In our project, we tried to replace the backbone network with [MobileNetV2/V3](https://arxiv.org/abs/1704.04861),[GhostNet](https://arxiv.org/abs/1911.11907),[ShuffleNet](https://arxiv.org/abs/1707.01083) respectively.

Replacement of backbone network with Mobilenet
```shell
# Mobilenetv2
python train.py --imgsz 640 --epochs 100 --data ./data/all.yaml --cfg /content/models/MobileNetv2.yaml  --cache --device 0 --name MobileNetv2_backbone --optimizer AdamW
# Mobilenetv3
python train.py --imgsz 640 --epochs 100 --data ./data/all.yaml --cfg /content/models/MobileNetv3_large.yaml  --cache --device 0 --name MobileNetv3_backbone --optimizer AdamW
```

ShuffleNet
```shell
python train.py --imgsz 640 --epochs 100 --data ./data/all.yaml --cfg /content/models/ShuffleNet.yaml  --cache --device 0 --name Shuffulenet_backbone --optimizer AdamW
```

GhostNet
```shell
python train.py --imgsz 640 --epochs 100 --data ./data/all.yaml --cfg /content/models/hub/yolov5s-ghost.yaml  --cache --device 0 --name Ghostnet_backbone --optimizer AdamW
```

### 2. Model Pruning
Model pruning is a technique used in machine learning to reduce the size of a model without significantly impacting its accuracy or performance. This technique is particularly valuable in deploying models to environments with limited resources, such as mobile devices or embedded systems. The main steps involved in model pruning are as follows:
Step1:Basic training
```shell
python train.py --imgsz 640 --epochs 100 --data ./data/all.yaml --cfg ./models/yolov5s.yaml --weights ./yolov5s.pt --cache --device 0 --name mydata_adam --optimizer AdamW
```
Step2:Sparcity training
```shell
!python train.py --imgsz 640 --epochs 50 --data ./data/all.yaml  --weights ./yolov5s.pt --cache --device 0 --name sparcity_training_0.0001 --optimizer AdamW --bn_sparsity --sparsity_rate 0.0001
```
Step3:Pruning
```shell
!python prune.py --percent 0.3 --weights /content/runs/sparcity_training_0.0007/weights/best.pt --data data/all.yaml --cfg models/yolov5s.yaml --imgsz 640 --name prune_sp=0.0007_pc=0.3
```
Step4:Fine-tuning
```shell
!python train.py --img 640 --batch 32 --epochs 100 --weights /content/runs/val/prune_sp=0.0007_pc=0.3/pruned_model.pt --cache --data /content/data/all.yaml --cfg models/yolov5s.yaml --name prune_sp=0.0007_pc=0.3_ft --device 0 --optimizer AdamW --ft_pruned_model --hyp hyp.finetune_prune.yaml
```

### Experiments
- Result of THWS Dataset
  #### 1.Basic Training
  |        exp_name                   | Precision | Recall | mAP50 | parameters    | model_size | training_time&epochs | GFLOPs |   |   | note                          |
    |---------------------------|-----------|--------|-------|----------|---------------|------------|----------------------|--------|---|---|---------------------------------|
    | basic_Adam       | 0.796     | 0.705  | 7,047,883.00  | 14.5MB     | 0.311h/100           | 15.9   |   |   |                                 |
    | basic_AdamW      | 0.764     | 0.754  | 7,047,883.00  | 14.5MB     | 0.312h/100           | 15.9   |   |   |                                 |
    | basic_SGD        | 0.822     | 0.732  | 7,047,883.00  | 14.5MB     | 0.3h/100             | 15.9   |   |   |                                 |
  
  
  

  

## References
[Learning Efficient Convolutional Networks Through Network Slimming](https://arxiv.org/abs/1708.06519)
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)


## Authors：Bangguo Xu & Simei Yan & Liang Liu
