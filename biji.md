注册方式：
```
# 方式1
@BACKBONE_REGISTRY.register()
class MyBackbone():
	...
		
# 方式2
class MyBackbone():
	...
BACKBONE_REGISTRY.register(MyBackbone)
```
```mermaid
detectron2
├─checkpoint    <- checkpointer and model catalog handlers
├─config        <- default configs and handlers
├─data          <- dataset handlers and data loaders
├─engine        <- predictor and trainer engines
├─evaluation    <- evaluator for each dataset
├─export        <- converter of detectron2 models to caffe2 (ONNX)
├─layers        <- custom layers e.g. deformable conv.
├─model_zoo     <- pre-trained model links and handler
├─modeling
│ ├─meta_arch           <- meta architecture e.g. R-CNN, RetinaNet
│ ├─backbone            <- backbone network e.g. ResNet, FPN
│ ├─proposal_generator  <- region proposal network
│ └─roi_heads           <- head networks for pooled ROIs e.g. box, mask heads
├─solver        <- optimizer and scheduler builders
├─structures    <- structure classes e.g. Boxes, Instances, etc
└─utils         <- utility modules e.g. visualizer, logger, etc
```

### Meta Architecture:

GeneralizedRCNN (meta_arch/rcnn.py)

这部分包含以下模块:

#### 1. Backbone Network:
```
FPN (backbone/fpn.py)
├ ResNet (backbone/resnet.py)
│   ├ BasicStem (backbone/resnet.py)
│   └ BottleneckBlock (backbone/resnet.py)
└ LastLevelMaxPool (backbone/fpn.py
```

```
├─modeling   
│  ├─backbone
│  │    ├─backbone.py   <- includes abstract base class Backbone
│  │    ├─build.py      <- call builder function specified in config
│  │    ├─fpn.py        <- includes FPN class and sub-classes
│  │    ├─resnet.py     <- includes ResNet class and sub-classes
```

#### 2. Region Proposal Network:
```
RPN(proposal_generator/rpn.py)
├ StandardRPNHead (proposal_generator/rpn.py)
└ RPNOutput (proposal_generator/rpn_outputs.py)
```
#### 3. ROI Heads (Box Head):
```
StandardROIHeads (roi_heads/roi_heads.py)
├ ROIPooler (poolers.py)
├ FastRCNNConvFCHead (roi_heads/box_heads.py)
├ FastRCNNOutputLayers (roi_heads/fast_rcnn.py)
└ FastRCNNOutputs (roi_heads/fast_rcnn.py)
```













GROUND-TRUTH数据用于区域提议网络（RPN）和BOX-HEAD

用于目标检测的标注数据有以下部分组成：
- 框标签：对象的位置和大小（例如[X,Y,W,H]）
- 类别标签：对象类别ID（例如12："停车计时器"）

请注意，RPN不会学习对对象类别进行分类，因此类别标签仅用于ROI-HEADS。



RPN由神经网络（RPN-HEAD）和非神经网络功能组成。在DETECTRON2中，RPN中的所有计算都GPU上执行。

