> Created on Fri Jan 15 01\18\04 2021  @author: Richie Bao-caDesign设计(cadesign.cn)

## 1. 对象检测(Mask R-CNN)与人流量估算；图像分割（torchvision.models）与城市空间内容统计，及关联网络结构
深度学习在影像和视频处理方面主要涉及到图像分类（Image classification）、对象检测(Object detection)，迁移学习(Transfer learnig)，对抗生产(Adversarial generation：DCGAN/deep convolutional generative adversarial network)等。每一个方向在城市空间分析研究中都能为之提供新的分析技术方法。对象检测是计算机视觉下，分析图像或影像，将其中的对象（例如车辆、行人、动物、桌椅、植被等，通常包括室外环境，室内环境，甚至某一对象的具体再分，例如人脸识别等）标记出来。通过图像对城市环境内对象的识别，可以统计对象的空间分布情况。同时，因为无人驾驶项目大量影像数据的获取，不仅可以对城市不连续的影像分析，亦可以通过连续影像的分析进一步获取目标对象的空间分布变化情况。在下文的分析中，对于对象的分析锁定在两个方向，一个是仅识别行人，实现人流量估计；再者是尽量多的识别对象，分析城市空间下各个对象的分布变化情况，以及对象之间的联系。而图像分割(segmentation)，是将图像细分为多个图像子区域（像素的集合）的过程，每个区域通常是代表具有意义的对象，这类似于目标检测，但是是对整个图像的分割，每个区域均加以标识。

目前PyTorch已经集合了大量分析算法模型，以及提供预训练模型的参数，一定程度上可以跨过训练阶段，直接用于预测。PyTorch也提供了大量参考的代码（还有大量的著作教程），这都为研究者更快速和轻松的应用已有模型研究提供了便利，从而实现即刻的将其应用到各自的研究领域当中，而不是计算机视觉、深度学习算法本身研究的专业。

### 1.1 对象/目标检测（行人）与人流量估算
该部分代码是直接迁移应用[TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)。通常PyTorch也提供了[Googel Colab版本](https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb)，可以直接在线运行代码，解决当前没有配置GPU，或者GPU算力较低的情况。并且深度学习的快速结果反馈成为可能。

该教程微调[Penn-Fudan Database (for Pedestrian Detection and Segmentation)](https://www.cis.upenn.edu/~jshi/ped_html/)数据库中预训练的[Mask R-CN](https://arxiv.org/abs/1703.06870)，用于行人检测和分割。该数据库包含170张图像和345个行人实例。
在人流量估算研究中，直接应用'Mask R-CN'训练的模型。对该方法的解释直接参看官方内容，仅是保留关键信息的解释，而重点在于，通过对行人的目标检测，以'KITTI'数据集为例，计算行人的空间分布核密度，估算人流量。


> 参考文献
1. Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick.[Mask R-CNN](https://arxiv.org/abs/1703.06870).[Submitted on 20 Mar 2017 (v1), last revised 24 Jan 2018 (this version, v3)]. 	arXiv:1703.06870
2. [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

#### 1.1.1 对象检测（行人）
用`!`符号直接运行`conda`(作为shell命令)。有些终端命令，如果无法在JupyterLab下直接实现，则打开Anaconda的终端(terminal)执行。


```python
! pip install cython
```

    Requirement already satisfied: cython in c:\users\richi\anaconda3\envs\opencvpytorch\lib\site-packages (0.29.21)
    

可以直接在浏览器下输入`https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip`，下载，也可以在终端执行`wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip `，并且解压缩`unzip PennFudanPed.zip`，获取PennFudan数据集。其数据库文件夹结构如下：

```
PennFudanPed/
  PedMasks/
    FudanPed00001_mask.png
    FudanPed00002_mask.png
    FudanPed00003_mask.png
    FudanPed00004_mask.png
    ...
  PNGImages/
    FudanPed00001.png
    FudanPed00002.png
    FudanPed00003.png
    FudanPed00004.png
```

直接打开一张图像，和其对应的对象分割掩码。


```python
from PIL import Image
import os

PennFudanPed_fp=r'E:\dataset\PennFudan\PennFudanPed'
Image.open(os.path.join(PennFudanPed_fp,'PNGImages/PennPed00019.png'))
```




    
<a href=""><img src="./imgs/23_02.png" height='auto' width='auto' title="caDesign"></a>  
    




```python
mask=Image.open(os.path.join(PennFudanPed_fp,'PedMasks/PennPed00019_mask.png'))
mask.putpalette([
    0, 0, 0,       # black background
    255, 0, 0,     # index 1 is red
    255, 255, 0,   # index 2 is yellow
    255, 153, 0,   # index 3 is orange
])
mask
```




    
<a href=""><img src="./imgs/23_04.png" height='auto' width='auto' title="caDesign"></a>  
    



定义继承父类`torch.utils.data.Dataset`，建立数据集(tensor数据类型)作为`torch.utils.data.DataLoader`输入生成可迭代对象，用于训练模型数据加载的类。


```python
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
```

返回定义的数据集包含一个`PIL.Image`和一个字典，包含'boxes'锚框，'labels'类标，'masks'分割掩码，'image_id'图像索引，以及'area'锚框面积和'iscrowd'。


```python
dataset = PennFudanDataset(PennFudanPed_fp)
dataset[0]
```




    (<PIL.Image.Image image mode=RGB size=559x536 at 0x25AB83B4670>,
     {'boxes': tensor([[159., 181., 301., 430.],
              [419., 170., 534., 485.]]),
      'labels': tensor([1, 1]),
      'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
      
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8),
      'image_id': tensor([0]),
      'area': tensor([35358., 36225.]),
      'iscrowd': tensor([0, 0])})



`torchvision.models.detection`提供了`fasterrcnn_resnet50_fpn`，`maskrcnn_resnet50_fpn`（含对象分割/对象掩码mask）对象检测模型，包含基于[COCO数据集](https://cocodataset.org/#home)预先训练的参数。针对特定类别对其微调(finetune)。

> [COCO数据集](https://cocodataset.org/#home)，是一个大规模对象检测(object detection)、分割(segmentation)和标注(captioning)数据集，其特征有：
1. 对象分割(Object segmentation)
2. 上下文识别(Recognition in context)
3. 超像素分割(Superpixel stuff segmentation)
4. 330K张图片（>200K个标签）(330K images (>200K labeled))
5. 150万个对象实例(1.5 million object instances)
6. 80个对象类(80 object categories)
7. [91个分割类(91 stuff categories)](https://github.com/nightrome/cocostuff)
8. 每张图像5个标注(5 captions per image)
9. 25万人带有关键点(250,000 people with keypoints)

可以从COCO官网下载数据查看，对于其对象的分类，可以从`http://images.cocodataset.org/annotations/annotations_trainval2014.zip`处下载2014年，以及从`http://images.cocodataset.org/annotations/annotations_trainval2017.zip`下载2017年。解压后打开'instances_val2017.json'，打印查看对象分类。可以看到其中分类'person'，其ID为1。


```python
annotations_trainval2017_fp=r'./data/instances_val2017.json'
import json
with open(annotations_trainval2017_fp,'r') as f:
    annotations=json.loads(f.read())
    object_categories=json.dumps(annotations['categories'])
print('COCO对象分类：\n',object_categories)    
```

    COCO对象分类：
     [{"supercategory": "person", "id": 1, "name": "person"}, {"supercategory": "vehicle", "id": 2, "name": "bicycle"}, {"supercategory": "vehicle", "id": 3, "name": "car"}, {"supercategory": "vehicle", "id": 4, "name": "motorcycle"}, {"supercategory": "vehicle", "id": 5, "name": "airplane"}, {"supercategory": "vehicle", "id": 6, "name": "bus"}, {"supercategory": "vehicle", "id": 7, "name": "train"}, {"supercategory": "vehicle", "id": 8, "name": "truck"}, {"supercategory": "vehicle", "id": 9, "name": "boat"}, {"supercategory": "outdoor", "id": 10, "name": "traffic light"}, {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"}, {"supercategory": "outdoor", "id": 13, "name": "stop sign"}, {"supercategory": "outdoor", "id": 14, "name": "parking meter"}, {"supercategory": "outdoor", "id": 15, "name": "bench"}, {"supercategory": "animal", "id": 16, "name": "bird"}, {"supercategory": "animal", "id": 17, "name": "cat"}, {"supercategory": "animal", "id": 18, "name": "dog"}, {"supercategory": "animal", "id": 19, "name": "horse"}, {"supercategory": "animal", "id": 20, "name": "sheep"}, {"supercategory": "animal", "id": 21, "name": "cow"}, {"supercategory": "animal", "id": 22, "name": "elephant"}, {"supercategory": "animal", "id": 23, "name": "bear"}, {"supercategory": "animal", "id": 24, "name": "zebra"}, {"supercategory": "animal", "id": 25, "name": "giraffe"}, {"supercategory": "accessory", "id": 27, "name": "backpack"}, {"supercategory": "accessory", "id": 28, "name": "umbrella"}, {"supercategory": "accessory", "id": 31, "name": "handbag"}, {"supercategory": "accessory", "id": 32, "name": "tie"}, {"supercategory": "accessory", "id": 33, "name": "suitcase"}, {"supercategory": "sports", "id": 34, "name": "frisbee"}, {"supercategory": "sports", "id": 35, "name": "skis"}, {"supercategory": "sports", "id": 36, "name": "snowboard"}, {"supercategory": "sports", "id": 37, "name": "sports ball"}, {"supercategory": "sports", "id": 38, "name": "kite"}, {"supercategory": "sports", "id": 39, "name": "baseball bat"}, {"supercategory": "sports", "id": 40, "name": "baseball glove"}, {"supercategory": "sports", "id": 41, "name": "skateboard"}, {"supercategory": "sports", "id": 42, "name": "surfboard"}, {"supercategory": "sports", "id": 43, "name": "tennis racket"}, {"supercategory": "kitchen", "id": 44, "name": "bottle"}, {"supercategory": "kitchen", "id": 46, "name": "wine glass"}, {"supercategory": "kitchen", "id": 47, "name": "cup"}, {"supercategory": "kitchen", "id": 48, "name": "fork"}, {"supercategory": "kitchen", "id": 49, "name": "knife"}, {"supercategory": "kitchen", "id": 50, "name": "spoon"}, {"supercategory": "kitchen", "id": 51, "name": "bowl"}, {"supercategory": "food", "id": 52, "name": "banana"}, {"supercategory": "food", "id": 53, "name": "apple"}, {"supercategory": "food", "id": 54, "name": "sandwich"}, {"supercategory": "food", "id": 55, "name": "orange"}, {"supercategory": "food", "id": 56, "name": "broccoli"}, {"supercategory": "food", "id": 57, "name": "carrot"}, {"supercategory": "food", "id": 58, "name": "hot dog"}, {"supercategory": "food", "id": 59, "name": "pizza"}, {"supercategory": "food", "id": 60, "name": "donut"}, {"supercategory": "food", "id": 61, "name": "cake"}, {"supercategory": "furniture", "id": 62, "name": "chair"}, {"supercategory": "furniture", "id": 63, "name": "couch"}, {"supercategory": "furniture", "id": 64, "name": "potted plant"}, {"supercategory": "furniture", "id": 65, "name": "bed"}, {"supercategory": "furniture", "id": 67, "name": "dining table"}, {"supercategory": "furniture", "id": 70, "name": "toilet"}, {"supercategory": "electronic", "id": 72, "name": "tv"}, {"supercategory": "electronic", "id": 73, "name": "laptop"}, {"supercategory": "electronic", "id": 74, "name": "mouse"}, {"supercategory": "electronic", "id": 75, "name": "remote"}, {"supercategory": "electronic", "id": 76, "name": "keyboard"}, {"supercategory": "electronic", "id": 77, "name": "cell phone"}, {"supercategory": "appliance", "id": 78, "name": "microwave"}, {"supercategory": "appliance", "id": 79, "name": "oven"}, {"supercategory": "appliance", "id": 80, "name": "toaster"}, {"supercategory": "appliance", "id": 81, "name": "sink"}, {"supercategory": "appliance", "id": 82, "name": "refrigerator"}, {"supercategory": "indoor", "id": 84, "name": "book"}, {"supercategory": "indoor", "id": 85, "name": "clock"}, {"supercategory": "indoor", "id": 86, "name": "vase"}, {"supercategory": "indoor", "id": 87, "name": "scissors"}, {"supercategory": "indoor", "id": 88, "name": "teddy bear"}, {"supercategory": "indoor", "id": 89, "name": "hair drier"}, {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]
    

精调模型（finetuning）。


```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    # 01-load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 

    # 02-get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 03-replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) #因为仅检测行人分类，包含背景总共2类，用其替换原输入特征数in_features(1024)

    # 04-now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 05-and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    return model
```

可以下载`https://github.com/pytorch/vision.git`代码，在'references\detection'下包含有大量帮助函数，可以简化对象/目标检测训练和评估。将对应的'utils.py，transforms.py，coco_eval.py，engine.py，coco_utils.py'这5个文件，直接复制到该文件（.ipynb）所在的目录下，执行调入。定义的`get_transform`函数，可以实现对图像指定方式的变换操作，即图像增广(image augmentation)，PyTorh的[`transforms`](https://pytorch.org/docs/stable/torchvision/transforms.html)提供了大量图像增广的方法。这里使用了`RandomHorizontalFlip`实现图像的随机水平向翻转。并用`ToTensor()`方法将图像转换为tensor数据类型。


```python
from engine import train_one_epoch, evaluate  #pip install pycocotools-windows  安装关联库
import utils
import transforms as T
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
```

将建立的数据集通过`torch.utils.data.DataLoader`方法加载为可迭代对象，用于训练模型的输入数据。


```python
dataset = PennFudanDataset(PennFudanPed_fp, get_transform(train=True))
dataset_test = PennFudanDataset(PennFudanPed_fp, get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,collate_fn=utils.collate_fn)
```

定义训练模型，优化函数，学习率。指定'GPU'或者'CPU'，训练模型。


```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
model = get_instance_segmentation_model(num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad] #提取含梯度的参数
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005) #对含梯度的参数执行梯度下降法-SGD
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1) #根据epoch训练次数调整学习率的方法。PyTorch也提供了torch.optim.lr_scheduler.ReduceLROnPlateau，基于训练中某些测量值使学习率下降的方法。
```


```python
from tqdm.auto import tqdm

num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=999)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    if epoch==5 or epoch==9:
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(),'./model/mask_R_CNN_person/mask_R_CNN_person_stateDict_{}.pth'.format(epoch))   #仅保存模型的状态字典(state_dict)，state_dict由PyTorch自动生成，包含各层可训练参数（通常为卷积层、线性层），例如权值、偏置等。
torch.save(model,'./model/mask_R_CNN_person/mask_R_CNN_person_final.pth') #保存整个模型
```


      0%|          | 0/10 [00:00<?, ?it/s]


    C:\Users\richi\anaconda3\envs\pytorch\lib\site-packages\torch\nn\functional.py:3103: UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details. 
      warnings.warn("The default behavior for interpolate/upsample with float scale_factor changed "
    

    Epoch: [0]  [ 0/60]  eta: 0:02:02  lr: 0.000090  loss: 4.6407 (4.6407)  loss_classifier: 0.7970 (0.7970)  loss_box_reg: 0.4012 (0.4012)  loss_mask: 3.4269 (3.4269)  loss_objectness: 0.0099 (0.0099)  loss_rpn_box_reg: 0.0058 (0.0058)  time: 2.0483  data: 0.0279  max mem: 2302
    Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.3663 (0.8203)  loss_classifier: 0.0415 (0.1554)  loss_box_reg: 0.1495 (0.2058)  loss_mask: 0.1606 (0.4403)  loss_objectness: 0.0013 (0.0118)  loss_rpn_box_reg: 0.0046 (0.0071)  time: 0.5133  data: 0.0368  max mem: 3220
    Epoch: [0] Total time: 0:00:31 (0.5195 s / it)
    Epoch: [1]  [ 0/60]  eta: 0:00:27  lr: 0.005000  loss: 0.3059 (0.3059)  loss_classifier: 0.0526 (0.0526)  loss_box_reg: 0.1000 (0.1000)  loss_mask: 0.1516 (0.1516)  loss_objectness: 0.0004 (0.0004)  loss_rpn_box_reg: 0.0013 (0.0013)  time: 0.4638  data: 0.0259  max mem: 3220
    Epoch: [1]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.2733 (0.2851)  loss_classifier: 0.0348 (0.0410)  loss_box_reg: 0.0792 (0.0888)  loss_mask: 0.1431 (0.1481)  loss_objectness: 0.0007 (0.0017)  loss_rpn_box_reg: 0.0045 (0.0055)  time: 0.5132  data: 0.0307  max mem: 3220
    Epoch: [1] Total time: 0:00:30 (0.5093 s / it)
    Epoch: [2]  [ 0/60]  eta: 0:00:29  lr: 0.005000  loss: 0.2330 (0.2330)  loss_classifier: 0.0167 (0.0167)  loss_box_reg: 0.0697 (0.0697)  loss_mask: 0.1445 (0.1445)  loss_objectness: 0.0009 (0.0009)  loss_rpn_box_reg: 0.0012 (0.0012)  time: 0.4883  data: 0.0299  max mem: 3220
    Epoch: [2]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.2453 (0.2406)  loss_classifier: 0.0386 (0.0333)  loss_box_reg: 0.0664 (0.0630)  loss_mask: 0.1367 (0.1388)  loss_objectness: 0.0005 (0.0014)  loss_rpn_box_reg: 0.0036 (0.0041)  time: 0.5197  data: 0.0344  max mem: 3220
    Epoch: [2] Total time: 0:00:30 (0.5145 s / it)
    Epoch: [3]  [ 0/60]  eta: 0:00:38  lr: 0.000500  loss: 0.3886 (0.3886)  loss_classifier: 0.0802 (0.0802)  loss_box_reg: 0.1235 (0.1235)  loss_mask: 0.1743 (0.1743)  loss_objectness: 0.0015 (0.0015)  loss_rpn_box_reg: 0.0091 (0.0091)  time: 0.6347  data: 0.0798  max mem: 3220
    Epoch: [3]  [59/60]  eta: 0:00:00  lr: 0.000500  loss: 0.1673 (0.1996)  loss_classifier: 0.0205 (0.0280)  loss_box_reg: 0.0314 (0.0448)  loss_mask: 0.1125 (0.1226)  loss_objectness: 0.0004 (0.0010)  loss_rpn_box_reg: 0.0015 (0.0032)  time: 0.5239  data: 0.0314  max mem: 3446
    Epoch: [3] Total time: 0:00:30 (0.5116 s / it)
    Epoch: [4]  [ 0/60]  eta: 0:00:28  lr: 0.000500  loss: 0.2047 (0.2047)  loss_classifier: 0.0359 (0.0359)  loss_box_reg: 0.0277 (0.0277)  loss_mask: 0.1275 (0.1275)  loss_objectness: 0.0118 (0.0118)  loss_rpn_box_reg: 0.0019 (0.0019)  time: 0.4703  data: 0.0329  max mem: 3446
    Epoch: [4]  [59/60]  eta: 0:00:00  lr: 0.000500  loss: 0.1946 (0.1929)  loss_classifier: 0.0271 (0.0277)  loss_box_reg: 0.0429 (0.0408)  loss_mask: 0.1176 (0.1202)  loss_objectness: 0.0005 (0.0012)  loss_rpn_box_reg: 0.0022 (0.0030)  time: 0.5445  data: 0.0371  max mem: 3446
    Epoch: [4] Total time: 0:00:31 (0.5220 s / it)
    Epoch: [5]  [ 0/60]  eta: 0:00:35  lr: 0.000500  loss: 0.1728 (0.1728)  loss_classifier: 0.0262 (0.0262)  loss_box_reg: 0.0308 (0.0308)  loss_mask: 0.1099 (0.1099)  loss_objectness: 0.0007 (0.0007)  loss_rpn_box_reg: 0.0052 (0.0052)  time: 0.5939  data: 0.0359  max mem: 3446
    Epoch: [5]  [59/60]  eta: 0:00:00  lr: 0.000500  loss: 0.1690 (0.1855)  loss_classifier: 0.0179 (0.0267)  loss_box_reg: 0.0290 (0.0386)  loss_mask: 0.1009 (0.1166)  loss_objectness: 0.0005 (0.0009)  loss_rpn_box_reg: 0.0019 (0.0028)  time: 0.5195  data: 0.0360  max mem: 3446
    Epoch: [5] Total time: 0:00:31 (0.5240 s / it)
    creating index...
    index created!
    Test:  [ 0/50]  eta: 0:00:06  model_time: 0.1296 (0.1296)  evaluator_time: 0.0030 (0.0030)  time: 0.1396  data: 0.0060  max mem: 3446
    Test:  [49/50]  eta: 0:00:00  model_time: 0.1007 (0.1025)  evaluator_time: 0.0030 (0.0050)  time: 0.1211  data: 0.0128  max mem: 3446
    Test: Total time: 0:00:06 (0.1227 s / it)
    Averaged stats: model_time: 0.1007 (0.1025)  evaluator_time: 0.0030 (0.0050)
    Accumulating evaluation results...
    DONE (t=0.01s).
    Accumulating evaluation results...
    DONE (t=0.01s).
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.821
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.956
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.513
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.832
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.387
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.866
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.866
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.762
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.873
    IoU metric: segm
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.748
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.872
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.348
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.794
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.794
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.725
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.799
    Epoch: [6]  [ 0/60]  eta: 0:00:35  lr: 0.000050  loss: 0.1801 (0.1801)  loss_classifier: 0.0194 (0.0194)  loss_box_reg: 0.0382 (0.0382)  loss_mask: 0.1201 (0.1201)  loss_objectness: 0.0002 (0.0002)  loss_rpn_box_reg: 0.0023 (0.0023)  time: 0.5884  data: 0.0459  max mem: 3446
    Epoch: [6]  [59/60]  eta: 0:00:00  lr: 0.000050  loss: 0.1564 (0.1838)  loss_classifier: 0.0237 (0.0255)  loss_box_reg: 0.0288 (0.0375)  loss_mask: 0.1087 (0.1167)  loss_objectness: 0.0006 (0.0014)  loss_rpn_box_reg: 0.0019 (0.0027)  time: 0.5229  data: 0.0324  max mem: 3446
    Epoch: [6] Total time: 0:00:31 (0.5212 s / it)
    Epoch: [7]  [ 0/60]  eta: 0:00:26  lr: 0.000050  loss: 0.1492 (0.1492)  loss_classifier: 0.0220 (0.0220)  loss_box_reg: 0.0239 (0.0239)  loss_mask: 0.1020 (0.1020)  loss_objectness: 0.0001 (0.0001)  loss_rpn_box_reg: 0.0012 (0.0012)  time: 0.4399  data: 0.0359  max mem: 3446
    Epoch: [7]  [59/60]  eta: 0:00:00  lr: 0.000050  loss: 0.1591 (0.1829)  loss_classifier: 0.0222 (0.0266)  loss_box_reg: 0.0283 (0.0379)  loss_mask: 0.1110 (0.1150)  loss_objectness: 0.0004 (0.0007)  loss_rpn_box_reg: 0.0015 (0.0027)  time: 0.5125  data: 0.0309  max mem: 3446
    Epoch: [7] Total time: 0:00:31 (0.5242 s / it)
    Epoch: [8]  [ 0/60]  eta: 0:00:33  lr: 0.000050  loss: 0.2167 (0.2167)  loss_classifier: 0.0311 (0.0311)  loss_box_reg: 0.0580 (0.0580)  loss_mask: 0.1219 (0.1219)  loss_objectness: 0.0002 (0.0002)  loss_rpn_box_reg: 0.0055 (0.0055)  time: 0.5635  data: 0.0409  max mem: 3446
    Epoch: [8]  [59/60]  eta: 0:00:00  lr: 0.000050  loss: 0.1563 (0.1810)  loss_classifier: 0.0199 (0.0259)  loss_box_reg: 0.0241 (0.0363)  loss_mask: 0.1137 (0.1154)  loss_objectness: 0.0003 (0.0007)  loss_rpn_box_reg: 0.0017 (0.0026)  time: 0.5025  data: 0.0297  max mem: 3446
    Epoch: [8] Total time: 0:00:31 (0.5296 s / it)
    Epoch: [9]  [ 0/60]  eta: 0:00:37  lr: 0.000005  loss: 0.1667 (0.1667)  loss_classifier: 0.0299 (0.0299)  loss_box_reg: 0.0399 (0.0399)  loss_mask: 0.0933 (0.0933)  loss_objectness: 0.0004 (0.0004)  loss_rpn_box_reg: 0.0032 (0.0032)  time: 0.6283  data: 0.0429  max mem: 3446
    Epoch: [9]  [59/60]  eta: 0:00:00  lr: 0.000005  loss: 0.1658 (0.1828)  loss_classifier: 0.0210 (0.0263)  loss_box_reg: 0.0254 (0.0365)  loss_mask: 0.1075 (0.1162)  loss_objectness: 0.0003 (0.0011)  loss_rpn_box_reg: 0.0017 (0.0027)  time: 0.5193  data: 0.0339  max mem: 3591
    Epoch: [9] Total time: 0:00:31 (0.5325 s / it)
    creating index...
    index created!
    Test:  [ 0/50]  eta: 0:00:06  model_time: 0.1297 (0.1297)  evaluator_time: 0.0030 (0.0030)  time: 0.1396  data: 0.0060  max mem: 3591
    Test:  [49/50]  eta: 0:00:00  model_time: 0.1037 (0.1058)  evaluator_time: 0.0030 (0.0052)  time: 0.1249  data: 0.0140  max mem: 3591
    Test: Total time: 0:00:06 (0.1268 s / it)
    Averaged stats: model_time: 0.1037 (0.1058)  evaluator_time: 0.0030 (0.0052)
    Accumulating evaluation results...
    DONE (t=0.01s).
    Accumulating evaluation results...
    DONE (t=0.01s).
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.809
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.948
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.381
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.855
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.855
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.762
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.862
    IoU metric: segm
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.744
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.880
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.753
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.345
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.792
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.792
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.738
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.796
    


```python
model_= get_instance_segmentation_model(num_classes)
model_.load_state_dict(torch.load('./model/mask_R_CNN_person/mask_R_CNN_person_stateDict_9.pth'))
```




    <All keys matched successfully>



<a href="https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/"><img src="./imgs/23_01.png" height='auto' width='600' title="caDesign"></a>

Intersection over Union(IoU)交并比，用于评价对象检测、图像分割的精度，其计算公式如上图所示。由于模型的参数变化，例如图像金子塔尺度(image pyramid scale)，滑动窗口大小(sliding window size，即卷积核)、特征提取方法(feature extraction method)等，预测的边界（或锚框）与实际情况(ground-truth)完全匹配是不现实的。而IoU评估指标能很好的表述预测的精度。

除了IoU评价指标，同时提取一幅图像测试，测试结果显示能够很好的提取图像场景中的行人对象。


> 参考：[Intersection over Union (IoU) for object detection](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

* 注意将模型和数据均转化到指定的同一device下，否则提示数据类型错误。


```python
img, _ = dataset_test[34]
model_.to(device)
model_.eval()
with torch.no_grad():
    prediction = model_([img.to(device)])
```


```python
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
```




    
<a href=""><img src="./imgs/23_05.png" height='auto' width='auto' title="caDesign"></a>  
    



行人掩码包含在返回预测值的'masks'键下，其数据形状为$torch.Size([12, 1, 378, 745])$，其中12为预测的行人数量，这与实际的基本吻合（行人中往往存在互相遮掩的情况），可以用于人流量的统计。


```python
import torchvision.transforms as transforms
print('掩码形状：',prediction[0]['masks'].shape)
transforms.ToPILImage()(torch.sum(prediction[0]['masks'],dim=0)).convert("RGB") #Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
```

    掩码形状： torch.Size([12, 1, 378, 745])
    




    
<a href=""><img src="./imgs/23_06.png" height='auto' width='auto' title="caDesign"></a>  
    



#### 1.1.2 人流量估算
直接将上述训练后的模型用于无人驾驶场景[KITTI数据集](http://www.cvlibs.net/datasets/kitti/raw_data.php)，先随机提取一副含有行人的图像，用该模型预测，查看预测结果。如果基本吻合，则说明该模型可以用于进一步的人流量分析。从观察结果来看，基本能够提取出场景内的行人。


```python
img_kitti_fp=r'./data/0000000181.png'
img_kitti=Image.open(img_kitti_fp)
img_kitti
```




    
<a href=""><img src="./imgs/23_07.png" height='auto' width='auto' title="caDesign"></a>  
    




```python
import torchvision.transforms as transforms
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device:{}'.format(device))

trans_2tensor=transforms.Compose([transforms.ToTensor(),]) #将图像转换为tensor
img_kitti_tensor=trans_2tensor(img_kitti)

model_.to(device)
model_.eval()
with torch.no_grad():
    img_kitti_pred= model_([img_kitti_tensor.to(device)])
print('估计行人数量={}'.format(img_kitti_pred[0]['masks'].shape[0]))
transforms.ToPILImage()(torch.sum(img_kitti_pred[0]['masks'],dim=0)).convert("RGB")      
```

    device:cuda
    估计行人数量=17
    




    
<a href=""><img src="./imgs/23_08.png" height='auto' width='auto' title="caDesign"></a>  
    



A-提取KITTI数据集图像的位置坐标。


```python
import util
KITTI_info_fp=r'E:\dataset\KITTI\2011_09_29_drive_0071_sync\oxts\data'
timestamps_fp=r'E:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\timestamps.txt'
drive_29_0071_info=util.KITTI_info(KITTI_info_fp,timestamps_fp)
drive_29_0071_info_coordi=drive_29_0071_info[['lat','lon','timestamps_']]
util.print_html(drive_29_0071_info_coordi)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lon</th>
      <th>timestamps_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.008645</td>
      <td>8.398104</td>
      <td>2011-09-29 13:54:59.990872576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.008645</td>
      <td>8.398103</td>
      <td>2011-09-29 13:55:00.094612992</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.008646</td>
      <td>8.398102</td>
      <td>2011-09-29 13:55:00.198486528</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49.008646</td>
      <td>8.398101</td>
      <td>2011-09-29 13:55:00.302340864</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49.008646</td>
      <td>8.398100</td>
      <td>2011-09-29 13:55:00.406079232</td>
    </tr>
  </tbody>
</table>



B-提取KITTI数据集图像（'2011_09_29_drive_0071_sync'数据子集），并转换为tensor


```python
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm.auto import tqdm

drive_29_0071_img_fp=util.filePath_extraction(r'E:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\data',['png'])
drive_29_0071_img_fp_list=util.flatten_lst([[os.path.join(k,f) for f in drive_29_0071_img_fp[k]] for k,v in drive_29_0071_img_fp.items()])
print("2011_09_29_drive_0071_sync-数据子集图像数据：",len(drive_29_0071_img_fp_list))

trans_2tensor=transforms.Compose([transforms.ToTensor(),]) #将图像转换为tensor
drive_29_0071_img_tensor=[trans_2tensor(Image.open(i)) for i in tqdm(drive_29_0071_img_fp_list)]
```

    2011_09_29_drive_0071_sync-数据子集图像数据： 1059
    


      0%|          | 0/1059 [00:00<?, ?it/s]


C-加载已训练的模型，并预测数据集图像


```python
model_entire=torch.load('./model/mask_R_CNN_person/mask_R_CNN_person_final.pth')
model_entire.eval()
with torch.no_grad():
    drive_29_0071_img_pred=[model_entire([i.to(device)])[0]['masks'].shape[0]  for i in tqdm(drive_29_0071_img_tensor)]
```


      0%|          | 0/1059 [00:00<?, ?it/s]


D-显示人流分布密度


```python
drive_29_0071_info_coordi['person_num']=drive_29_0071_img_pred

import plotly.express as px
fig = px.density_mapbox(drive_29_0071_info_coordi, lat='lat', lon='lon', z='person_num', radius=10,
                        center=dict(lat=49.008645, lon=8.398104), zoom=18,
                        mapbox_style="stamen-terrain")
fig.show()
```

<a href=""><img src="./imgs/23_12.png" height='auto' width='auto' title="caDesign"></a>  


### 1.2 图像分割（Semantic Segmentation,torchvision.models）_FCN-RESNET101，城市空间内容统计与关联网络结构
PyTorch图像/语义分割模型(semantic segmentation)，包括[Faster R-CNN ResNet-50 FPN](https://arxiv.org/abs/1506.01497),[Mask R-CNN ResNet-50 FPN](https://arxiv.org/abs/1703.06870)，其预先训练的模型采用的数据集为[COCO train2017](https://cocodataset.org/#home)的子集[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)，包括有20个分类。$['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']$。使用预先训练的模型，输入的图像期望已训练时相同的方式归一化处理。图像映射到[0,1]区间，使用mean =[0.485, 0.456, 0.406]和std =[0.229, 0.224, 0.225]进行归一化。图像变换的处理使用`torchvision.transforms`实现。

通过图像分割可以获取一些城市对象（Pascal VOC有20个分类），对于KITTI数据集而言，就可以获取每一位置的城市对象内容，这样就可以对城市空间的内容加以统计，并通过建立关联结构分析对象之间的关系（即在多处场景中，同时出现某些对象的可能性大小）。

> 参考：
1. [FCN-RESNET101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)
2. [PyTorch for Beginners: Semantic Segmentation using torchvision](https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/)

#### 1.2.1 FCN-RESNET101图像分割
直接读取PyTorch预训练的模型，用于新场景的应用。

* 关于[`TORCH.UNSQUEEZE`](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)

指定参数dim(轴)，即插入位置，返回包含一个轴尺寸为1的新张量。


```python
import torch
x=torch.tensor([1, 2, 3, 4])
print('dim=0',torch.unsqueeze(x, 0))
print('dim=1',torch.unsqueeze(x, 1))
```

    dim=0 tensor([[1, 2, 3, 4]])
    dim=1 tensor([[1],
            [2],
            [3],
            [4]])
    

* 关于[TORCH.SQUEEZE](https://pytorch.org/docs/stable/generated/torch.squeeze.html)

可以根据指定轴，移除轴尺寸为1的轴，与`torch.unsqueeze`互逆。


```python
x = torch.zeros(2, 1, 2, 1, 2)
print('x.size:{}\ndim=default:{}\ndim=0:{}\ndim=1:{}'.format(x.size(),torch.squeeze(x).size(),torch.squeeze(x,dim=0).size(),torch.squeeze(x,dim=1).size()))
```

    x.size:torch.Size([2, 1, 2, 1, 2])
    dim=default:torch.Size([2, 2, 2])
    dim=0:torch.Size([2, 1, 2, 1, 2])
    dim=1:torch.Size([2, 2, 1, 2])
    

---


```python
import torch,os
import util
#A-加载模型
from torchvision import models
fcn=models.segmentation.fcn_resnet101(pretrained=True).eval() 

#B-加载数据集
drive_29_0071_img_fp=util.filePath_extraction(r'E:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\data',['png'])
drive_29_0071_img_fp_list=util.flatten_lst([[os.path.join(k,f) for f in drive_29_0071_img_fp[k]] for k,v in drive_29_0071_img_fp.items()])
#指定运算设备,GPU or CPU
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```


```python
#C-映射图像分割的类标（类别）
def decode_segmap_FCN_RESNET101(image, nc=21):
    import numpy as np
    '''
    function - fcn_resnet101模型，图像分割的类别给与颜色标识
    ''' 
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb
```


```python
#D-预测图像分割结果，并打印
def segmentation_FCN_RESNET101_plot(net, path, show_orig=True, dev='cuda',img_resize=640,figsize=(20, 20)):
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    import torchvision.transforms as T
    '''
    function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，并打印显示分割预测结果
    '''
    img = Image.open(path)
    plt.figure(figsize=figsize)
    if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([
                   T.Resize(img_resize), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap_FCN_RESNET101(om)
    plt.figure(figsize=figsize)
    plt.imshow(rgb); plt.axis('off'); plt.show()

#E-提取一副图像预测
segmentation_FCN_RESNET101_plot(fcn, drive_29_0071_img_fp_list[200],dev=device,img_resize=480) #可以通过调整img_resize参数，即调整图像大小来减少GPU使用量，避免GPU溢出
```


    
<a href=""><img src="./imgs/23_09.png" height='auto' width='auto' title="caDesign"></a>  
    



    
<a href=""><img src="./imgs/23_10.png" height='auto' width='auto' title="caDesign"></a>  
    



```python
#F-计算KITTI-drive_29_0071_img子集的所有图像分割，返回结果并动态显示
def segmentation_FCN_RESNET101_animation(net, paths,save_path='./animation.mp4' ,dev='cuda',img_resize=640,interval=150,figsize=(20, 20)):
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    import torchvision.transforms as T
    import matplotlib.animation as animation
    from tqdm.auto import tqdm
    '''
    function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，并打印显示分割预测结果的动画
    '''
    plt.figure(figsize=figsize)
    imgs=[]
    fig=plt.figure(figsize=figsize)
    for path in tqdm(paths):
        img=Image.open(path)
        trf=T.Compose([ T.Resize(img_resize), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        inp=trf(img).unsqueeze(0).to(dev)
        out=net.to(dev)(inp)['out']
        sementic_seg=torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb=decode_segmap_FCN_RESNET101(sementic_seg)        
        imgs.append([plt.imshow(rgb,animated=True,)])        
    anima=animation.ArtistAnimation(fig,imgs,interval=interval, blit=True,repeat_delay=1000)
    anima.save(save_path)
    print(".mp4 saved.")
    return anima,imgs

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import HTML

save_path=r'./imgs/segmentation_FCN_RESNET101_animation.mp4'
anima,_=segmentation_FCN_RESNET101_animation(fcn, drive_29_0071_img_fp_list, save_path,dev='cuda',img_resize=320,figsize=(20,8))
HTML(anima.to_html5_video())
```

<video width='auto' height='auto' controls><source src="./imgs/segmentation_FCN_RESNET101_animation.mp4" height='auto' width='auto' title="caDesign" type='video/mp4'></video>

#### 1.2.1 城市空间内容统计与关联网络结构
城市空间内容统计是确定每一位置空间下存在有哪些物，因为使用的训练模型包括20个分类，因此只能识别这些已训练的对象，而树木、建筑等不能识别；再者，该模型只返回对象掩码（不同对象，不同索引，但是同一对象不可再分，例如对人数的统计不能实现）。但是对每一位置空间已有对象的识别可以初步的判断该空间的特征；并通过统计所有位置下，与每一对象同时存在的其它对象的频数，可以应用[NetworkX](https://networkx.org/)库构建网络结构，更直观的观察对象之间的空间存在关系。并统计每一位置出现对象种类的数量，以热力图的形式可视化，可以初步判断不同地段的对象混杂程度，通常对象种类越丰富的区域，空间表现出的活力越高。

首先调整了预测函数，使其返回值为图像分割索引及对应的分类名称。


```python
def segmentation_FCN_RESNET101(net, path, show_orig=True, dev='cuda',img_resize=640):
    from PIL import Image
    import torch
    import torchvision.transforms as T
    import numpy as np    
    '''
    function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，返回预测结果
    '''
    seg_FCN_RESNET101_classi_mapping={0:'background',1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle',6:'bus', 7:'car', 8:'cat', 9:'chair', 10:'cow',11:'dining table', 12:'dog', 
                                      13:'horse', 14:'motorbike', 15:'person',16:'potted plant', 17:'sheep', 18:'sofa', 19:'train', 20:'tv/monitor'}
    img=Image.open(path)
    trf=T.Compose([
                   T.Resize(img_resize), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    inp=trf(img).unsqueeze(0).to(dev)
    out=net.to(dev)(inp)['out']
    sementic_seg=torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    sementic_seg_classi=[seg_FCN_RESNET101_classi_mapping[i] for i in np.unique(sementic_seg)]
    return (np.unique(sementic_seg).tolist(),sementic_seg_classi)
```


```python
sementic_seg=segmentation_FCN_RESNET101(fcn, drive_29_0071_img_fp_list[200],dev=device,img_resize=280)
print('预测的图像包含的对象标签：{}'.format(sementic_seg))
```

    预测的图像包含的对象标签：([0, 2, 7, 9, 11, 15], ['background', 'bicycle', 'car', 'chair', 'dining table', 'person'])
    

计算所有图像，返回预测结果。


```python
from tqdm.auto import tqdm
drive_29_0071_img_seg_pred=[segmentation_FCN_RESNET101(fcn, img,dev=device,img_resize=280) for img in tqdm(drive_29_0071_img_fp_list)]
```


      0%|          | 0/1059 [00:00<?, ?it/s]


通过查看预测结果，可以发现之前未注意场景中存在狗和火车，进一步核实图像，确实存在。而一些未出现的类，例如sheep、hourse等应该狗在不同影像位置下识别的错误。其它的分类错误，可能与场景中出现的海报等图画有关。观察最终的网络结构，因为将每一个对象与其它对象在不同场景中共存的情况进行统计，即计算共存对象的频数，将该频数或者其倍数作为网络边的权重值，并通过粗细显示。因此可以观察到，线越细的对象在整个1059张图像所代表的位置空间下，出现的位置较少；而线越粗的则出现位置相对较多。经常同时出现的对象为'person','car','bicycle',与之次之包含的有'chair','motorbike','potted plant'等。有些信息的出现是合乎常理，例如场景中骑车的人，因此这些分析结果似乎价值偏弱；但是，'chair'和'potted plant'的出现，可以判定该条街道室外活动的主要内容，餐饮、休闲等。


```python
def count_list_frequency(lst):
    '''
    function - 计算列表的频数
    '''
    freq={}
    for i in lst:
        if(i in freq):
            freq[i]+=1
        else:
            freq[i]=1
    return freq       
    

def objects_network_PascalVOC(seg_idxs,figsize=(15,15),layout='spring_layout',w_ratio=0.5):
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    
    '''
    function - 根据连续的图像分割数据，计算各个一对象（真实世界存在的物）与其它对象对应的数量，构建网络结构，分析相互关系
    '''
    seg_FCN_RESNET101_classi_mapping={0:'background',1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle',6:'bus', 7:'car', 8:'cat', 9:'chair', 10:'cow',11:'dining table', 12:'dog', 
                                      13:'horse', 14:'motorbike', 15:'person',16:'potted plant', 17:'sheep', 18:'sofa', 19:'train', 20:'tv/monitor'}    
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    unique_idxs=np.unique(flatten_lst(seg_idxs)).tolist()
    unique_idxs_=list(filter(lambda x: x != 0, unique_idxs))
    print('存在的对象有：',[(i,seg_FCN_RESNET101_classi_mapping[i]) for i in unique_idxs])
    
    #01-收集每一对象所有存在时刻包含的其他对象
    object_associations={}
    for obj in unique_idxs_:
        obj_associations_list=flatten_lst([i for i in seg_idxs if obj in i])
        obj_associations_list_=list(filter(lambda x: x != 0,obj_associations_list))
        object_associations[obj]=obj_associations_list_
    #print(object_associations)
    
    #02-计算每一对象，包含其他对象的频数
    object_associations_frequency={}
    for k,v in object_associations.items():
        v_=list(filter(lambda x: x != k,v))
        object_associations_frequency[k]=count_list_frequency(v_)
    #print(object_associations_frequency)
    
    #03-构建网络，以频数或其倍数为权重
    fig, ax = plt.subplots(figsize=figsize)
    G=nx.Graph()
    layout_dic={
        'spring_layout':nx.spring_layout,   
        'random_layout':nx.random_layout,
        'circular_layout':nx.circular_layout,
        'kamada_kawai_layout':nx.kamada_kawai_layout,
        'shell_layout':nx.shell_layout,
        'spiral_layout':nx.spiral_layout,
    }
    
    
    for k,v in object_associations_frequency.items():
        for obj,w in v.items():
            G.add_edge(seg_FCN_RESNET101_classi_mapping[k],seg_FCN_RESNET101_classi_mapping[obj],weight=w*w_ratio)
            
    pos=layout_dic[layout](G) 
    weights=nx.get_edge_attributes(G,'weight').values()
    nx.draw(G,pos,with_labels=True,font_size=18,alpha=0.4, edge_color="r",node_size=200,width=list(weights))

seg_idxs=[i[0] for i in drive_29_0071_img_seg_pred]
objects_network_PascalVOC(seg_idxs,layout='shell_layout',w_ratio=0.03)
```

    存在的对象有： [(0, 'background'), (1, 'aeroplane'), (2, 'bicycle'), (3, 'bird'), (6, 'bus'), (7, 'car'), (9, 'chair'), (11, 'dining table'), (12, 'dog'), (13, 'horse'), (14, 'motorbike'), (15, 'person'), (16, 'potted plant'), (17, 'sheep'), (18, 'sofa'), (19, 'train')]
    


    
<a href=""><img src="./imgs/23_11.png" height='auto' width='auto' title="caDesign"></a> 
    


* 空间对象种类的丰富程度


```python
import util
KITTI_info_fp=r'E:\dataset\KITTI\2011_09_29_drive_0071_sync\oxts\data'
timestamps_fp=r'E:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\timestamps.txt'
drive_29_0071_info=util.KITTI_info(KITTI_info_fp,timestamps_fp)
drive_29_0071_info_coordi=drive_29_0071_info[['lat','lon','timestamps_']]

obj_num=[len(i) for i in seg_idxs]
drive_29_0071_info_coordi['obj_num']=obj_num
drive_29_0071_info_coordi['idx']=drive_29_0071_info_coordi.index
drive_29_0071_info_coordi
```

    <ipython-input-81-0aa38d547d2e>:8: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    <ipython-input-81-0aa38d547d2e>:9: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lon</th>
      <th>timestamps_</th>
      <th>obj_num</th>
      <th>idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.008645</td>
      <td>8.398104</td>
      <td>2011-09-29 13:54:59.990872576</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.008645</td>
      <td>8.398103</td>
      <td>2011-09-29 13:55:00.094612992</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.008646</td>
      <td>8.398102</td>
      <td>2011-09-29 13:55:00.198486528</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49.008646</td>
      <td>8.398101</td>
      <td>2011-09-29 13:55:00.302340864</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49.008646</td>
      <td>8.398100</td>
      <td>2011-09-29 13:55:00.406079232</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>49.009498</td>
      <td>8.395251</td>
      <td>2011-09-29 13:56:49.458599424</td>
      <td>4</td>
      <td>1054</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>49.009498</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.562463744</td>
      <td>4</td>
      <td>1055</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>49.009497</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.666327808</td>
      <td>3</td>
      <td>1056</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>49.009498</td>
      <td>8.395249</td>
      <td>2011-09-29 13:56:49.770316544</td>
      <td>3</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>49.009498</td>
      <td>8.395248</td>
      <td>2011-09-29 13:56:49.874179584</td>
      <td>3</td>
      <td>1058</td>
    </tr>
  </tbody>
</table>
<p>1059 rows × 5 columns</p>
</div>




```python
import plotly.express as px
fig = px.density_mapbox(drive_29_0071_info_coordi, lat='lat', lon='lon', z='obj_num', radius=10,
                        center=dict(lat=49.008645, lon=8.398104), zoom=18,
                        mapbox_style="stamen-terrain",
                        title='The richness of the spatial object types',
                        #hover_data=['idx'],
                        hover_name='idx'
                       )
fig.show()
```

<a href=""><img src="./imgs/23_13.png" height='auto' width='auto' title="caDesign"></a> 

### 1.5 要点
#### 1.5.1 数据处理技术

* 直接应用PyTorch提供的预训练模型

* 预训练模型精调

* 直接应用`https://github.com/pytorch/vision.git`代码中的训练、评估方法，减小代码编写量

* 深度模型的保存方式与读取

#### 1.5.2 新建立的函数

* class - PennFudanPed数据集，建立可迭代对象(迁移). `PennFudanDataset(torch.utils.data.Dataset)`

* function - PennFudanPed数据集，行人对象检测模型精调（迁移）. `get_instance_segmentation_model(num_classes)`

* function - 转换图像为tensor，可以包含多种变化（迁移）. `get_transform(train)`

* function - fcn_resnet101模型，图像分割的类别给与颜色标识（迁移）. `decode_segmap_FCN_RESNET101(image, nc=21)`

* function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，并打印显示分割预测结果（迁移，更新）. `segmentation_FCN_RESNET101_plot(net, path, show_orig=True, dev='cuda',img_resize=640,figsize=(20, 20))`

* function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，并打印显示分割预测结果的动画. `segmentation_FCN_RESNET101_animation(net, paths,save_path='./animation.mp4' ,dev='cuda',img_resize=640,interval=150,figsize=(20, 20))`

* function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，返回预测结果. `egmentation_FCN_RESNET101(net, path, show_orig=True, dev='cuda',img_resize=640)`

* function - 计算列表的频数. `count_list_frequency(lst)`

* function - 根据连续的图像分割数据，计算各个一对象（真实世界存在的物）与其它对象对应的数量，构建网络结构，分析相互关系. `objects_network_PascalVOC(seg_idxs,figsize=(15,15),layout='spring_layout',w_ratio=0.5)`

#### 1.5.3 所调用的库


```python
from PIL import Image
import os
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image
import json

from engine import train_one_epoch, evaluate  #pip install pycocotools-windows  安装关联库
import transforms as T
from tqdm.auto import tqdm
import plotly.express as px
import networkx as nx
```

#### 1.5.4 参考文献
1. Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick.Mask R-CNN.[Submitted on 20 Mar 2017 (v1), last revised 24 Jan 2018 (this version, v3)]. arXiv:1703.06870
2. TorchVision Object Detection Finetuning Tutorial
3. FCN-RESNET101
4. PyTorch for Beginners: Semantic Segmentation using torchvision
5. Intersection over Union (IoU) for object detection
