from detectron2.config import get_cfg
from detectron2 import model_zoo

### BACKBONE

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from torchvision.models import mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor
from detectron2.modeling.backbone.fpn import FPN
import torch
import torchvision
from os import environ
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'static'))
RESULT_DIR = os.path.join(STATIC_FOLDER, 'ndre_result')
os.makedirs(RESULT_DIR, exist_ok=True)

TEMP_DIR = os.path.join(STATIC_FOLDER, 'ndre_result')
os.makedirs(TEMP_DIR, exist_ok=True)



BACKBONE_REGISTRY._obj_map.clear()

@BACKBONE_REGISTRY.register()
class mobilenetv2(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    mobilenet = mobilenet_v2(pretrained=True).features

    self.stages = torch.nn.ModuleList()
    self.out_strides = []
    self.out_channels = []

    res1 = torch.nn.Sequential(*mobilenet[:4])
    res2 = torch.nn.Sequential(*mobilenet[4:7])
    res3 = torch.nn.Sequential(*mobilenet[7:14])
    res4 = torch.nn.Sequential(*mobilenet[14:])

    self.stages.extend([res1, res2, res3, res4])
    self._out_features = ["res2", "res3", "res4", "res5"]
    self._out_channels = {
        "res2" : 24,
        "res3" : 32,
        "res4" : 96,
        "res5" : 1280
    }
    self._out_strides = {
        "res2" : 4,
        "res3" : 8,
        "res4" : 16,
        "res5" : 32
    }

  def forward(self, x):
    out = {}
    feat = x
    for i, stage in enumerate(self.stages, start=2):
      feat = stage(feat)
      out[f"res{i}"] = feat
    return out

  def output_shape(self):
    return {
        f: ShapeSpec(
            channels=self._out_channels[f],
            stride = self._out_strides[f]
        ) for f in self._out_features
    }

@BACKBONE_REGISTRY.register()
def buildCustomFPNBackbone(cfg, input_shape):
    # 1) Build the ResNet-18 bottom-up
    bottom_up = mobilenetv2(cfg, input_shape)
    # 2) Wrap in FPN
    from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
    return FPN(
        bottom_up=bottom_up,
        in_features=["res2","res3","res4","res5"],
        out_channels=256,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )

### BACKONE
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE = "cuda" if environ.get("INFERENCE_DEVICE", "cpu") == "cuda" else "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "model/weights/weight.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.BACKBONE.NAME = "buildCustomFPNBackbone"