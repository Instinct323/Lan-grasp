import torch

from .contact_graspnet import ContactGraspNet
from .grounding_dino import GroundingDINO
from .qwen_vl import QwenVL
from .segment_anything import SegmentAnythingV2

cgn = ContactGraspNet()
gdino = GroundingDINO()
sam2 = SegmentAnythingV2("tiny")
qwen_vl = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16)


FUNCTIONS = {
    "ContactGraspNet": cgn.__call__,
    "GroundingDINO": gdino.__call__,
    "SAM2": sam2.__call__,
    "Qwen-VL": qwen_vl.query_once,
}
