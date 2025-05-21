import numpy as np

from .contact_graspnet import ContactGraspNet
from .grounding_dino import GroundingDINO
from .segment_anything import SegmentAnythingV2
from .ai_client import QwenClient

cgn = ContactGraspNet()
gdino = GroundingDINO()
sam2 = SegmentAnythingV2("large")
qwen_vl = QwenClient("qwen-max-latest")


def grounded_sam(image: np.ndarray,
                 caption: str):
    """ Grounding the object in the image and segment it """
    dets = gdino(image, caption)
    return sam2(image, box=dets.xyxy)


FUNCTIONS = {
    "ContactGraspNet": cgn.__call__,
    "GroundingDINO": gdino.__call__,
    "GroundedSAM": grounded_sam,
    "SAM2": sam2.__call__,
    "Qwen-VL": qwen_vl.query_once
}


def invoke(func, *args, **kwargs):
    return FUNCTIONS[func](*args, **kwargs)
