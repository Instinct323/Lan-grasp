from .ai_client import QwenClient
from .contact_graspnet import ContactGraspNet
from .grounding_dino import GroundingDINO
from .segment_anything import SegmentAnythingV2

FUNCTIONS = {
    "ContactGraspNet": ContactGraspNet().__call__,
    "GroundingDINO": GroundingDINO().__call__,
    "SAM2": SegmentAnythingV2("large").__call__,
    "Qwen-VL": QwenClient("qwen-vl-max-2025-04-08").query_once
}
