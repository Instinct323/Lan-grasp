import json

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class JSONprompter(dict):

    def prompt(self):
        return ("You need to return following fields in the JSON format, without any other text: \n" +
                "\n".join(f"- {k}: {v}" for k, v in self.items()))

    def decode(self,
               response: str,
               as_code_block: bool = True):
        return json.loads(response[7:-3] if as_code_block else response)


class Gripper:

    def __init__(self,
                 url: str,
                 img_size: int = 512):
        self.remote = FunctionsAPI(url)
        self.img_size = img_size
        self.json_prompter = JSONprompter(
            grasping_part="i.e., word or phrase",
            point="point ID",
            reason="i.e., a sentence",
        )

    def grouding_obj(self,
                     bgr: np.ndarray,
                     obj: str,
                     padding: float):
        """ :return: bounding box of the object in the image """
        dets = self.remote.invoke("GroundingDINO", bgr, caption=obj)
        LOGGER.info("Confidence: " + str(dets.confidence))
        bbox = dets.xyxy[0]
        # Scale the bounding box
        pad = padding * (bbox[2:] - bbox[:2])
        bbox[:2] = np.maximum(bbox[:2] - pad, 0)
        bbox[2:] = np.minimum(bbox[2:] + pad, bgr.shape[1::-1])
        return bbox

    def process(self,
                bgr: np.ndarray,
                obj: str,
                task: str):
        """ :param bgr: observe image
            :param obj: target object
            :param task: task name """
        bgr = sv.resize_image(bgr, [self.img_size] * 2, keep_aspect_ratio=True)
        # Grouding the object
        bbox = self.grouding_obj(bgr, obj, 0.15)
        bgr_obj = sv.crop_image(bgr, bbox)
        # Annotate the image
        pts = sample_pts(*bgr_obj.shape[1::-1], 50)
        bgr_obj_ann = annotate_pts(bgr_obj, pts)
        # Obtain the gripping area
        prompt = (
                f"You are an intelligent robotic arm. "
                f"If you want to {task} the {obj} in the image, which part makes the most sense to grasp? Name one part. "
                f"Please indicate the part by the point ID in the image. "
                + self.json_prompter.prompt()
        )
        vlm_ret = self.remote.invoke("Qwen-VL", 128, image=Image.fromarray(bgr_obj_ann[..., ::-1]), text=prompt)
        vlm_ret = self.json_prompter.decode(vlm_ret)

        print(prompt)
        print(vlm_ret)
        plt.imshow(bgr_obj_ann[..., ::-1]), plt.show()


if __name__ == '__main__':
    gripper = Gripper("http://127.0.0.1:8000")

    img = cv2.imread("assets/cup.jpeg")
    gripper.process(img, "glass", "pick up")
