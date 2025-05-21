import json
import re

import matplotlib.pyplot as plt

import api
from toolkit.client_utils import *


class JSONprompter(dict):

    def prompt(self):
        return ("Your response should be formatted as a JSON code block, without any additional text. "
                "The required fields are:\n" + "\n".join(f"- {k}: {v}" for k, v in self.items()))

    def decode(self,
               response: str):
        content = re.search(r"```json(.*)```", response, flags=re.S)
        return json.loads(content.group(1))


class PointAnnotator:

    def __init__(self,
                 n_pts: int):
        self.n_pts = n_pts

    def __call__(self,
                 image: np.ndarray,
                 mask: np.ndarray):
        pts = np.stack(np.where(mask), axis=-1)
        # pts = pts[np.random.choice(range(len(pts)), self.n_pts, replace=False)]
        pts = farthest_point_sampling(pts, self.n_pts)[0]
        return annotate_pts(image, pts), pts


class Gripper:

    def __init__(self,
                 img_size: int = 512):
        self.img_size = img_size
        self.json_prompter = JSONprompter(
            grasping_part="A word or phrase describing the part to grasp.",
            point="The point ID of the chosen part in the image.",
            reason="A sentence explaining why this part is the best choice for grasping.",
        )
        self.pts_anno = PointAnnotator(30)

    def grouding_obj(self,
                     bgr: np.ndarray,
                     obj: str):
        """ :return: bounding box of the object in the image """
        dets = api.invoke("GroundingDINO", bgr, caption=obj)
        LOGGER.info("Confidence: " + str(dets.confidence))
        return dets[0]

    def process(self,
                bgr: np.ndarray,
                obj: str,
                task: str):
        """ :param bgr: observe image
            :param obj: target object
            :param task: task name """
        bgr = sv.resize_image(bgr, [self.img_size] * 2, keep_aspect_ratio=True)
        anno_mask = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        # Grouding the object
        dets = api.invoke("GroundingDINO", bgr, caption=obj)
        LOGGER.info("Confidence: " + str(dets.confidence))
        dets = dets[0]
        bgr_obj = sv.crop_image(bgr, sv.scale_boxes(dets.xyxy, 1.))

        # Obtain the mask of the object
        dets_obj = api.invoke("SAM2", bgr_obj, box=[[0, 0, *bgr_obj.shape[1::-1]]])
        # Erode then dilate the mask
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_obj = dets_obj.mask[0].astype(np.uint8)
        mask_obj = cv2.erode(mask_obj, kernel, iterations=2)
        mask_obj = cv2.dilate(mask_obj, kernel, iterations=2)
        dets_obj.mask[0] = mask_obj.astype(np.bool_)

        # anno_mask.annotate(bgr_obj, dets_obj)
        # Annotate the image
        bgr_obj_ann, pts = self.pts_anno(bgr_obj, dets_obj.mask[0])
        # Obtain the grasping area
        prompt = (
                f"You are an intelligent robotic arm. "
                f"If you want to {task} the {obj} in the image, which part makes the most sense to grasp? Name one part. "
                + self.json_prompter.prompt()
        )
        LOGGER.info("Prompt: " + prompt)
        vlm_ret = api.invoke("Qwen-VL", None, image=bgr_obj_ann, text=prompt)
        LOGGER.info("VLM response: " + vlm_ret)
        vlm_ret = self.json_prompter.decode(vlm_ret)
        LOGGER.info("JSON response: " + str(vlm_ret))

        # Extract the grasping part
        pt_id = int(vlm_ret["point"])
        dets_obj = api.invoke("SAM2", bgr_obj_ann, point_coords=[pts[pt_id]], point_labels=[1])
        print(dets_obj.area)
        anno_mask.annotate(bgr_obj_ann, dets_obj)

        plt.imshow(bgr_obj_ann[..., ::-1])
        plt.show()


if __name__ == '__main__':
    gripper = Gripper()

    img = cv2.imread("assets/cup.jpg")
    gripper.process(img, "glass", "pick up")
