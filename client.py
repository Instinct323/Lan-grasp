import json

import matplotlib.pyplot as plt
from PIL import Image

from toolkit.client_utils import *


class JSONprompter(dict):

    def prompt(self):
        return ("You need to return following fields in the JSON format, without any other text: \n" +
                "\n".join(f"- {k}: {v}" for k, v in self.items()))

    def decode(self,
               response: str,
               as_code_block: bool = True):
        return json.loads(response[7:-3] if as_code_block else response)


class PatchAnnotator:

    def __init__(self,
                 n_patches: int,
                 patch_size: int):
        self.n_patches = n_patches
        self.patch_size = patch_size
        LOGGER.info(f"The image will be resized to {np.sqrt(n_patches) * patch_size:.0f}^2 pixels.")

    def __call__(self,
                 image: np.ndarray):
        h, w = image.shape[:2]
        nh = round(np.sqrt(self.n_patches / h * w))
        nv = round(np.sqrt(self.n_patches * h / w))
        image = cv2.resize(image, (nh * self.patch_size, nv * self.patch_size))

        pts = np.mgrid[self.patch_size // 2: image.shape[1]: self.patch_size,
              self.patch_size // 2: image.shape[0]: self.patch_size].T.reshape(-1, 2)
        return annotate_pts(image, pts), pts


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
        self.patch_anno = PatchAnnotator(50, 14 * 2)

    def grouding_obj(self,
                     bgr: np.ndarray,
                     obj: str,
                     padding: float):
        """ :return: bounding box of the object in the image """
        dets = self.remote.invoke("GroundingDINO", bgr, caption=obj)
        LOGGER.info("Confidence: " + str(dets.confidence))
        bbox = dets.xyxy[0]
        # Scale the bounding box
        if padding > 0:
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
        bbox = self.grouding_obj(bgr, obj, 0.1)
        bgr_obj = sv.crop_image(bgr, bbox)
        # Annotate the image
        bgr_obj_ann, pts = self.patch_anno(bgr_obj)
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

    img = cv2.imread("assets/cup.jpg")
    gripper.process(img, "glass", "pick up")
