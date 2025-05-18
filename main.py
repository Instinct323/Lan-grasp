import concurrent.futures
import logging
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import supervision as sv
from PIL import Image

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


class FunctionsAPI:

    def __init__(self, url):
        self.url = url
        self.executor = concurrent.futures.ThreadPoolExecutor()
        LOGGER.info("Connecting...")
        while True:
            try:
                res = requests.get(f"{self.url}/docs")
                if res.status_code == 200: break
            except:
                pass
        LOGGER.info(f"See {self.url}/docs for API documentation.")

    def invoke(self, func, *args, **kwargs):
        t0 = time.time()
        data = {"func": func, "args": args, "kwargs": kwargs}
        response = requests.post(f"{self.url}/invoke", data=pickle.dumps(data))
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        LOGGER.info(f"{func}: {time.time() - t0:.3f}s")
        return pickle.loads(response.content)

    def invoke_async(self, func, *args, **kwargs):
        return self.executor.submit(self.invoke, func, *args, **kwargs)


def mesh_points(w, h, stride):
    x = np.arange(stride // 2, w, stride)
    y = np.arange(stride // 2, h, stride)
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


class Gripper:
    vlm_ret_parts = [
        "<reason, i.e., why you choose this part>",
        "<grasping part, i.e., word or phrase>",
        "<bbox, e.g., [top-left, top-right, bottom-left, bottom-right]>",
    ]
    vlm_ret_separator = " | "

    def __init__(self,
                 url: str,
                 img_size: int = 512):
        self.remote = FunctionsAPI(url)
        self.img_size = img_size

    def process(self,
                bgr: np.ndarray,
                obj: str,
                task: str):
        """ :param bgr: observe image
            :param obj: target object
            :param task: task name """
        bgr = sv.resize_image(bgr, [self.img_size] * 2, keep_aspect_ratio=True)
        pts = mesh_points(*bgr.shape[1::-1], 64)
        # Annotate the image
        bgr_ann = bgr.copy()
        for i, pt in enumerate(pts.tolist()):
            bgr_ann = cv2.circle(bgr_ann, pt, 5, (0, 255, 0), -1)
            bgr_ann = cv2.putText(bgr_ann, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Obtain the gripping area
        prompt = (
                f"You are an intelligent robotic arm. "
                f"If you want to {task} the {obj} in the image, which part makes the most sense to grasp? Name one part. "
                f"Please connect the IDs of labeled points to form a bounding box that approximates the grasping part."
                f"You need to return the result in the specified format: "
                + self.vlm_ret_separator.join(self.vlm_ret_parts)
        )
        vlm_ret = self.remote.invoke("Qwen-VL", 128, image=Image.fromarray(bgr_ann[..., ::-1]), text=prompt
                                     ).split(self.vlm_ret_separator)
        assert len(vlm_ret) == len(self.vlm_ret_parts), f"Invalid response: {vlm_ret}"

        print(prompt)
        print(vlm_ret)
        plt.imshow(bgr_ann[..., ::-1]), plt.show()


if __name__ == '__main__':
    gripper = Gripper("http://10.16.95.165:8000")

    img = cv2.imread("assets/cup.jpeg")
    gripper.process(img, "cup", "pick up")
