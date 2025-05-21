import concurrent.futures
import logging
import pickle
import time
from typing import Union

import cv2
import numpy as np
import requests
import supervision as sv

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


def annotate_pts(bgr, pts):
    size_min = min(bgr.shape[1::-1])
    pt_size = max(1, round(0.01 * size_min))
    font_scale = 0.002 * size_min

    bgr = bgr.copy()
    for i, pt in enumerate(pts.tolist()):
        bgr = cv2.circle(bgr, pt, pt_size, (0, 255, 0), -1)
        bgr = cv2.putText(bgr, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
    return bgr


def farthest_point_sampling(pts: np.ndarray,
                            num_samples: Union[int, float]):
    """ Farthest Point Sampling (FPS) algorithm.
        :param pts: The input point cloud.
        :param num_samples: The number of points to sample.
        :return: The sampled points and their indices."""
    n = len(pts)
    if isinstance(num_samples, float): num_samples = int(num_samples * n)
    if num_samples > n: return pts, None

    distances = np.full(n, np.inf)
    indices = np.zeros(num_samples, dtype=int)
    indices[0] = np.random.randint(0, n)

    for i in range(num_samples - 1):
        distances = np.minimum(distances, np.linalg.norm(pts - pts[indices[i]], axis=1))
        indices[i + 1] = np.argmax(distances)

    return pts[indices], indices


class FunctionsAPI:

    def __init__(self, url):
        self.url = url
        self.executor = concurrent.futures.ThreadPoolExecutor()
        assert requests.get(f"{self.url}/docs").status_code == 200
        LOGGER.info(f"See {self.url}/docs for API documentation.")

    def invoke(self, func, *args, **kwargs):
        data = {"func": func, "args": args, "kwargs": kwargs}
        response = requests.post(f"{self.url}/invoke", data=pickle.dumps(data), headers={"t-send": str(int(time.time()))})
        assert response.status_code == 200, f"{response.status_code}, {response.text}"

        headers = response.headers
        LOGGER.info(f"[{func}] {headers['cost']}, recv:{time.time() - float(headers['t-send']):.3f}s")
        return pickle.loads(response.content)

    def invoke_async(self, func, *args, **kwargs):
        return self.executor.submit(self.invoke, func, *args, **kwargs)


def sv_annotate(image: np.ndarray,
                detections: sv.Detections,
                mask_opacity: float = 0.7,
                anno_box: bool = True,
                smart_label: bool = True) -> np.ndarray:
    """ :param image: OpenCV image
        :param detections: Supervision Detections with xyxy, confidence ..."""
    color_lookup = sv.ColorLookup.CLASS if detections.mask is None else sv.ColorLookup.INDEX
    image = image.copy()

    if anno_box:
        anno_box = sv.BoxAnnotator(color_lookup=color_lookup)
        image = anno_box.annotate(image, detections=detections)

    if detections.mask is not None:
        anno_mask = sv.MaskAnnotator(color_lookup=color_lookup, opacity=mask_opacity)
        image = anno_mask.annotate(image, detections=detections)

    if detections.confidence is not None:
        anno_label = sv.LabelAnnotator(color_lookup=color_lookup, smart_position=smart_label)
        labels = [f"{score:.2f}" for score in detections.confidence] if detections.class_id is None else [
            f"{phrase} {score:.2f}" for phrase, score in zip(detections.metadata[detections.class_id], detections.confidence)]
        image = anno_label.annotate(image, detections=detections, labels=labels)

    return image
