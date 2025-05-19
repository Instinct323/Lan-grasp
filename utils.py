import concurrent.futures
import logging
import pickle
import time

import numpy as np
import requests
import cv2
import supervision as sv

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


def sample_pts(w, h, npts):
    aspect_radio = h / w
    stride_h = w / round(np.sqrt(npts / aspect_radio))
    stride_v = h / round(np.sqrt(npts * aspect_radio))

    x = np.arange(stride_h / 2, w, stride_h)
    y = np.arange(stride_v / 2, h, stride_v)
    x, y = map(lambda i: np.round(i).astype(int), (x, y))
    return np.stack(list(map(np.ravel, np.meshgrid(x, y))), axis=1)


def annotate_pts(bgr, pts):
    size_min = min(bgr.shape[1::-1])
    pt_size = max(1, round(0.01 * size_min))
    font_scale = 0.002 * size_min
    bgr = bgr.copy()
    for i, pt in enumerate(pts.tolist()):
        bgr = cv2.circle(bgr, pt, pt_size, (0, 255, 0), -1)
        bgr = cv2.putText(bgr, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
    return bgr


class FunctionsAPI:

    def __init__(self, url):
        self.url = url
        self.executor = concurrent.futures.ThreadPoolExecutor()
        assert requests.get(f"{self.url}/docs").status_code == 200
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


def sv_annotate(image: np.ndarray,
                detections: sv.Detections,
                mask_opacity: float = 0.7,
                smart_label: bool = True) -> np.ndarray:
    """ :param image: OpenCV image
        :param detections: Supervision Detections with xyxy, confidence ..."""
    color_lookup = sv.ColorLookup.CLASS if detections.mask is None else sv.ColorLookup.INDEX
    anno_mask = sv.MaskAnnotator(color_lookup=color_lookup, opacity=mask_opacity)
    anno_box = sv.BoxAnnotator(color_lookup=color_lookup)
    anno_label = sv.LabelAnnotator(color_lookup=color_lookup, smart_position=smart_label)

    labels = [f"{score:.2f}" for score in detections.confidence] if detections.class_id is None else [
        f"{phrase} {score:.2f}" for phrase, score in zip(detections.metadata[detections.class_id], detections.confidence)]

    return anno_label.annotate(
        anno_box.annotate(
            anno_mask.annotate(image.copy(), detections=detections),
            detections=detections),
        detections=detections, labels=labels)
