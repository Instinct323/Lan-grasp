import concurrent.futures
import json
import logging
import pickle
import re
import time

import cv2
import numpy as np
import requests
import supervision as sv

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


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


class JSONprompter(dict):

    def prompt(self):
        return ("\nYour response should be formatted as a JSON code block, without any additional text. "
                "The required fields are:\n" + "\n".join(f"- {k}: {v}" for k, v in self.items()))

    def decode(self,
               response: str):
        content = re.search(r"\{.*}", response, flags=re.S)
        try:
            return json.loads(content.group(0))
        except:
            raise ValueError(f"Invalid JSON response: {response}")


class GridAnnotator:

    def __init__(self,
                 ngrid: int,
                 color: tuple = (255, 255, 255),
                 thickness: float = 5e-3):
        self.ngrid = ngrid
        self.color = color
        self.thickness = thickness

    def make_grid(self, w, h):
        nr = round(np.sqrt(self.ngrid / h * w))
        nc = round(np.sqrt(self.ngrid * h / w))
        rows = np.round(np.linspace(0, h - 1, nr + 1)).astype(int)
        cols = np.round(np.linspace(0, w - 1, nc + 1)).astype(int)
        return dict(rows=rows, cols=cols, ngrid=nr * nc, shape=(nr, nc))

    def annotate(self,
                 image: np.ndarray,
                 grid_info: dict):
        image = image.copy()
        h, w = image.shape[:2]
        thickness = max(1, round(min(h, w) * self.thickness))
        for r in grid_info["rows"]: cv2.line(image, (0, r), (w, r), self.color, thickness)
        for c in grid_info["cols"]: cv2.line(image, (c, 0), (c, h), self.color, thickness)
        return image

    def index_grid(self,
                   grid_id: int,
                   grid_info: dict):
        if grid_id >= grid_info["ngrid"] or grid_id < 0: return None
        nr, nc = grid_info["shape"]
        r, c = grid_id // nc, grid_id % nc
        rows, cols = grid_info["rows"], grid_info["cols"]
        return np.array([cols[c], rows[r], cols[c + 1], rows[r + 1]])


class Pinhole:

    def __init__(self, w, h, fx, fy, cx, cy):
        self.size = w, h
        self.intrinsic = np.array([fx, fy, cx, cy])
        # cache for unprojection
        self.__coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
        self.__unproj = (self.__coords - [cx, cy]) / [fx, fy]

    def unproj(self,
               depth: np.ndarray):
        assert depth.ndim == 2, f"Depth map should be 2D, but got {depth.ndim}D"
        pcd = np.repeat(depth[..., None], 3, axis=-1)
        pcd[..., :2] *= self.__unproj
        return pcd


class FunctionsAPI:

    def __init__(self,
                 url: str = None,
                 functions: dict = None):
        self.url = url
        self.functions = functions
        self.executor = concurrent.futures.ThreadPoolExecutor()

        assert self.url or self.functions, "Please provide either a URL or a function dictionary."
        if self.url:
            assert requests.get(f"{self.url}/docs").status_code == 200
            LOGGER.info(f"See {self.url}/docs for API documentation.")

    def invoke(self, func, *args, **kwargs):
        if self.functions: return self.functions[func](*args, **kwargs)

        data = {"func": func, "args": args, "kwargs": kwargs}
        response = requests.post(f"{self.url}/invoke", data=pickle.dumps(data), headers={"t-send": str(int(time.time()))})
        assert response.status_code == 200, f"{response.status_code}, {response.text}"

        headers = response.headers
        LOGGER.info(f"[{func}] {headers['cost']}, recv:{time.time() - float(headers['t-send']):.3f}s")
        return pickle.loads(response.content)

    def invoke_async(self, func, *args, **kwargs):
        return self.executor.submit(self.invoke, func, *args, **kwargs)
