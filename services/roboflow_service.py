import cv2
import numpy as np
import requests
import json
import os

from config import (
    ROBOFLOW_MODEL_ID,
    ROBOFLOW_API_KEY,
    DATA_FOLDER
)

# ---------- helper: keep only bottom connected component ----------

def keep_bottom_component(mask):
    mask = (mask > 0).astype(np.uint8)

    num, labels = cv2.connectedComponents(mask)

    if num <= 1:
        return mask * 255

    best_label = 0
    best_lowest_y = -1

    for i in range(1, num):
        ys = np.where(labels == i)[0]
        if len(ys) == 0:
            continue

        lowest_y = ys.max()

        if lowest_y > best_lowest_y:
            best_lowest_y = lowest_y
            best_label = i

    return (labels == best_label).astype(np.uint8) * 255


# ---------- roboflow api call ----------

def roboflow_inference(image_path):

    if not ROBOFLOW_MODEL_ID or not ROBOFLOW_API_KEY:
        raise RuntimeError("Roboflow env vars not set")

    url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"

    with open(image_path, "rb") as f:
        resp = requests.post(
            url,
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": 0.3,
                "overlap": 0.3
            },
            files={"file": f},
            timeout=(10, 60)
        )

    if not resp.ok:
        raise RuntimeError(resp.text)

    result = resp.json()
    
    # Flatten nested structure - predictions are under result.result.predictions
    if "result" in result and isinstance(result["result"], dict):
        result = result["result"]

    os.makedirs(DATA_FOLDER, exist_ok=True)

    try:
        with open(os.path.join(DATA_FOLDER, "last_inference.json"), "w") as fp:
            json.dump(result, fp, indent=2, default=str)
    except Exception as e:
        print(f"Failed to save inference JSON: {e}")

    return result


# ---------- build mask from roboflow predictions ----------

def build_surface_mask(result, image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image for mask build")

    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    for pred in result.get("predictions", []):

        # polygon segmentation
        pts = pred.get("points") or pred.get("polygon")
        if pts:
            poly = np.array(
                [[int(p["x"]), int(p["y"])] for p in pts],
                dtype=np.int32
            )
            cv2.fillPoly(mask, [poly], 255)
            continue

        # bbox fallback
        if all(k in pred for k in ["x", "y", "width", "height"]):
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    return mask


# ---------- main pipeline function ----------

def run_two_model_pipeline(image_path):

    try:
        result = roboflow_inference(image_path)
    except Exception as e:
        print(f"Roboflow inference failed: {e}")
        return {
            "floor_mask": None,
            "surface_result": {}
        }

    # build mask from surface predictions
    surface_mask = build_surface_mask(result, image_path)

    # remove ceiling automatically → keep only floor
    floor_mask = keep_bottom_component(surface_mask)

    return {
        "floor_mask": floor_mask,
        "surface_result": result
    }