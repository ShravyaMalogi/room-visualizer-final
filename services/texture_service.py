import os
import cv2
import numpy as np

from room_processing import load_img, brightness_transfer, save_image
from texture_mapping import (
    get_wall_corners,
    map_texture,
    load_texture
)

from wall_segmentation.segmentation import wall_segmenting
from wall_estimation.estimation import wall_estimation

# ✅ two-model roboflow pipeline
from services.roboflow_service import run_two_model_pipeline

from utils.geometry import get_floor_corners, keep_bottom_component

from config import (
    IMG_FOLDER,
    WALL_TEXTURES,
    FLOOR_TEXTURES,
    SINGLE_PIECE_TEXTURES
)


# ---------------- WALL ----------------

def apply_wall_texture(uid, tex_name, wall_model):

    room_path = os.path.join(IMG_FOLDER, f"{uid}_room.jpg")
    out_path = os.path.join(IMG_FOLDER, f"{uid}_textured.jpg")

    img = load_img(room_path)

    mask = wall_segmenting(wall_model, room_path)
    estimation_map = wall_estimation(room_path)

    corners = get_wall_corners(estimation_map)
    if corners is None:
        raise RuntimeError("Wall corners not found")

    texture_path = os.path.join(WALL_TEXTURES, tex_name)
    texture = load_texture(texture_path, 6, 6)

    textured = map_texture(texture, img, np.array(corners), mask)
    out = brightness_transfer(img, textured, mask)

    save_image(out, out_path)

def apply_wall_single_texture(uid, tex_name, wall_model, scale=0.45):

    room_path = os.path.join(IMG_FOLDER, f"{uid}_room.jpg")
    out_path = os.path.join(IMG_FOLDER, f"{uid}_textured.jpg")

    img = load_img(room_path)

    mask = wall_segmenting(wall_model, room_path)

    # ✅ SAFE mask normalization (no torch→numpy bridge)
    if not isinstance(mask, np.ndarray):
        try:
            mask = np.array(mask.tolist(), dtype=np.uint8)
        except Exception:
            mask = np.array(mask, dtype=np.uint8)

    mask = (mask > 0).astype(np.uint8) * 255

    if np.sum(mask) < 500:
        raise RuntimeError("Wall not detected")

    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)

    piece_w = max(1, int(box_w * scale))
    piece_h = max(1, int(box_h * scale))

    cx = x0 + box_w // 2
    cy = y0 + box_h // 2

    px0 = max(0, cx - piece_w // 2)
    py0 = max(0, cy - piece_h // 2)
    px1 = min(img.shape[1], px0 + piece_w)
    py1 = min(img.shape[0], py0 + piece_h)

    texture_path = os.path.join(SINGLE_PIECE_TEXTURES, tex_name)
    texture = load_img(texture_path)
    texture = cv2.resize(texture, (px1 - px0, py1 - py0))

    overlay = img.copy()
    overlay[py0:py1, px0:px1] = texture

    patch_mask = np.zeros(mask.shape, dtype=np.uint8)
    patch_mask[py0:py1, px0:px1] = 255
    patch_mask = cv2.bitwise_and(patch_mask, mask)

    out = img.copy()
    out[patch_mask > 0] = overlay[patch_mask > 0]
    out = brightness_transfer(img, out, patch_mask)

    save_image(out, out_path)
   

# ---------------- FLOOR ----------------

def apply_floor_texture(uid, tex_name):

    room_path = os.path.join(IMG_FOLDER, f"{uid}_room.jpg")
    out_path = os.path.join(IMG_FOLDER, f"{uid}_textured.jpg")

    img = load_img(room_path)

    # run roboflow model
    result_dict = run_two_model_pipeline(room_path)
    result = result_dict.get("surface_result", {})
    
    # Extract detections from result - ONLY SURFACE
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for pred in result.get("predictions", []):
        pts = pred.get("points") or pred.get("polygon")
        
        if pts:
            poly = np.array(
                [[int(p["x"]), int(p["y"])] for p in pts],
                dtype=np.int32
            )
            cv2.fillPoly(mask, [poly], 255)
        
        elif all(k in pred for k in ["x", "y", "width", "height"]):
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    # Keep only the bottom-most component (floor, not walls/ceiling)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    num, labels = cv2.connectedComponents(mask)
    if num > 1:
        best_label = 0
        best_lowest_y = -1
        for i in range(1, num):
            ys = np.where(labels == i)[0]
            if len(ys) > 0:
                lowest_y = ys.max()
                if lowest_y > best_lowest_y:
                    best_lowest_y = lowest_y
                    best_label = i
        mask = (labels == best_label).astype(np.uint8) * 255
    
    # Keep logging minimal and avoid dumping coordinates to stdout.

    if mask is None or np.sum(mask) < 100:
        raise RuntimeError("No surface detected in the image")

    texture_path = os.path.join(FLOOR_TEXTURES, tex_name)
    texture = load_img(texture_path)

    H, W = img.shape[:2]
    tile_h, tile_w = texture.shape[:2]

    # create tiled texture
    tiles_y = int(np.ceil(H / tile_h)) + 2
    tiles_x = int(np.ceil(W / tile_w)) + 2

    tiled = np.tile(texture, (tiles_y, tiles_x, 1))
    tiled = tiled[:H + tile_h, :W + tile_w]

    textured = img.copy()

    # perspective mapping
    corners = get_floor_corners(mask)

    if corners is not None and len(corners) == 4:
        try:
            src = np.array(
                [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                dtype="float32"
            )

            dst = corners.astype("float32")

            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(tiled, M, (W, H))

            textured[mask > 127] = warped[mask > 127]

        except Exception as e:
            print("Perspective warp failed — fallback tiling:", e)
            textured[mask > 127] = tiled[:H, :W][mask > 127]

    else:
        textured[mask > 127] = tiled[:H, :W][mask > 127]

    # Skip brightness_transfer since it applies the mask
    # Just save the textured image directly
    save_image(textured, out_path)

    if result:
        print("Model predictions:", len(result.get("predictions", [])))