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
    SINGLE_PIECE_TEXTURES,
    LOUVRE_TEXTURE
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

def apply_wall_single_texture(uid, tex_name, wall_model):

    room_path = os.path.join(IMG_FOLDER, f"{uid}_room.jpg")
    out_path = os.path.join(IMG_FOLDER, f"{uid}_textured.jpg")

    img = load_img(room_path)

    # segmentation mask
    mask = wall_segmenting(wall_model, room_path)

    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=np.uint8)

    mask = (mask > 0).astype(np.uint8)

    if np.sum(mask) < 500:
        raise RuntimeError("Wall not detected")

    # get bounding rectangle of detected wall
    ys, xs = np.where(mask > 0)

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    wall_width = x_max - x_min
    wall_height = y_max - y_min

    if wall_width <= 0 or wall_height <= 0:
        raise RuntimeError("Invalid wall size")

    # -------- LOAD MAIN TEXTURE --------

    texture_path = os.path.join(SINGLE_PIECE_TEXTURES, tex_name)
    texture = load_img(texture_path)

    tex_h, tex_w = texture.shape[:2]

    # -------- DETECT ORIENTATION --------

    if tex_h > tex_w:
        orientation = "vertical"
    elif tex_w > tex_h:
        orientation = "horizontal"
    else:
        orientation = "square"

    # -------- SCALE MAIN TEXTURE --------

    if orientation == "vertical":
        scale = wall_height / tex_h
    elif orientation == "horizontal":
        scale = wall_width / tex_w
    else:
        scale = max(wall_width / tex_w, wall_height / tex_h)

    new_w = int(tex_w * scale)
    new_h = int(tex_h * scale)

    texture_scaled = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # -------- COMPUTE PLACEMENT --------

    paste_w = min(new_w, wall_width)
    paste_h = min(new_h, wall_height)

    x_offset_src = max((new_w - wall_width) // 2, 0)
    y_offset_src = max((new_h - wall_height) // 2, 0)

    x_offset_dst = max((wall_width - new_w) // 2, 0)
    y_offset_dst = max((wall_height - new_h) // 2, 0)

    # -------- CREATE EMPTY WALL CANVAS --------

    texture_resized = np.zeros((wall_height, wall_width, 3), dtype=np.uint8)

    # -------- PLACE MAIN TEXTURE FIRST --------

    texture_resized[
        y_offset_dst:y_offset_dst + paste_h,
        x_offset_dst:x_offset_dst + paste_w
    ] = texture_scaled[
        y_offset_src:y_offset_src + paste_h,
        x_offset_src:x_offset_src + paste_w
    ]

    # -------- FILL REMAINING AREA WITH LOUVRE TILE --------

    louvre = load_img(LOUVRE_TEXTURE)
    louvre_h, louvre_w = louvre.shape[:2]

    for y in range(0, wall_height, louvre_h):
        for x in range(0, wall_width, louvre_w):

            y_end = min(y + louvre_h, wall_height)
            x_end = min(x + louvre_w, wall_width)

            region = texture_resized[y:y_end, x:x_end]

            empty_mask = np.all(region == 0, axis=2)

            louvre_crop = louvre[0:y_end - y, 0:x_end - x]

            region[empty_mask] = louvre_crop[empty_mask]

            texture_resized[y:y_end, x:x_end] = region

    # -------- APPLY TO WALL --------

    result = img.copy()

    wall_region = result[y_min:y_max, x_min:x_max]
    mask_region = mask[y_min:y_max, x_min:x_max]

    wall_region[mask_region > 0] = texture_resized[mask_region > 0]

    result[y_min:y_max, x_min:x_max] = wall_region

    # brightness correction
    result = brightness_transfer(img, result, mask)

    save_image(result, out_path)

# ---------------- FLOOR ----------------

def apply_floor_texture(uid, tex_name):

    room_path = os.path.join(IMG_FOLDER, f"{uid}_room.jpg")
    out_path = os.path.join(IMG_FOLDER, f"{uid}_textured.jpg")

    img = load_img(room_path)

    # run roboflow model
    result_dict = run_two_model_pipeline(room_path)
    mask = result_dict.get("floor_mask")
    result = result_dict.get("surface_result", {})

    if mask is None:
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
