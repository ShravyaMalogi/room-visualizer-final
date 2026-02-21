import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import math


def load_img(img_path):
    image = Image.open(img_path)
    data = np.asarray(image)
    return data


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(40, 30))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title(), color="black")
        plt.imshow(image)
    plt.show()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def get_wall_corners(image):
    image = image[..., ::-1]

    rgb_unique = set(
        tuple(rgb) for rgb in image.reshape(image.shape[0] * image.shape[1], 3)
    )

    result = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for color in colors:
        if color not in rgb_unique:
            continue

        mask = np.all(image == color, axis=-1)
        mask = mask.astype(np.uint8)

        img = np.copy(image)
        img[np.where(mask != 1)] = [0, 0, 0]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(contours, key=lambda x: cv2.contourArea(x))
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * perimeter, True)

        area = cv2.contourArea(approx)
        if area < (image.shape[0] * image.shape[1]) / 20:
            continue

        approx = [tuple(point[0]) for point in approx]
        points = countour_rect_corners(approx)
        result.append(points)

    return result


def countour_rect_corners(approx):
    pts = np.asarray(approx, dtype=np.float32)
    if pts.shape[0] < 4:
        raise ValueError("Not enough contour points to estimate corners")

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]

    top_left_point = pts[np.argmin(s)]
    bottom_right_point = pts[np.argmax(s)]
    top_right_point = pts[np.argmax(d)]
    bottom_left_point = pts[np.argmin(d)]

    return [
        (int(top_left_point[0]), int(top_left_point[1])),
        (int(top_right_point[0]), int(top_right_point[1])),
        (int(bottom_left_point[0]), int(bottom_left_point[1])),
        (int(bottom_right_point[0]), int(bottom_right_point[1])),
    ]


def getAngle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return ang + 360 if ang < 0 else ang


def _order_quad_points(points):
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _expand_quad(quad, image_shape, expand_ratio=1.04):
    if expand_ratio <= 1.0:
        return quad

    h, w = image_shape[:2]
    center = np.mean(quad, axis=0, keepdims=True)
    expanded = center + (quad - center) * expand_ratio

    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

    return expanded.astype(np.float32)


def _texture_for_wall(texture, wall_width, wall_height, zoom=1.25):
    zoom = max(1.0, float(zoom))
    target_w = max(32, int(wall_width))
    target_h = max(32, int(wall_height))

    if zoom <= 1.0:
        return cv2.resize(texture, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Zoom-in mode: resize larger, then take centered crop,
    # and stretch to exact wall size for full coverage.
    zoom_w = max(target_w + 1, int(target_w * zoom))
    zoom_h = max(target_h + 1, int(target_h * zoom))
    zoomed = cv2.resize(texture, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)

    start_x = max(0, (zoom_w - target_w) // 2)
    start_y = max(0, (zoom_h - target_h) // 2)
    end_x = start_x + target_w
    end_y = start_y + target_h

    crop = zoomed[start_y:end_y, start_x:end_x]
    if crop.shape[0] != target_h or crop.shape[1] != target_w:
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return crop


def map_texture(texture, image, dsts, mask, zoom=1.25, quad_expand=1.04):
    img = image.copy()
    coverage_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for dst in dsts:
        ordered_dst = _order_quad_points(dst)
        ordered_dst = _expand_quad(ordered_dst, image.shape, expand_ratio=quad_expand)

        cv2.fillConvexPoly(coverage_mask, ordered_dst.astype(np.int32), 255)

        top_w = np.linalg.norm(ordered_dst[1] - ordered_dst[0])
        bottom_w = np.linalg.norm(ordered_dst[2] - ordered_dst[3])
        left_h = np.linalg.norm(ordered_dst[3] - ordered_dst[0])
        right_h = np.linalg.norm(ordered_dst[2] - ordered_dst[1])

        wall_width = int(max(1, (top_w + bottom_w) / 2))
        wall_height = int(max(1, (left_h + right_h) / 2))

        zoomed_texture = _texture_for_wall(texture, wall_width, wall_height, zoom)

        height, width = zoomed_texture.shape[:2]

        src = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ])

        M = cv2.getPerspectiveTransform(src, ordered_dst.astype(np.float32))
        warped = cv2.warpPerspective(zoomed_texture, M, image.shape[:2][::-1])

        # ✅ FIXED SECTION — proper quad masking
        wall_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(wall_mask, ordered_dst.astype(np.int32), 255)
        wall_mask_3ch = cv2.merge([wall_mask, wall_mask, wall_mask])

        img = np.where(wall_mask_3ch == 255, warped, img)

    final = image.copy()

    if mask is not None:
        if len(mask.shape) == 3:
            mask_gray = mask[..., 0]
        else:
            mask_gray = mask

        seg_mask = (mask_gray != 0).astype(np.uint8) * 255
        effective_mask = cv2.bitwise_or(seg_mask, coverage_mask)
    else:
        effective_mask = coverage_mask

    effective_mask = cv2.dilate(
        effective_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    final[np.where(effective_mask != 0)] = img[np.where(effective_mask != 0)]

    return final


def load_texture(path, n=5, m=5):
    texture = load_img(path)
    texture = np.tile(texture, (n, m, 1))
    return texture
