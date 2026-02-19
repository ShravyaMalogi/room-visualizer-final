import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_floor_corners(mask):

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 500:
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)

    return order_points(np.array(box, dtype="float32"))

def keep_bottom_component(mask):
    """
    From a mask that may contain floor + ceiling (surface),
    keep only the lowest connected component (floor).
    """

    # ensure binary
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

    floor_mask = (labels == best_label).astype(np.uint8) * 255
    return floor_mask