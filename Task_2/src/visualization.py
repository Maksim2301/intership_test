import cv2
import numpy as np

def draw_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    confidences: np.ndarray = None,
    max_draw: int = 100,
    conf_thresh: float = 0.5
) -> np.ndarray:
    # Filter points based on confidence score threshold
    if confidences is not None:
        mask = confidences > conf_thresh
        pts0, pts1 = pts0[mask], pts1[mask]
        confidences = confidences[mask]
    # Randomly sample matches if count exceeds max_draw limit
    if len(pts0) > max_draw:
        indices = np.random.choice(len(pts0), max_draw, replace=False)
        pts0, pts1 = pts0[indices], pts1[indices]
    # Get dimensions for side-by-side visualization canvas
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    canvas_h = max(h0, h1)
    canvas_w = w0 + w1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Copy images onto the canvas side by side
    canvas[:h0, :w0] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if len(img0.shape) == 2 else img0
    canvas[:h1, w0:w0+w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1

    # Draw circles and matching lines between point pairs
    for pt0, pt1 in zip(pts0, pts1):
        x0, y0 = int(pt0[0]), int(pt0[1])
        x1, y1 = int(pt1[0]) + w0, int(pt1[1]) # Shift x1 coordinate by w0 for right image

        # Generate random RGB color for each line
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, (x0, y0), 4, color, -1)
        cv2.circle(canvas, (x1, y1), 4, color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

    return canvas