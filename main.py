import cv2
import mediapipe as mp
import numpy as np
import random
import threading
from screeninfo import get_monitors

BACKGROUND_COLOR = (40, 40, 40)
PERSON_COLOR = (0, 0, 0)
EDGE_COLOR = (130, 130, 130)
GHOST_TINT = (0, 0, 0)

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def style_silhouette(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    condition = mask > 0.5
    output = np.ones_like(frame, dtype=np.uint8) * BACKGROUND_COLOR[0]
    output[condition] = PERSON_COLOR
    return output


def style_pixelated(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    condition = mask > 0.5
    output = np.ones_like(frame, dtype=np.uint8) * BACKGROUND_COLOR[0]
    output[condition] = PERSON_COLOR
    small = cv2.resize(output, (80, 60), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def style_outline(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    edges = cv2.Canny(mask_binary, 100, 200)
    output = np.ones_like(frame, dtype=np.uint8) * BACKGROUND_COLOR[0]
    output[edges > 0] = EDGE_COLOR
    return output


def style_ghost(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ghostly background (gray fade) with people blocked out."""
    condition = mask > 0.5

    frame[condition] = PERSON_COLOR

    return frame


def style_inverted(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    condition = mask > 0.5
    output = np.ones_like(frame, dtype=np.uint8) * PERSON_COLOR[0]
    output[condition] = BACKGROUND_COLOR
    return output


STYLES = [style_silhouette, style_pixelated, style_outline, style_ghost, style_inverted]

current_style = [random.choice(STYLES)]

def display_on_monitor(monitor):
    window_name = f"Art Wall - Monitor {monitor.name}"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, monitor.x, monitor.y)

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segment.process(rgb_frame)
        mask = result.segmentation_mask

        style_fn = current_style[0]
        output = style_fn(frame, mask)
        resized = cv2.resize(output, (monitor.width, monitor.height))
        cv2.imshow(window_name, resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            new_style = random.choice(STYLES)
            current_style[0] = new_style
            print(f"Changed to {new_style.__name__}")

    cap.release()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    monitors = get_monitors()
    print(f"Detected {len(monitors)} monitor(s).")

    threads = []
    for monitor in monitors:
        print(f"Launching on {monitor.name}")
        t = threading.Thread(target=display_on_monitor, args=(monitor,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
