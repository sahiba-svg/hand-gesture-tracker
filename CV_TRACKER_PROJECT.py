import argparse
import logging
import sys
import cv2
import mediapipe as mp

__version__ = "0.1.0"


def run(camera_index: int = 0, window_name: str = "Hand Tracker") -> int:
    """Run the hand-tracker using MediaPipe and OpenCV.

    Returns the exit code (0 on success).
    """
    logging.info("Starting hand tracker (camera=%s)", camera_index)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Configure MediaPipe Hands with reasonable defaults
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Unable to open camera index %s", camera_index)
        return 2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Empty frame captured, stopping")
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Draw hand landmarks onto the original BGR frame
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Quit key pressed, exiting")
                break
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="cv-tracker", description="Hand gesture tracker using MediaPipe + OpenCV")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--window", "-w", type=str, default="Hand Tracker", help="Window title")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    return run(camera_index=args.camera, window_name=args.window)


if __name__ == "__main__":
    raise SystemExit(main())
