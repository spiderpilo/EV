import cv2

from ev.vision.face_recognition import FaceRecognizer


def main():
    print("=" * 50)
    print("  EV — Face Enrollment")
    print("=" * 50)
    print("\nThis will capture 5 photos of your face from the webcam.")
    print("Position yourself in front of the camera with good lighting.")
    print("Press SPACE to capture, Q to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    images = []
    target = 5

    while len(images) < target:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(
            display,
            f"Captured: {len(images)}/{target} — SPACE to capture, Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("EV Face Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            images.append(frame.copy())
            print(f"  Captured {len(images)}/{target}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not images:
        print("No images captured. Enrollment cancelled.")
        return

    print(f"\nEnrolling {len(images)} images...")
    recognizer = FaceRecognizer()
    recognizer.enroll(images)
    print("Face enrollment complete!")


if __name__ == "__main__":
    main()
