import cv2

for i in range(5):  # Check indices 0-4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} found")
        cap.release()
    else:
        print(f"Camera {i} not available")
