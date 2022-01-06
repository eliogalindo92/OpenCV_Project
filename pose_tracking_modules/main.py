import cv2
import time
import pose_tracking as pt


def main():
    current_time = 0
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = pt.PoseTracker()

    while True:
        success, image = cap.read()
        image = detector.find_pose(image)
        landmark_list = detector.find_position(image)
        if len(landmark_list) != 0:
            print(landmark_list)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(image, f"FPS:{str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
