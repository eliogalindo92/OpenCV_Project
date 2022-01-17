import cv2
import time
import face_detection as pt


def main():
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = pt.FaceDetector()

    while True:
        success, image = cap.read()
        image, bounding_box_list = detector.find_face(image)
        if len(bounding_box_list) != 0:
            print(bounding_box_list)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(image, f"FPS:{str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        if fps > 30:
            cv2.putText(image, f"FPS:{str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else:
            cv2.putText(image, f"FPS:{str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
