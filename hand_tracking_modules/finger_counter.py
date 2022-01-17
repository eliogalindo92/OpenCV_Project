import time
import cv2
import hand_tracking as ht


def main():
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = ht.HandTracker()

    # Fingertip ids for each hand
    tips_id = [4, 8, 12, 16, 20]

    while True:
        success, image = cap.read()
        image = detector.find_hands(image)
        landmark_list = detector.find_position(image, draw=False)

        if len(landmark_list) != 0:
            if landmark_list[3][1] < landmark_list[17][1]:
                left_hand_counter()
            else:
                right_hand_counter()

        # Counter method for the right-hand
        def right_hand_counter():
            # Right thumb
            right_hand_fingers = []
            if landmark_list[tips_id[0]][1] > landmark_list[tips_id[0] - 1][1]:
                right_hand_fingers.append(1)
            else:
                right_hand_fingers.append(0)
            # Four left_hand_fingers
            for identifier in range(1, 5):
                if landmark_list[tips_id[identifier]][2] < landmark_list[tips_id[identifier] - 2][2]:
                    right_hand_fingers.append(1)
                else:
                    right_hand_fingers.append(0)

            finger_number = right_hand_fingers.count(1)
            cv2.putText(image, "Fingers: ", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if finger_number == 0:
                cv2.putText(image, str(finger_number), (150, 82), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, str(finger_number), (150, 82), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Counter method for the left-hand
        def left_hand_counter():
            # Left thumb
            left_hand_fingers = []
            if landmark_list[tips_id[0]][1] < landmark_list[tips_id[0] - 1][1]:
                left_hand_fingers.append(1)
            else:
                left_hand_fingers.append(0)
            # Four left_hand_fingers
            for identifier in range(1, 5):
                if landmark_list[tips_id[identifier]][2] < landmark_list[tips_id[identifier] - 2][2]:
                    left_hand_fingers.append(1)
                else:
                    left_hand_fingers.append(0)

            finger_number = left_hand_fingers.count(1)
            cv2.putText(image, "Fingers: ", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if finger_number == 0:
                cv2.putText(image, str(finger_number), (150, 82), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, str(finger_number), (150, 82), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
