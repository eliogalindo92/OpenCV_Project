import cv2
import time
import hand_tracking as ht


def main():
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = ht.HandTracker()

    while True:
        success, image = cap.read()
        image, hands_list = detector.find_2_hands(image, draw_connectors=True, draw_circles=False)

        if len(hands_list) != 0:
            if len(hands_list) == 2:
                thumb_x_2, thumb_y_2 = hands_list[1][4][1], hands_list[1][4][2]
                index_x_2, index_y_2 = hands_list[1][8][1], hands_list[1][8][2]
                center_x, center_y = (thumb_x_2 + index_x_2) // 2, (thumb_y_2 + index_y_2) // 2
                cv2.circle(image, (thumb_x_2, thumb_y_2), 8, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, (index_x_2, index_y_2), 8, (0, 255, 0), cv2.FILLED)
                cv2.line(image, (thumb_x_2, thumb_y_2), (index_x_2, index_y_2), (0, 255, 0), 2)
                cv2.circle(image, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)

            thumb_x_1, thumb_y_1 = hands_list[0][4][1], hands_list[0][4][2]
            index_x_1, index_y_1 = hands_list[0][8][1], hands_list[0][8][2]
            center_x, center_y = (thumb_x_1 + index_x_1) // 2, (thumb_y_1 + index_y_1) // 2
            cv2.circle(image, (thumb_x_1, thumb_y_1), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, (index_x_1, index_y_1), 8, (0, 255, 0), cv2.FILLED)
            cv2.line(image, (thumb_x_1, thumb_y_1), (index_x_1, index_y_1), (0, 255, 0), 2)
            cv2.circle(image, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)

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
