import math
import time
import cv2
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import hand_tracking as ht


def main():
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = ht.HandTracker()

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume_range = volume.GetVolumeRange()
    min_volume = volume_range[0]
    max_volume = volume_range[1]

    while True:
        success, image = cap.read()
        image = detector.find_hands(image)
        landmark_list = detector.find_position(image, draw=False)

        if len(landmark_list) != 0:
            thumb_x, thumb_y = landmark_list[4][1], landmark_list[4][2]
            index_x, index_y = landmark_list[8][1], landmark_list[8][2]
            center_x, center_y = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2
            cv2.circle(image, (thumb_x, thumb_y), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, (index_x, index_y), 8, (0, 255, 0), cv2.FILLED)
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
            cv2.circle(image, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)
            length = math.hypot(index_x - thumb_x, index_y - thumb_y)

            if length < 20 or length > 140:
                cv2.circle(image, (center_x, center_y), 8, (0, 0, 255), cv2.FILLED)
            volume_level = np.interp(length, [20, 140], [min_volume, max_volume])
            volume_bar = np.interp(length, [20, 140], [400, 150])
            volume_percentage = np.interp(length, [20, 140], [0, 100])
            cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 2)
            cv2.rectangle(image, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, "Volume level:", (40, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(image, f"{str(int(volume_percentage))}%", (250, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

            if length < 20 or length > 140:
                cv2.rectangle(image, (50, int(volume_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(image, f"{str(int(volume_percentage))}%", (250, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            volume.SetMasterVolumeLevel(volume_level, None)

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
