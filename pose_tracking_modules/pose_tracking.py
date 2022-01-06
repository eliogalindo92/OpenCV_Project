import cv2
import mediapipe as mp


class PoseTracker:

    def __init__(self, mode=False, upBody=False, smooth=True, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, image, draw=True):

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(rgb_image)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def find_position(self, image, draw=True):

        landmark_list = []
        if self.results.pose_landmarks:
            for identifier, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = image.shape
                coordinate_x, coordinate_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([identifier, coordinate_x, coordinate_y])
                if draw:
                    cv2.circle(image, (coordinate_x, coordinate_y), 5, (50, 0, 255), cv2.FILLED)

        return landmark_list

