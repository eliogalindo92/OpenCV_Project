import cv2
import mediapipe as mp


class HandTracker:

    def __init__(self):
        self.results = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_image)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def find_position(self, image, hands_number=0, draw=True):

        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hands_number]
            for identifier, landmark in enumerate(hand.landmark):
                height, width, channel = image.shape
                coordinate_x, coordinate_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([identifier, coordinate_x, coordinate_y])
                if draw:
                    cv2.circle(image, (coordinate_x, coordinate_y), 8, (0, 255, 0), cv2.FILLED)

        return landmark_list

    def find_2_hands(self, image, draw_connectors=True, draw_circles=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_image)
        hands_list = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw_connectors:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for identifier, landmark in enumerate(hand_landmarks.landmark):
                    height, width, channel = image.shape
                    coordinate_x, coordinate_y = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([identifier, coordinate_x, coordinate_y])
                    if draw_circles:
                        cv2.circle(image, (coordinate_x, coordinate_y), 8, (0, 255, 0), cv2.FILLED)
                hands_list.append(landmark_list)
        return image, hands_list
