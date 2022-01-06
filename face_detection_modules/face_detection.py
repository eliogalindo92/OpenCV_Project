import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self, detection_confidence=0.5):
        self.detection_confidence = detection_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection()
        self.mp_draw = mp.solutions.drawing_utils

    def find_face(self, image, draw=True):

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(rgb_image)
        bounding_box_list = []
        if self.results.detections:
            if draw:
                for identifier, detection in enumerate(self.results.detections):
                    bounding_box_position = detection.location_data.relative_bounding_box
                    height, width, channel = image.shape
                    bounding_box = int(bounding_box_position.xmin * width), int(bounding_box_position.ymin * height), \
                                   int(bounding_box_position.width * width), int(bounding_box_position.height * height)
                    bounding_box_list.append([identifier, bounding_box, detection.score])
                    if draw:
                        cv2.rectangle(image, bounding_box, (255, 0, 0), 1)
                        cv2.putText(image, f"Detection Score:{int(detection.score[0] * 100)}%", (bounding_box[0],
                                    bounding_box[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image, bounding_box_list
