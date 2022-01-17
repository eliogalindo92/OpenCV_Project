import cv2
import mediapipe as mp


class FaceMeshDetector:

    def __init__(self):
        self.results = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_specs = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def find_face_mesh(self, image, draw=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(rgb_image)
        faces_list = []
        if self.results.multi_face_landmarks:
            for face_landmark in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, face_landmark, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                self.drawing_specs, self.drawing_specs)
                face = []
                for identifier, landmark in enumerate(face_landmark.landmark):
                    height, width, channel = image.shape
                    coordinate_x, coordinate_y = int(landmark.x * width), int(landmark.y * height)
                    face.append([identifier, coordinate_x, coordinate_y])
                faces_list.append(face)

        return image, faces_list
