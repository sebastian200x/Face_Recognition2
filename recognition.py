import face_recognition
import os, sys
import cv2
import numpy as np
import math


# Function to calculate the confidence level of face recognition
def face_confidence(face_distance, face_match_threshold=0.6):
    range = 1.0 - face_match_threshold
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (
            linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
        ) * 100
        return str(round(value, 2)) + "%"


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        # Initialize lists to store face locations, encodings, and names
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True

        # Encode faces at initialization
        self.encode_faces()

    # Function to encode faces
    def encode_faces(self):
        # Loop over all images in the 'faces' directory
        for image in os.listdir("faces"):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]
        # Append face encoding and name to respective lists
        self.known_face_encodings.append(face_encoding)
        # Remove the extension from the image name before appending
        noext = os.path.splitext(image)[0]
        self.known_face_names.append(noext)
        print(self.known_face_names)

    # Function to run face recognition
    def run_recognition(self):
        # Capture video from the webcam
        video_capture = cv2.VideoCapture(0)

        # Exit if video source not found
        if not video_capture.isOpened():
            sys.exit("Video source not found...")
        # Loop over frames from the video source
        while True:
            ret, frame = video_capture.read()
            # Process every other frame to save time
            if self.process_current_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert color space from BGR to RGB
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find faces and face encodings in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations
                )

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # Compare current face with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding
                    )
                    name = "Unknown"
                    confidence = "???"

                    # Calculate face distance for the best match
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f"{name} ({confidence})")

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(
                self.face_locations, self.face_names
            ):
                # Scale face locations back up since the frame was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face and label it
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                )
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                )

            # Display the resulting image
            cv2.imshow("Face Recognition", frame)

            # Exit the loop when 's' key is pressed
            if cv2.waitKey(1) == ord("s"):
                break

        # Release the video capture when done
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create a FaceRecognition object and run the recognition
    fr = FaceRecognition()
    fr.run_recognition()
