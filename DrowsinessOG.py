import cv2
import dlib
import time
from scipy.spatial import distance
from imutils import face_utils
import pygame  # For sound
import pyttsx3  # For text-to-speech alerts
import matplotlib.pyplot as plt
import numpy as np

# Initialize pygame mixer for sound
pygame.mixer.init()

# Text-to-Speech engine initialization
engine = pyttsx3.init()


# Define the eye aspect ratio (EAR) threshold for drowsiness detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Function to calculate head pose
def get_head_pose(shape):
    image_pts = np.float32([shape[30],  # Nose tip
                            shape[8],  # Chin
                            shape[36],  # Left eye left corner
                            shape[45],  # Right eye right corner
                            shape[48],  # Left mouth corner
                            shape[54]])  # Right mouth corner

    model_pts = np.float32([[0.0, 0.0, 0.0],  # Nose tip
                            [0.0, -330.0, -65.0],  # Chin
                            [-225.0, 170.0, -135.0],  # Left eye left corner
                            [225.0, 170.0, -135.0],  # Right eye right corner
                            [-150.0, -150.0, -125.0],  # Left mouth corner
                            [150.0, -150.0, -125.0]])  # Right mouth corner

    focal_length = 1.0 * frame.shape[1]
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_pts, image_pts, camera_matrix, dist_coeffs)

    return rotation_vector, translation_vector


# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Eye landmarks indexes based on dlib's 68 point model
(lStart, lEnd) = (42, 48)  # Left eye
(rStart, rEnd) = (36, 42)  # Right eye

EAR_THRESHOLD = 0.25  # Threshold for EAR to trigger alert
EAR_CONSEC_FRAMES = 48  # Number of consecutive frames the EAR must be below the threshold

frame_counter = 0  # Initialize the frame counter for drowsiness detection
blink_count = 0  # Blink count

# Load the alert sound file
pygame.mixer.music.load('alarm.wav')  # Replace 'alarm.wav' with the path to your sound file

# Prepare EAR history for plotting
ear_history = []

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture each frame from the camera
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    rects = detector(gray, 0)  # Detect faces in the grayscale frame

    # Loop over the detected faces
    for rect in rects:
        shape = predictor(gray, rect)  # Determine the facial landmarks for the face region
        shape = face_utils.shape_to_np(shape)  # Convert the landmark (x, y)-coordinates to a NumPy array

        # Extract the left and right eye coordinates
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        # Compute the EAR for both eyes
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Append EAR to history
        ear_history.append(ear)

        # Visualize the landmarks (optional)
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 2)

        # Check if the EAR is below the threshold, and if so, increment the frame counter
        if ear < EAR_THRESHOLD:
            frame_counter += 1

            # If eyes were closed for a sufficient number of frames, trigger the alert sound
            if frame_counter >= EAR_CONSEC_FRAMES:
                if not pygame.mixer.music.get_busy():  # Check if the sound is not already playing
                    pygame.mixer.music.play()  # Play the alert sound
                    engine.say("Drowsiness detected! Please wake up.")  # Text-to-speech alert
                    engine.runAndWait()

                # Record drowsiness event
                with open("drowsiness_log.txt", "a") as f:
                    f.write(f"Drowsiness detected at: {time.ctime()} with EAR: {ear:.2f}\n")

                # Display "Drowsiness detected !!" in red
                cv2.putText(frame, "DROWSINESS DETECTED !!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            if frame_counter > 1:
                blink_count += 1  # Increment blink count
            frame_counter = 0  # Reset the frame counter if EAR is above the threshold

        # Display EAR score and blink count
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Blink Count: {blink_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display the frame to the screen
    cv2.imshow("Frame", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Stop any playing sound when the program ends
pygame.mixer.music.stop()

# Plot EAR history
plt.figure()
plt.plot(ear_history, label="EAR")
plt.axhline(y=EAR_THRESHOLD, color='r', linestyle='--', label="Threshold")
plt.title("EAR History")
plt.xlabel("Frames")
plt.ylabel("Eye Aspect Ratio (EAR)")
plt.legend()
plt.show()
