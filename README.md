Drowsiness Detection System
This project is a Drowsiness Detection System that uses real-time video analysis to monitor the driver's eyes and head movements, providing alerts if drowsiness is detected. It leverages computer vision, facial landmark detection, and audio notifications to enhance driver safety and prevent accidents caused by fatigue.

Features
Eye Aspect Ratio (EAR) Calculation: Monitors the eye aspect ratio to detect prolonged eye closure.
Head Pose Estimation: Analyzes head pose to check for signs of drowsiness.
Real-time Alerts: Generates audio and text-to-speech alerts when drowsiness is detected.
Blink Count Tracking: Keeps track of blink frequency to monitor driver attention.
Logging: Records drowsiness events with timestamps in a log file (drowsiness_log.txt).
Visualization: Draws facial landmarks, displays EAR, and plots EAR history using Matplotlib.
Requirements
The project requires the following Python libraries and dependencies:

OpenCV (cv2)
dlib
imutils
scipy
pygame
pyttsx3
numpy
matplotlib
Ensure you have these installed using pip:

bash
Copy code
pip install opencv-python dlib imutils scipy pygame pyttsx3 numpy matplotlib
Additionally, you need to download the shape predictor model (shape_predictor_68_face_landmarks.dat) from dlib's official website.

How to Run the Project
Clone or download the repository.

Download the shape predictor model and place it in the same directory as the code file.

Ensure you have a webcam connected to your system.

Run the code using the command:

bash
Copy code
python drowsiness_detection.py
The application will start capturing video from your webcam and analyzing it for signs of drowsiness. When drowsiness is detected, it will play an alert sound and display a warning message.

Press q to quit the application.

Additional Notes
Make sure to replace 'alarm.wav' with the path to your custom alert sound file.
You can adjust the EAR_THRESHOLD and EAR_CONSEC_FRAMES parameters to tune the sensitivity of the detection system.
Project Overview
This project combines computer vision techniques and machine learning for a real-world application in driver safety. By continuously monitoring the driver's eyes and head movements, it aims to reduce accidents caused by driver drowsiness, making roads safer for everyone.

Demo
Include screenshots or GIFs of the project in action to provide a visual understanding of the application.