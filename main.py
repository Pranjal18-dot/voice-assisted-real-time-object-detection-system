import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3 
import sounddevice as sd 
import speech_recognition as sr  
from scipy.io.wavfile import write
import tempfile
import os
import time


engine = pyttsx3.init()
engine.setProperty('rate', 150)


model = YOLO("gpModel.pt")


object_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
}


CONFIDENCE_THRESHOLD = 0.5

EXCLUDED_CLASSES = ["skateboard"]


spoken_objects = set()
pause = False
cap = cv2.VideoCapture("car_night.mp4")

def compute_distance(box, width_of_frame, avg_sizes, label):
    """Calculates the distance of the object based on its size and frame width."""
    obj_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()
    
    if label in avg_sizes:
        obj_width *= avg_sizes[label]["width_ratio"]
    
    distance = (width_of_frame * 0.5) / np.tan(np.radians(70 / 2)) / (obj_width + 1e-6)
    return round(distance, 2)

def get_direction(x_center, frame_width):
    """Returns the direction based on the object's X-coordinate (left, center, right)."""
    if x_center < frame_width / 3:
        return "left"
    elif x_center > 2 * frame_width / 3:
        return "right"
    else:
        return "center"

def speak_object(label, distance, direction):
    """Speaks out the detected object's label, its distance, and direction."""
    if label not in spoken_objects:
        engine.say(f"Detected {label} at {distance} meters to the {direction}.")
        engine.runAndWait()
        spoken_objects.add(label)

def speak_welcome():
    """Speaks the welcome message."""
    engine.say("Welcome to the object detection system. Please say 'START' to begin.")
    engine.runAndWait()

def speak_start():
    """Speaks the starting message."""
    engine.say("Starting object detection.")
    engine.runAndWait()

def record_audio():
    """Records audio from the microphone using sounddevice."""
    fs = 16000  
    duration = 5  
    
    
    print("Listening for 'START' command...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  
    
    
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        write(tmpfile.name, fs, audio_data)
        audio_file_path = tmpfile.name
    
    return audio_file_path

def listen_for_start():
    """Listens for the 'START' voice command."""
    audio_file_path = record_audio()
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Heard: {command}")
        if "start" in command:
            return True
        else:
            return False
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Please say 'START' to begin.")
        return False
    except sr.RequestError:
        print("Sorry, there was an error with the speech recognition service.")
        return False
    finally:
        
        os.remove(audio_file_path)

def process_frame(frame):
    """Processes the video frame for object detection and voice output."""
    results = model.predict(frame)
    result = results[0]  
    
    closest_object = None
    minimum_distance = float('inf')

    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        
        
        if label in EXCLUDED_CLASSES:
            continue

        coordinates = [round(x) for x in box.xyxy[0].tolist()]
        confidence = box.conf[0].item()
        
        
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        distance = compute_distance(box, frame.shape[1], object_avg_sizes, label)

        if distance < minimum_distance:
            minimum_distance = distance
            closest_object = (label, round(distance, 1), coordinates)

        
        x_center = (coordinates[0] + coordinates[2]) // 2
        direction = get_direction(x_center, frame.shape[1])

        
        if label == "person":
            cv2.rectangle(frame, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 255, 0), 2)
        elif label == "car":
            cv2.rectangle(frame, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 255, 255), 2)
        elif label in object_avg_sizes:
            cv2.rectangle(frame, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (255, 0, 0), 2)

        
        if label not in spoken_objects:
            speak_object(label, distance, direction)
            spoken_objects.add(label) 

        cv2.putText(frame, f"{label} - {distance:.1f}m", (coordinates[0], coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


frame_counter = 0 
speak_welcome()


while not listen_for_start():
    pass

speak_start()

while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        pause = not pause

cap.release()
cv2.destroyAllWindows()