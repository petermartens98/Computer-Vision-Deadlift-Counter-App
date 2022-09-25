
import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk

from landmarks import landmarks

window = tk.Tk()
window.geometry("480x720")
window.title("Deadlift Counter")
ck.set_appearance_mode("dark")

classLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial",20), text_color='black', padx=10)
classLabel.place(x=20, y=1)
classLabel.configure(text='Stage')

counterLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial",20), text_color='black', padx=10)
counterLabel.place(x=180, y=1)
counterLabel.configure(text='Reps')

probLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial",20), text_color='black', padx=10)
probLabel.place(x=340, y=1)
probLabel.configure(text='Prob')

classBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial",20), text_color='white', fg_color="blue")
classBox.place(x=20, y=41)
classBox.configure(text='0')

counterBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=180, y=41)
counterBox.configure(text='0')

probBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial",20), text_color='white', fg_color="blue")
probBox.place(x=340, y=41)
probBox.configure(text='0')

def reset_counter():
    global counter
    counter = 0

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, text_font=("Arial", 20),
                      text_color="white", fg_color="red")
button.place(x=180, y=600)

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)

lmain = tk.Label(frame)
lmain.place(x=0, y=0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ''

def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

    try:
        row = np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns=landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1

    except Exception as e:
        print(e)

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)

    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    counterBox.configure(text=counter)
    probBox.configure(text=bodylang_prob[bodylang_prob.argmax()])
    classBox.configure(text=current_stage)

detect()
window.mainloop()
print("Code Completed")
