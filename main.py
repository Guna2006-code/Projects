#############################################
# SMART FACE ATTENDANCE SYSTEM (UI PRO FINAL)
#############################################

import tkinter as tk
from tkinter import ttk, messagebox
import cv2, os, csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime

# -------------------- FOLDERS --------------------
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)

# -------------------- RECOGNIZER --------------------
def get_recognizer():
    return cv2.face.LBPHFaceRecognizer_create()

# -------------------- SWITCH SCREENS --------------------
def show_admin():
    student_frame.pack_forget()
    admin_frame.pack(fill="both", expand=True)

def show_student():
    admin_frame.pack_forget()
    student_frame.pack(fill="both", expand=True)

# -------------------- ADMIN: CAPTURE --------------------
def capture_images():
    Id = entry_id.get()
    name = entry_name.get()

    if Id == "" or name == "":
        status.set("❌ Enter ID and Name")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            count += 1
            cv2.imwrite(f"TrainingImage/{name}.{Id}.{count}.jpg",
                        gray[y:y+h, x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Capturing Faces", img)

        if count >= 30 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    with open("StudentDetails/StudentDetails.csv","a",newline='') as f:
        csv.writer(f).writerow([Id,name])

    load_students()
    status.set(f"✅ {name} Added")

# -------------------- DELETE --------------------
def delete_student():
    selected = tree.selection()
    if not selected:
        messagebox.showwarning("Warning","Select student")
        return

    item = tree.item(selected[0])
    Id, name = item["values"]

    rows = []
    with open("StudentDetails/StudentDetails.csv") as f:
        for row in csv.reader(f):
            if row[0] != str(Id):
                rows.append(row)

    with open("StudentDetails/StudentDetails.csv","w",newline='') as f:
        csv.writer(f).writerows(rows)

    for file in os.listdir("TrainingImage"):
        if f".{Id}." in file:
            os.remove(os.path.join("TrainingImage", file))

    load_students()
    status.set(f"🗑 Deleted {name}")

# -------------------- LOAD --------------------
def load_students():
    for i in tree.get_children():
        tree.delete(i)

    if os.path.exists("StudentDetails/StudentDetails.csv"):
        with open("StudentDetails/StudentDetails.csv") as f:
            for row in csv.reader(f):
                tree.insert("", "end", values=row)

# -------------------- TRAIN --------------------
def train_model():
    recognizer = get_recognizer()

    faces, ids = [], []

    for file in os.listdir("TrainingImage"):
        path = os.path.join("TrainingImage", file)
        img = Image.open(path).convert('L')
        imgNp = np.array(img, 'uint8')

        try:
            Id = int(file.split(".")[1])
        except:
            continue

        faces.append(imgNp)
        ids.append(Id)

    if len(faces) == 0:
        status.set("❌ No Images")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainingImageLabel/model.yml")

    status.set("✅ Model Trained")

# -------------------- STUDENT ATTENDANCE (IMPROVED) --------------------
def student_attendance():
    recognizer = get_recognizer()

    if not os.path.exists("TrainingImageLabel/model.yml"):
        student_status.set("❌ Train model first")
        return

    recognizer.read("TrainingImageLabel/model.yml")

    df = pd.read_csv("StudentDetails/StudentDetails.csv", names=["ID","Name"])
    cam = cv2.VideoCapture(0)

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    recognized_count = 0
    required_frames = 10
    current_id = None

    result = "❌ Absent"

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 70:
                name = df[df["ID"]==Id]["Name"].values[0]

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img, f"{name} ({int(conf)})",
                            (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

                if current_id == Id:
                    recognized_count += 1
                else:
                    current_id = Id
                    recognized_count = 1

                if recognized_count >= required_frames:
                    now = datetime.datetime.now()
                    date = now.strftime("%d-%m-%Y")
                    time = now.strftime("%H:%M:%S")

                    with open(f"Attendance/Attendance_{date}.csv","a",newline='') as f:
                        csv.writer(f).writerow([Id,name,date,time])

                    result = f"✅ {name} Present"
                    cam.release()
                    cv2.destroyAllWindows()
                    student_status.set(result)
                    return
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img, "Unknown",
                            (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,0,255), 2)

        cv2.imshow("Recognizing... Press Q to Exit", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    student_status.set(result)

# -------------------- MAIN UI --------------------
root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("1000x650")
root.configure(bg="#0f172a")

nav = tk.Frame(root, bg="#1e293b")
nav.pack(fill="x")

tk.Button(nav, text="👨‍💼 Admin", command=show_admin,
          bg="#3b82f6", fg="white", width=15).pack(side="left", padx=10, pady=10)

tk.Button(nav, text="🎓 Student", command=show_student,
          bg="#f59e0b", fg="white", width=15).pack(side="left")

# ADMIN FRAME
admin_frame = tk.Frame(root, bg="#0f172a")

tk.Label(admin_frame, text="ADMIN PORTAL",
         font=("Arial",20,"bold"),
         bg="#0f172a", fg="#38bdf8").pack(pady=10)

tree = ttk.Treeview(admin_frame, columns=("ID","Name"), show="headings")
tree.heading("ID", text="ID")
tree.heading("Name", text="Name")
tree.pack(pady=10)

form = tk.Frame(admin_frame, bg="#1e293b")
form.pack(pady=10)

tk.Label(form, text="ID", bg="#1e293b", fg="white").grid(row=0,column=0,padx=5)
entry_id = tk.Entry(form)
entry_id.grid(row=0,column=1)

tk.Label(form, text="Name", bg="#1e293b", fg="white").grid(row=1,column=0)
entry_name = tk.Entry(form)
entry_name.grid(row=1,column=1)

tk.Button(admin_frame, text="Capture Images",
          command=capture_images,
          bg="#22c55e", fg="white", width=20).pack(pady=5)

tk.Button(admin_frame, text="Train Model",
          command=train_model,
          bg="#3b82f6", fg="white", width=20).pack(pady=5)

tk.Button(admin_frame, text="Delete Student",
          command=delete_student,
          bg="#ef4444", fg="white", width=20).pack(pady=5)

status = tk.StringVar()
tk.Label(admin_frame, textvariable=status,
         bg="#0f172a", fg="white").pack(pady=10)

# STUDENT FRAME
student_frame = tk.Frame(root, bg="#0f172a")

tk.Label(student_frame, text="STUDENT PORTAL",
         font=("Arial",20,"bold"),
         bg="#0f172a", fg="#f59e0b").pack(pady=20)

tk.Button(student_frame, text="🎥 Mark Attendance",
          command=student_attendance,
          bg="#10b981", fg="white",
          font=("Arial",14), width=25).pack(pady=20)

student_status = tk.StringVar()
tk.Label(student_frame, textvariable=student_status,
         bg="#0f172a", fg="white",
         font=("Arial",14)).pack(pady=20)

# START
load_students()
show_admin()

root.mainloop()