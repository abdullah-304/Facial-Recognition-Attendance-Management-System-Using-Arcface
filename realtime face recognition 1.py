import cv2
import os
import numpy as np
import pickle
import csv
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIGURATION ------------------
KNOWN_FACES_DIR = r"C:\Users\Call Me Ghost\OneDrive\Desktop\PAI Project\non_Faces"
ENCODINGS_FILE = "face_arc_encodings.pkl"
ATTENDANCE_CSV = "attendance.csv"
SIMILARITY_THRESHOLD = 0.5
DEBUG = False
# ---------------------------------------------------

def log(message):
    if DEBUG:
        print(message)

# Initialize face analyzer
face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Extract name from folder structure
def get_name_from_id(person_id):
    try:
        person_path = os.path.join(KNOWN_FACES_DIR, person_id)
        for filename in os.listdir(person_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return ''.join(filter(str.isalpha, os.path.splitext(filename)[0]))
    except:
        pass
    return "Unknown"

# Map person IDs to names
id_to_name = {
    person_id: get_name_from_id(person_id)
    for person_id in os.listdir(KNOWN_FACES_DIR)
    if os.path.isdir(os.path.join(KNOWN_FACES_DIR, person_id))
}

# Load known face embeddings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_encodings = pickle.load(file)
else:
    print("❌ Encodings not found. Run the script to generate them first.")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Attendance setup
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Roll Number", "Name", "Timestamp"])

marked_ids = set()  # Prevent marking the same face repeatedly

print("🔴 Starting real-time face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_analyzer.get(rgb_frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        unknown_embedding = face.embedding.reshape(1, -1)

        all_similarities = []
        for person_id, encodings in known_encodings.items():
            similarities = cosine_similarity(unknown_embedding, np.array(encodings))[0]
            max_sim = np.max(similarities)
            all_similarities.append((person_id, max_sim))

        all_similarities.sort(key=lambda x: x[1], reverse=True)

        best_id = None
        best_name = "Unknown"
        if all_similarities and all_similarities[0][1] >= SIMILARITY_THRESHOLD:
            best_id = all_similarities[0][0]
            best_name = id_to_name.get(best_id, "Unknown")

        color = (0, 255, 0) if best_id else (0, 0, 255)
        label = f"{best_name}({best_id})" if best_id else "Unknown"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + 220, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mark attendance
        if best_id and best_id not in marked_ids:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(ATTENDANCE_CSV, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([best_id, best_name, now])
            marked_ids.add(best_id)
            print(f"✅ Attendance marked for: {best_name} ({best_id})")

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
