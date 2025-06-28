import cv2
import os
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# ------------------ CONFIGURATION ------------------
KNOWN_FACES_DIR = r"C:\Users\Call Me Ghost\OneDrive\Desktop\PAI Project\non_Faces"
ENCODINGS_FILE = "face_arc_encodings.pkl"
DEBUG = True
# ---------------------------------------------------

def log(message):
    if DEBUG:
        print(message)

# Initialize face analyzer
log("🚀 Initializing ArcFace model...")
face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Function to extract name from folder structure
def get_name_from_id(person_id):
    try:
        person_path = os.path.join(KNOWN_FACES_DIR, person_id)
        for filename in os.listdir(person_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return ''.join(filter(str.isalpha, os.path.splitext(filename)[0]))
    except:
        pass
    return "Unknown"

# Generate known embeddings
log("🆕 Generating new ArcFace embeddings...")
known_encodings = {}

for person_id in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_id)
    
    if os.path.isdir(person_path):
        log(f"🔹 Processing ID: {person_id}")
        encodings_list = []
        successful_count = 0
        
        for filename in os.listdir(person_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, filename)
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        log(f"⚠️ Could not read image: {img_path}")
                        continue
                    
                    # Convert to RGB for better face detection
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces with ArcFace
                    faces = face_analyzer.get(rgb_image)
                    
                    if faces:
                        # Take the face with highest detection score
                        best_face = max(faces, key=lambda x: x.det_score)
                        embedding = best_face.embedding
                        encodings_list.append(embedding)
                        successful_count += 1
                        log(f"✅ Embedded: {filename}")
                    else:
                        log(f"⚠️ No face found in: {filename}")
                except Exception as e:
                    log(f"⚠️ Error processing {img_path}: {e}")
        
        if encodings_list:
            known_encodings[person_id] = encodings_list
            log(f"✅ ID {person_id}: Successfully embedded {successful_count} images")
        else:
            log(f"⚠️ ID {person_id}: Failed to embed any faces!")

# Save the encodings to file
try:
    with open(ENCODINGS_FILE, "wb") as file:
        pickle.dump(known_encodings, file)
    log(f"✅ Saved {len(known_encodings)} encodings to {ENCODINGS_FILE}")
except Exception as e:
    log(f"⚠️ Error saving encodings: {e}")

print(f"\n🎉 Encoding generation complete! Found {len(known_encodings)} people with embeddings.")
print("You can now run realtime_face_recognition.py") 