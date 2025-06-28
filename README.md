**# Facial-Recognition-Attendance-Management-System-Using-Arcface**
The Facial Recognition Attendance Management System is an AI-powered solution that automates the attendance marking process using live camera input. This project uses ArcFace (via InsightFace) for face recognition and attendance marking.  
**Repository:** [https://github.com/abdullah-304/Facial-Recognition-Attendance-Management-System-Using-Arcface].

------

## Prerequisites

- **Python 3.11.9** (other 3.11.x versions should work)
- **pip** (Python package manager)
- **Git** (for cloning and version control)
- **InsightFace** (ArcFace, installed via pre-built wheel)
- **OpenCV**
- **Virtual Environment** (recommended)
- **Git LFS** (for large model files like `best.pt`)

------

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/abdullah-304/Facial-Recognition-Attendance-Management-System-Using-Arcface.git
cd https://github.com/abdullah-304/Facial-Recognition-Attendance-Management-System-Using-Arcface
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install dependencies

#### a. Install from requirements.txt

```bash
pip install -r requirements.txt
```

#### b. Install InsightFace (ArcFace) using the pre-built wheel

Replace the path with your actual wheel file location if different:

```bash
# Example (adjust path if needed)
& d:/face_recognition/.venv/Scripts/python.exe -m pip install "D:\Download\insightface-0.7.3-cp311-cp311-win_amd64.whl"
```

#### c. If you don't have requirements.txt, install manually:

```bash
pip install ultralytics opencv-python numpy scikit-learn
```

---

## File & Folder Setup

- Place your known faces in subfolders under `D:\face\non_face` (or change the path in the script):
  - Each subfolder should be named with a unique ID (e.g., roll number) and contain images of that person.

---

## Usage

1. **Run the first script to create encodings/embeddings `generate_encodings.py`**

2. **Then run the second script `realtime_face_recognition.py`**

```bash
python realtime_face_recognition.py
```

- The script will:
  - Recognize faces using ArcFace (InsightFace)
  - Draw bounding boxes and labels on the image
  - Log recognized faces to `attendance.csv`

---

## Example Directory Structure
in Non_face folder folder name must be id and inside u have the images such as Alisa1.jpg , Alisa2.jpg like and same goes for all folders 
but folder name must be ID (015 , 344, like that)

```
project/
├── requirement.txt
├── realtime_face_recognition.py
├── README.md
├── generate_encoding.py
└── D:/face/non_face/
    ├── 12345/        
    │   ├── img1.jpg
    │   └── img2.jpg
    └── 67890/
        └── img1.jpg
```

------

## Troubleshooting

- **InsightFace install issues:**  
  Use the provided wheel for Python 3.11.9 as shown above.
- **OpenCV errors:**  
  Ensure your Python and OpenCV versions are compatible.
- **Large files on GitHub:**  
  Use Git LFS for model files.

---
if you face any issue do hit me up on my 
LinkedIn: [https://www.linkedin.com/in/muhammad-abdullah-225b05317/]

For more, see the [GitHub repository](https://github.com/abdullah-304/Facial-Recognition-Attendance-Management-System-Using-Arcface). 
