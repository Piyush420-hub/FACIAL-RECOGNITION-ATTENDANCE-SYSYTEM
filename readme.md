
# Face Recognition Attendance System

A Django-based face recognition attendance system that automates the process of marking attendance using a webcam and face recognition. This system identifies students and marks their attendance automatically when they are detected by the camera.

## Features

- Recognizes students using face recognition and marks their attendance.
- Stores student information and attendance records in a database.
- API to trigger attendance marking.
- Easy to set up and run locally.

## Tech Stack

- **Backend**: Django
- **Face Recognition**: `face_recognition` library
- **Database**: SQLite (default, or PostgreSQL for production)
- **Additional Libraries**: OpenCV, NumPy, and face_recognition for image processing and face encoding

## Project Structure

```plaintext
FR-attendance-system/
├── .gitignore
├── requirements.txt
├── manage.py
├── README.md
├── attendance_system/          # Main Django project folder
│   ├── settings.py             # Django settings file
│   ├── urls.py                 # Main URL routing
│   └── wsgi.py                 # WSGI entry point for deployment
├── app/                        # Django app for attendance
│   ├── models.py               # Django models (Student, Attendance)
│   ├── views.py                # Django views (including recognize_attendance API)
│   ├── urls.py                 # URL routing for app-specific endpoints
│   ├── recognize_attendance.py # Main face recognition script
└── known_faces/                # Directory to store known face encodings
    └── known_faces.dat         # Serialized encodings for face recognition
```

## API Endpoints

The Face Recognition Attendance System exposes several APIs to handle student registration, attendance marking, and retrieval. The following API endpoints are available:

### 1. **Register Student**
- **URL**: `/register_student/`
- **Method**: POST
- **Description**: Allows teachers to register students. This API will accept student details (e.g., name, roll number) and store them in the database.
  
### 2. **View Attendance (Student)**
- **URL**: `/attendance/`
- **Method**: GET
- **Description**: Allows students to view their own attendance records.

### 3. **View All Attendance (Teacher)**
- **URL**: `/attendance/all/`
- **Method**: GET
- **Description**: Allows teachers to view attendance records for all students.

### 4. **Mark Attendance (Teacher)**
- **URL**: `/mark_attendance/`
- **Method**: POST
- **Description**: Allows teachers to mark attendance for a student manually.

### 5. **Start Attendance (Face Recognition)**
- **URL**: `/start-attendance/`
- **Method**: POST
- **Description**: Starts the face recognition process, where the system uses a webcam to detect faces, recognize students, and mark their attendance automatically.

### 6. **Register Teacher**
- **URL**: `/register-teacher/`
- **Method**: POST
- **Description**: Allows an admin to register teachers into the system.

### 7. **Admin Login**
- **URL**: `/admin/login/`
- **Method**: GET/POST
- **Description**: Allows the admin to log in to the admin dashboard.

### 8. **Teacher Login**
- **URL**: `/teacher/login/`
- **Method**: GET/POST
- **Description**: Allows a teacher to log in to the teacher dashboard.

### 9. **Student Login**
- **URL**: `/student/login/`
- **Method**: GET/POST
- **Description**: Allows a student to log in to the student dashboard.

### 10. **Admin Dashboard**
- **URL**: `/admin-dashboard/`
- **Method**: GET
- **Description**: Displays the admin dashboard with an overview of the system.

### 11. **Teacher Dashboard**
- **URL**: `/teacher-dashboard/`
- **Method**: GET
- **Description**: Displays the teacher dashboard with options to manage attendance and students.

### 12. **Student Dashboard**
- **URL**: `/student-dashboard/`
- **Method**: GET
- **Description**: Displays the student dashboard with options to view attendance and profile information.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FR-attendance-system.git
cd FR-attendance-system
```

### 2. Install Dependencies

Create a virtual environment and install the required packages.

```bash
python -m venv env
source env/bin/activate   # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

### 3. Set Up the Database

Run migrations to set up the database.

```bash
python manage.py migrate
```

### 4. Load Known Faces

Save face encodings for known students in `known_faces/known_faces.dat`. You may use a separate script to encode and save faces. Here’s an example of how you could save face encodings:

```python
# save_face_encodings.py
import face_recognition
import pickle

known_faces = [("Student Name", face_encoding)]

with open("known_faces.dat", "wb") as f:
    for name, encoding in known_faces:
        pickle.dump((name, encoding), f)
```

### 5. Run the Development Server

Start the Django development server:

```bash
python manage.py runserver
```

### 6. Using the API to Mark Attendance

You can access the API endpoint to run face recognition and mark attendance:

```plaintext
POST http://localhost:8000/api/start-attendance/
```

## Usage

1. The `recognize_attendance.py` script will activate the webcam and identify faces in real-time.
2. If a face matches a known encoding, it will mark that student’s attendance in the database.

### Example Code for Face Recognition Script

Here’s a simplified version of the `recognize_attendance.py` script:

```python
import face_recognition
import cv2
import pickle
from app.models import Student, Attendance

def mark_attendance(name):
    # Fetch student record and mark attendance
    pass

# Load known faces
with open("known_faces.dat", "rb") as f:
    # Load encodings
    pass

# Open webcam and perform recognition
```

## Requirements

To ensure the project runs smoothly, make sure you have the following libraries installed:

- Django
- face_recognition
- OpenCV (cv2)
- NumPy

Add any other dependencies to `requirements.txt`.

## License

This project is licensed under the MIT License.


