import cv2
import numpy as np
import face_recognition
import pickle
from .models import Student, Attendance
from django.utils import timezone


def load_known_faces(file_path="known_faces.dat"):
    known_face_encodings = []
    known_face_names = []

    try:
        with open(file_path, "rb") as f:
            while True:
                try:
                    name, encoding = pickle.load(f)
                    known_face_names.append(name)
                    known_face_encodings.append(encoding)
                except EOFError:
                    break
    except FileNotFoundError:
        print("⚠️ known_faces.dat not found.")
        return [], []

    return known_face_encodings, known_face_names


def get_frame(video_capture):
    ret, frame = video_capture.read()
    if not ret:
        raise RuntimeError("Failed to grab frame from webcam.")
    return frame


def preprocess_frame(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    return np.ascontiguousarray(small_frame[:, :, ::-1])  # BGR to RGB safely


def detect_and_recognize_faces(rgb_frame, known_encodings, known_names):
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')  # you can use 'cnn' if GPU available
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        face_names.append(name)

    return face_locations, face_names


def mark_attendance(name):
    try:
        student = Student.objects.get(full_name=name)
        _, created = Attendance.objects.get_or_create(
            student=student,
            timestamp__date=timezone.now().date(),
            defaults={'status': 'Present'}
        )
        if created:
            print(f"✅ {name}'s attendance marked.")
    except Student.DoesNotExist:
        print(f"⚠️ Student '{name}' not found in database.")


def draw_labels(frame, locations, names):
    for (top, right, bottom, left), name in zip(locations, names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def start_attendance_system():
    import ipdb; ipdb.set_trace()
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("❌ No known faces found. Exiting...")
        return

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("❌ Could not access webcam.")
        return

    print("🎥 Face recognition started. Press 'q' to quit.")

    while True:
        try:
            frame = get_frame(video_capture)
            rgb_frame = preprocess_frame(frame)
            locations, names = detect_and_recognize_faces(rgb_frame, known_encodings, known_names)

            for name in names:
                if name != "Unknown":
                    mark_attendance(name)

            draw_labels(frame, locations, names)
            cv2.imshow('Face Attendance', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting attendance system.")
                break

        except Exception as e:
            print(f"🚨 Error: {e}")
            break

    video_capture.release()
    cv2.destroyAllWindows()
