# register_face.py (can be run via Django management command)
import face_recognition
import os
import pickle
from django.conf import settings
from attendace_app.models import Student

def register_face(image_path, student_name):
    # Load image and encode the face
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]

        # Save encoding and name to known_faces.dat
        with open("known_faces.dat", "ab") as f:
            pickle.dump((student_name, face_encoding), f)

        # Save the embedding in the Student model
        student, created = Student.objects.get_or_create(full_name=student_name)
        student.face_embedding = pickle.dumps(face_encoding)
        student.save()
        
        print(f"{student_name}'s face registered.")
    else:
        print("No face found in the image.")

# Example usage: register_face("path_to_image.jpg", "John Doe")
