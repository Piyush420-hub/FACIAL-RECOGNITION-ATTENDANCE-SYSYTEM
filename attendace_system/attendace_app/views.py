from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import user_passes_test, login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User,Group
from .models import Student, Attendance, StudentProfile, TeacherProfile
from .register_face import register_face
import subprocess
from django.views.decorators.http import require_POST


def is_teacher(user):
    return hasattr(user, 'teacherprofile')  # Check if user has TeacherProfile

def is_student(user):
    return hasattr(user, 'studentprofile')  # Check if user has studentprofile

def is_teacher_or_admin(user):
    return user.groups.filter(name__in=['Teacher', 'Admin']).exists()

# Attendance view for displaying today's attendance
@login_required
def attendance_view(request):
    today_attendance = Attendance.objects.filter(timestamp__date=timezone.now().date()).select_related('student')
    return render(request, 'attendance.html', {'attendance': today_attendance})


@csrf_exempt
@user_passes_test(is_teacher_or_admin)  # Only teachers or admins can register students
def register_student(request):
    if request.method == 'POST':
        try:
            name = request.POST.get('full_name')
            roll_no = request.POST.get('roll_no')
            image_file = request.FILES.get('image')

            if not name or not roll_no or not image_file:
                return JsonResponse({'message': 'All fields are required.'}, status=400)

            # Validate if the roll_no is unique
            if User.objects.filter(username=roll_no).exists():
                return JsonResponse({'message': 'A student with this roll number already exists.'}, status=400)

            # Create a new Student object
            student = Student(full_name=name, roll_no=roll_no, image=image_file)
            student.save()

            # Register face for recognition
            image_path = student.image.path
            register_face(image_path=image_path, student_name=name)

            # Create a user account for the student
            user = User.objects.create_user(username=roll_no, password='temporarypassword123')  # Set a default/temporary password
            StudentProfile.objects.create(user=user, roll_no=roll_no, student=student)

            # Add the user to the 'Student' group
            student_group, created = Group.objects.get_or_create(name='Student')
            user.groups.add(student_group)

            return JsonResponse({"message": "Student registered successfully"}, status=201)

        except Exception as e:
            # Log the error message for debugging (optional)
            # logger.error(f"Error registering student: {str(e)}")
            return JsonResponse({'message': f"An error occurred: {str(e)}"}, status=400)
    
    # Render the registration form for GET requests
    return render(request, 'register_student.html')

# View for students to view their own attendance
@login_required
@user_passes_test(is_student)
def view_attendance(request):
    try:
        student_profile = request.user.studentprofile
        student = student_profile.student
        attendance_records = Attendance.objects.filter(student=student).order_by('-timestamp')

        attendance_data = [
            {"timestamp": record.timestamp, "status": record.status} for record in attendance_records
        ]
        return JsonResponse({"attendance": attendance_data}, status=200)
    except StudentProfile.DoesNotExist:
        return JsonResponse({"error": "Student profile not found."}, status=404)


# View for teachers to view all attendance records

@login_required
@user_passes_test(is_teacher_or_admin)
def view_all_attendance(request):
    if request.method == 'GET':
        attendance_records = Attendance.objects.all().select_related('student').order_by('-timestamp')
        attendance_data = [
            {
                "student": record.student.full_name,
                "roll_no": record.student.roll_no,
                "timestamp": record.timestamp,
                "status": record.status
            } for record in attendance_records
        ]
        return JsonResponse({"attendance": attendance_data}, status=200)
    return JsonResponse({"error": "Invalid request method"}, status=405)

# View to mark attendance (teacher access only)
@csrf_exempt
@user_passes_test(is_teacher)
def mark_attendance(request):
    if request.method == 'POST':
        roll_no = request.POST.get('roll_no')
        status = request.POST.get('status')  # Either 'Present' or 'Absent'
        student = get_object_or_404(Student, roll_no=roll_no)

        Attendance.objects.create(student=student, status=status)
        return JsonResponse({"message": "Attendance marked successfully"}, status=201)
    return JsonResponse({"error": "Invalid request method"}, status=405)

def is_admin(user):
    return User.is_superuser

@csrf_exempt
@user_passes_test(is_admin)  # Ensure only admin can access this view
def register_teacher(request):
    if request.method == 'POST':
        try:
            # Get teacher's details from the form
            full_name = request.POST.get('full_name')
            username = request.POST.get('username')  # Username for teacher login
            password = request.POST.get('password')  # Teacher's password
            
            # Create a new User object for the teacher
            user = User.objects.create_user(username=username, password=password)
            user.first_name = full_name
            user.save()

            # Create a TeacherProfile linked to the User
            TeacherProfile.objects.create(user=user)
            teacher_group, created = Group.objects.get_or_create(name='Teacher')
            user.groups.add(teacher_group)
            return JsonResponse({"message": "Teacher registered successfully"}, status=201)
        except Exception as e:
            return JsonResponse({"message": f"got an error: {e}"}, status=400) 
    # Render the registration form for GET request
    return render(request, 'register_teacher.html')

# Home Page View
def home(request):
    return render(request, 'base.html')

# Login view for Admin
def admin_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None and user.is_superuser:  # Check if user is an admin
            login(request, user)
            return redirect('admin_dashboard')
        else:
            return JsonResponse({"error": "Invalid credentials or not authorized"}, status=403)
    return render(request, 'admin_login.html')

# Login view for Teacher
def teacher_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None and hasattr(user, 'teacherprofile'):  # Check if user is a teacher
            login(request, user)
            return redirect('teacher_dashboard')
        else:
            return JsonResponse({"error": "Invalid credentials or not authorized"}, status=403)
    return render(request, 'teacher_login.html')

# Login view for Student
def student_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        print(f"Authenticated User: {user}")  # Debugging: Check if user is authenticated
        
        if user is not None:
            # Check if user has a StudentProfile
            if hasattr(user, 'studentprofile'):
                print(f"User has StudentProfile: {user.studentprofile}")  # Debugging: Check if StudentProfile exists
                login(request, user)
                return redirect('student_dashboard')
            else:
                print(f"User does not have StudentProfile.")
                return JsonResponse({"error": "User is not authorized as a student."}, status=403)
        else:
            print("Invalid credentials.")
            return JsonResponse({"error": "Invalid credentials or not authorized"}, status=403)
    
    return render(request, 'student_login.html')

@user_passes_test(is_admin)
def admin_dashboard(request):
    return render(request, 'admin_dashboard.html')

@user_passes_test(is_teacher)
def teacher_dashboard(request):
    return render(request, 'teacher_dashboard.html')

@user_passes_test(is_student)
def student_dashboard(request):
    return render(request, 'student_dashboard.html')

@require_POST
def recognize_attendance(request):
    try:
        # Run the recognize_attendance.py script as a subprocess
        process = subprocess.Popen(
            ["python", "recognize_attendance.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        # Check for errors
        if process.returncode != 0:
            return JsonResponse({"status": "error", "message": stderr.decode("utf-8")}, status=500)

        return JsonResponse({"status": "success", "message": "Attendance recognition completed successfully."})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)