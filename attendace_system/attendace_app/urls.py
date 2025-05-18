from django.urls import path
from . import views

urlpatterns = [
    path('register_student/', views.register_student, name='register_student'),  # Teacher registers a student
    # path('register_student_template/', views.register_student_template, name='register_student_template'),  # Teacher registers a student
    path('student-dashboard/attendance/', views.view_attendance, name='view_attendance'),  # Student views their attendance
    path('attendance/all/', views.view_all_attendance, name='view_all_attendance'),  # Teacher views all attendance
    path('mark_attendance/', views.mark_attendance, name='mark_attendance'),  # Teacher marks attendance
    path('', views.home, name='home'),
    path('admin/login/', views.admin_login, name='admin_login'),
    path('teacher/login/', views.teacher_login, name='teacher_login'),
    path('student/login/', views.student_login, name='student_login'),
    path('register-teacher/', views.register_teacher, name='register_teacher'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('teacher-dashboard/', views.teacher_dashboard, name='teacher_dashboard'),
    path('student-dashboard/', views.student_dashboard, name='student_dashboard'),
    path('start-attendance/', views.recognize_attendance, name='start_attendance'),
]
