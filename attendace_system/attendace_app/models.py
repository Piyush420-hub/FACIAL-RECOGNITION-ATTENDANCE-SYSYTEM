from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import Group
import uuid
# Create your models here.
# models.py

class Student(models.Model):
    full_name = models.CharField(max_length=100)
    roll_no = models.CharField(max_length=13,primary_key=True)
    # Specify the path where images will be uploaded
    image = models.ImageField(upload_to='students/')  # Uploads to MEDIA_ROOT/students/

    def __str__(self):
        return self.name

class StudentProfile(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE)
    roll_no = models.CharField(max_length=13,unique=True)
    student = models.OneToOneField('Student',on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.user.username} - {self.student.full_name}"
    
class TeacherProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    employee_id = models.CharField(max_length=20, unique=True, blank=True)

    def __str__(self):
        return f"Teacher: {self.user.username}"
    
    def save(self, *args, **kwargs):
        if not self.employee_id:
            # Generate a unique employee ID (you can customize this logic as needed)
            self.employee_id = f"EMP-{uuid.uuid4().hex[:8].upper()}"
        super().save(*args, **kwargs)

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=[('Present', 'Present'), ('Absent', 'Absent')])

    def __str__(self):
        return f'{self.student.name} - {self.status} on {self.timestamp}'

@receiver(post_save, sender=User)
def assign_user_group(sender, instance, created, **kwargs):
    if created:
        if hasattr(instance, 'studentprofile'):
            group = Group.objects.get(name='Students')
            instance.groups.add(group)
        elif hasattr(instance, 'teacherprofile'):
            group = Group.objects.get(name='Teachers')
            instance.groups.add(group)