# management/commands/start_attendance.py
from django.core.management.base import BaseCommand
from attendace_app.recognize_attendance import start_attendance_system

class Command(BaseCommand):
    help = 'Start face recognition for attendance'

    def handle(self, *args, **kwargs):
        start_attendance_system()
        
