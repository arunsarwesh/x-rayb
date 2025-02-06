# urls.py
from django.urls import path
from .views import detect_bone_fracture, detect_mumograph

urlpatterns = [
    path('fractdetect/', detect_bone_fracture, name='frac'),
    path('mumographdetect/', detect_mumograph, name='mumo'),
]
