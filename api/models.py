from django.db import models

class DetectionModel(models.Model):
    text = models.CharField(max_length = 200)
    date = models.DateTimeField(auto_now_add=True)
    uploaded_image = models.ImageField(upload_to='upload/')
    generated_image = models.ImageField(upload_to='generated/')

