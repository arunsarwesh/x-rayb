from rest_framework import serializers
from .models import DetectionModel

class DetectionSerializer(serializers.ModelSerializer):
    uploaded_image_url = serializers.SerializerMethodField()
    generated_image_url = serializers.SerializerMethodField()

    class Meta:
        model = DetectionModel
        fields = ["text", "date", "uploaded_image_url", "generated_image_url"]

    def get_uploaded_image_url(self, obj):
        return self.context["request"].build_absolute_uri(obj.uploaded_image.url)

    def get_generated_image_url(self, obj):
        return self.context["request"].build_absolute_uri("/media/" + obj.generated_image)
