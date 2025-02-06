import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ultralytics import YOLO
from PIL import Image
from .models import DetectionModel  # Use your actual model
from pathlib import Path
import os
from django.core.files import File  # Add this import

# Load YOLO models
fractmodel = YOLO("yolov8_bone_fracture.pt")  # Bone fracture detection model
mumomodel = YOLO("mumo.pt")  # Mammography detection model

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])  # Handles images & form data
def detect_bone_fracture(request):
    return process_detection(request, fractmodel)

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])  # Handles images & form data
def detect_mumograph(request):
    return process_detection(request, mumomodel)

def process_detection(request, model):
    text = request.data.get("patient_name")  # Get text input
    uploaded_image = request.FILES.get("image")

    if not uploaded_image:
        return JsonResponse({"error": "No image provided"}, status=400)

    # Convert image to OpenCV format
    try:
        pil_image = Image.open(uploaded_image)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return JsonResponse({"error": f"Invalid image file: {str(e)}"}, status=400)

    # Run YOLO inference
    results = model.predict(image, conf=0.25)

    # Access the first result object
    result = results[0]

    # Get the image with bounding boxes
    img_with_boxes = result.plot()  # This will draw bounding boxes on the image

    # Save the image with bounding boxes to the media folder
    output_image_path = Path(settings.MEDIA_ROOT) / f"generated/{text}.jpg"
    os.makedirs(output_image_path.parent, exist_ok=True)  # Create directory if it doesn't exist
    cv2.imwrite(str(output_image_path), img_with_boxes)  # Save image with bounding boxes

    # Store in Django model (store image path as URL)
    with open(output_image_path, "rb") as f:
        model_instance = DetectionModel.objects.create(
            text=text,
            uploaded_image=uploaded_image,
            generated_image=File(f, name=output_image_path.name),  # Save generated image to the database
        )

    # Define response data
    response_data = {
        "id": model_instance.id,
        "patient_name": model_instance.text,
        "uploaded_image_url": request.build_absolute_uri(model_instance.uploaded_image.url),
        "generated_image_url": request.build_absolute_uri(model_instance.generated_image.url),  # Return generated image URL
    }

    return JsonResponse(response_data, safe=False)
