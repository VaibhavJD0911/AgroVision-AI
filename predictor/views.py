from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import predict_breed


# Predictor Upload Page
def predictor_page(request):
    return render(request, "predictor.html")


# Prediction API
@csrf_exempt
def predict(request):

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)

    try:
        image = request.FILES.get("image")

        if not image:
            return JsonResponse({"error": "No image uploaded"}, status=400)

        if not image.content_type.startswith("image"):
            return JsonResponse({"error": "File must be an image"}, status=400)

        result = predict_breed(image)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({
            "error": "Prediction failed",
            "details": str(e)
        }, status=500)
