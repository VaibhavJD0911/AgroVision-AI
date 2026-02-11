from django.urls import path
from .views import predictor_page, predict

app_name = "predictor"

urlpatterns = [
    path("", predictor_page, name="home"),        # /predictor/
    path("predict/", predict, name="predict"),    # /predictor/predict/
]
