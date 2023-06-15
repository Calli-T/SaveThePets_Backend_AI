from django.urls import path, include
from .views import HelloAPI, Breed_classify, Image_Similarity


urlpatterns = [
    path("hello/", HelloAPI),
    path("classify/", Breed_classify),
    path("similarity/", Image_Similarity)
]
