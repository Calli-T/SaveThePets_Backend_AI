from django.urls import path, include
from .views import HelloAPI, Breed_classify, Image_Similarity, POSTID


urlpatterns = [
    path("", POSTID, name='index'),
    path("classify/", Breed_classify),
    path("similarity/", Image_Similarity)
]
