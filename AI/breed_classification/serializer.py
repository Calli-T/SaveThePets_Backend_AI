from django.core import serializers
from rest_framework import serializers


class ClassifySerializer(serializers.Serializer):
    image = serializers.CharField(max_length=200)


class SimilaritySerializer(serializers.Serializer):
    post_id = serializers.IntegerField()
