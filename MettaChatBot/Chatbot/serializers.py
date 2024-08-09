from rest_framework.serializers import ModelSerializer
from .models import ChatMessage, ChatSession


class SessionSerializer(ModelSerializer):
    class Meta:
        model = ChatSession
        fields = '__all__'


class MesssageSerializer(ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = '__all__'
