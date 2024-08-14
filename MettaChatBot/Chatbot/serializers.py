from rest_framework.serializers import ModelSerializer
from .models import ChatMessage, ChatSession


class SessionSerializer(ModelSerializer):
    class Meta:
        model = ChatSession
        # Include other fields as necessary
        fields = ['id', 'user', 'session_name']
        read_only_fields = ['user']


class MessageSerializer(ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = '__all__'
