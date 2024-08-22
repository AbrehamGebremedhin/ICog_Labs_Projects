from rest_framework import status
from Chatbot.models import ChatSession, ChatMessage
from Chatbot.serializers import SessionSerializer, MessageSerializer
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from chatbot import Chat

chat = Chat()

# Create your views here.


class SessionList(APIView):
    """
    List of all user sessions, or create a new session.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        sessions = ChatSession.objects.filter(user=self.request.user)
        serializer = SessionSerializer(sessions, many=True)
        return Response(serializer.data)

    def post(self, request):
        request.data['system_name'] = "llama 3.1"
        serializer = SessionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=self.request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SessionDetail(APIView):
    """
    Retrieve, update or delete a session instance.
    """
    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        try:
            return ChatSession.objects.get(pk=pk)
        except ChatSession.DoesNotExist:
            raise Http404

    def get_messages(self, pk):
        try:
            return ChatMessage.objects.filter(session=pk)
        except ChatMessage.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        messages = self.get_messages(pk)
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)

    def delete(self, request, pk):
        session = self.get_message(pk)
        session.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class MessageList(APIView):
    """
    List of all user sessions, or create a new session.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        # Retrieve messages for the given session
        messages = ChatMessage.objects.filter(session=pk)
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = MessageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            chat_response = chat.query_db(serializer.data['content'])
            response_data = {
                'content': chat_response,
                'session': serializer.data['session'],
                'sender_type': 'SYSTEM'
            }
            message_serializer = MessageSerializer(data=response_data)
            if message_serializer.is_valid():
                message_serializer.save()
                return Response(message_serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MessageDetail(APIView):
    """
    Retrieve, update or delete a session instance.
    """
    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        try:
            return ChatMessage.objects.get(pk=pk)
        except ChatMessage.DoesNotExist:
            raise Http404

    def delete(self, request, pk):
        message = self.get_object(pk)
        message.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
