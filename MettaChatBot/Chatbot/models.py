from django.db import models

# Create your models here.


class ChatSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    user = models.ForeignKey(
        'auth.User', on_delete=models.CASCADE)  # the human user
    # name of the chatbot or system
    system_name = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    sender_type = models.CharField(max_length=10, choices=[(
        'USER', 'User'), ('SYSTEM', 'System')])  # who sent this message?
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
