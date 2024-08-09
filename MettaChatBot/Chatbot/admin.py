from django.contrib import admin
from .models import ChatSession, ChatMessage


# Register your models here.
admin.site.register(ChatSession)
admin.site.register(ChatMessage)
