from django.urls import path
from Chatbot.views import SessionList, SessionDetail, MessageDetail, MessageList

urlpatterns = [
    path('sessions/', SessionList.as_view(), name='session-list'),
    path('sessions/<int:pk>/', SessionDetail.as_view(), name='session-detail'),
    path('messages/', MessageList.as_view(),
         name='message-list'),  # New pattern without pk
    path('messages/<int:pk>/', MessageDetail.as_view(), name='message-detail')
]
