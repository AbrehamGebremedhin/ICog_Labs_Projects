# test_views.py
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase
from Chatbot.models import ChatSession, ChatMessage
from django.contrib.auth.models import User


class SessionListTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        self.url = reverse('session-list')

    def test_get_sessions(self):
        ChatSession.objects.create(
            user=self.user, session_id='123', system_name='TestSystem')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_create_session(self):
        data = {'user': self.user.id, 'session_id': '123',
                'system_name': 'TestSystem'}
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)


class SessionDetailTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        self.session = ChatSession.objects.create(
            user=self.user, session_id='123', system_name='TestSystem')
        self.url = reverse('session-detail', args=[self.session.pk])

    def test_get_session(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_delete_session(self):
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)


class MessageListTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser', password='testpassword')
        self.client.force_authenticate(user=self.user)
        self.url = '/api/messages/'

    def test_get_messages(self):
        # Create a ChatSession with the authenticated user
        ChatSession.objects.create(user=self.user)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_create_message(self):
        response = self.client.post(
            self.url, data={'content': 'Hello, world!'})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)


class MessageDetailTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        self.session = ChatSession.objects.create(
            user=self.user, session_id='123', system_name='TestSystem')
        self.message = ChatMessage.objects.create(
            session=self.session, sender_type='USER', content='Hello')
        self.url = reverse('message-detail', args=[self.message.pk])

    def test_delete_message(self):
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
