# Generated by Django 5.1 on 2024-08-21 14:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Chatbot', '0002_chatsession_session_name'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='chatsession',
            name='session_id',
        ),
    ]
