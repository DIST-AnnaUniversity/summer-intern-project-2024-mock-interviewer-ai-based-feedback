# Generated by Django 5.1 on 2024-08-24 07:12

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mookie', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RenameField(
            model_name='resume',
            old_name='uploaded_at',
            new_name='last_updated',
        ),
        migrations.AlterUniqueTogether(
            name='resume',
            unique_together={('user',)},
        ),
    ]
