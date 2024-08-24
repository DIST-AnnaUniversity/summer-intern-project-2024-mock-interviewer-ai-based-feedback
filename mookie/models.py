from django.utils import timezone
from django.db import models
from django.contrib.auth.models import User

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Resume(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # One-to-one relationship to ensure a single resume per user
    resume_file = models.FileField(upload_to='resumes/')
    extracted_text = models.TextField(blank=True, null=True)
    last_updated = models.DateTimeField(auto_now=True)

    def custom_save(self, *args, **kwargs):
        self.last_updated = timezone.now()
        super(Resume, self).save(*args, **kwargs)

    def __str__(self):
        return f"{self.user} {self.extracted_text} {self.last_updated}"
    
    @classmethod
    def has_uploaded_resume(cls, user):
        return cls.objects.filter(user=user).exists()

