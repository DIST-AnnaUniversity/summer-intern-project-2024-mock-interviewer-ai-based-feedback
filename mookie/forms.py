from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Resume

class LoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)


class ResumeUploadForm(forms.ModelForm):
    class Meta:
        model = Resume
        fields = ['resume_file']
