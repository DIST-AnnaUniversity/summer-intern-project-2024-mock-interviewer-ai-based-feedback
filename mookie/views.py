from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import dlib
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import ResumeUploadForm, LoginForm
from .models import Resume
from PyPDF2 import PdfReader

def login_user(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('upload_resume')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def logout_user(request):
    logout(request)
    return redirect('login_user')


@login_required
def upload_resume(request):
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            resume = form.save(commit=False)
            resume.user = request.user

            # Extract text from the uploaded PDF
            reader = PdfReader(resume.resume_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            resume.extracted_text = text

            resume.save()
            return redirect('success_url')  # Redirect to a success page or profile view
    else:
        form = ResumeUploadForm()

    return render(request, 'upload_resume.html', {'form': form})

def home(request):
    return render(request,'home.html',{})

def face_detection():
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(face_detection(), content_type='multipart/x-mixed-replace; boundary=frame')

