from django.http import StreamingHttpResponse
import cv2
import dlib
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Resume
from PyPDF2 import PdfReader
from .forms import ResumeUploadForm

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Resume
from .forms import ResumeUploadForm
from PyPDF2 import PdfReader


def upload_resume(request):
    if not request.user.is_authenticated:
        messages.error(request, 'You need to be logged in to upload a resume.')
        return redirect('home')
    
    try:
        resume = Resume.objects.get(user=request.user)  # Get the existing resume if it exists
        has_uploaded = True
    except Resume.DoesNotExist:
        resume = None
        has_uploaded = False

    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES, instance=resume)
        if form.is_valid():
            resume = form.save(commit=False)
            resume.user = request.user

            # Extract text from the uploaded PDF
            reader = PdfReader(resume.resume_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            resume.extracted_text = text
            resume.custom_save()
            messages.success(request, 'Your resume was uploaded successfully!')
            return redirect('home')  # Redirect to a success page or profile view
    else:
        form = ResumeUploadForm(instance=resume)

    return render(request, 'upload_resume.html', {'form': form, 'has_uploaded': has_uploaded})

def home(request):
    if request.method ==  'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "Logged In")
            return redirect('home')
        else: 
            messages.success(request, "Error logging in")
            return redirect('home')
    else:
        return render(request,'home.html',{})
    
def logout_user(request):
    logout(request)
    messages.success(request,"Logged out")
    return redirect('home')

def face_detection():
    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
    
        if not ret or frame is None or frame.size == 0:
            print("Frame not captured correctly")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray,1)
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

