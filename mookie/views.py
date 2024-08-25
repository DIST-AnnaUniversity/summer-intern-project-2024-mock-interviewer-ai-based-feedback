from django.http import StreamingHttpResponse
import cv2
import dlib
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Resume
from PyPDF2 import PdfReader
from .forms import ResumeUploadForm
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

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
    questions = [] 

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, "Logged In")
            
            # Generate questions after login
            questions = generate_questions()
            return render(request, 'home.html', {'questions': questions})  # Use 'questions' to match the template
        
        else: 
            messages.error(request, "Error logging in")
            return redirect('home')
    
    return render(request, 'home.html', {'questions': questions})
    
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


def generate_questions():
    os.environ["HF_TOKEN"] = "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"  

    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=os.environ["HF_TOKEN"])

    resume_content = """
    Name: Abharna Shree M
    Education:
    - College of Engineering, Guindy, Chennai: B.Tech Information Technology (CGPA: 8.62, first three semesters)
    - Anna Adarsh Matriculation Higher Secondary School, Chennai: HSC (97.66%)
    - Anna Adarsh Matriculation Higher Secondary School, Chennai: SSLC (97.60%)

    Projects:
    - CUIC Web Portal: Designed frontend using React.js, Tailwind CSS, integrated with backend API (Node.js, Express, MongoDB)
    - Railway Rooster: Designed frontend, implemented employee shift management, PostgreSQL for data storage, shift swap features
    - Expense Tracker: Implemented CRUD operations, used Express.js, PostgreSQL, deployed on Render
    - Culinary Connect: Implemented UI/UX for online food ordering website using HTML, CSS, JS, MySQL, PHP

    Technical Skills:
    - Languages: PHP, Python, C, C++, JavaScript
    - Tools, Frameworks, and Libraries: Node.js, Express, React.js, Tailwind CSS, Git
    - Database Management: PostgreSQL, MySQL

    Achievements:
    - Arduino programming for Padvending Technology project, winner at Technovation - Kurukshetra '24
    - Track Prize in Fintech Domain, CodHer '24, 24-hour Women-only hackathon by ACM-CEG
    - Winner of Webtrix (Website Frontend Contest) at i++ '23, intra-college symposium
    - School Topper in HSLC and SSLC examinations

    Experience:
    - Industrial Relations and Media Coordinator at CEG Tech Forum
    - Associate Head of Web and Apps at Computer Society of Anna University
    - Participated in 5+ hackathons, shortlisted in Smart India Hackathon 2023
    - Member of Anna University School of CsEng ACM Student Chapter
    - Interviewer, Content Writer, Photographer at Guindy Times
    """

    prompt = f"""
    Based on the following resume content:

    {resume_content}

    Generate four interview questions that can be asked to the candidate, along with the expected answers based on the candidate's field of knowledge and experience.
    """

    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
    
    response = llm_chain.invoke({})
    questions_text = response["text"]  

    questions_and_answers = questions_text.strip().split("\n\n")
    questions = [qa.split("Expected Answer:")[0].strip() for qa in questions_and_answers]

    questions = [q for q in questions if q.strip()]

    print(questions)

    return questions
