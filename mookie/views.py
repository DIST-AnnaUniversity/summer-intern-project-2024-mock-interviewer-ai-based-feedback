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
import requests

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
           
            try:
                # Generate questions after login
                questions = generate_questions(user)
            except Resume.DoesNotExist:
                messages.error(request, "No resume found for this user.")
            except Exception as e:
                messages.error(request, f"An error occurred: {str(e)}")  # Use 'questions' to match the template
            
            return render(request, 'home.html', {'questions': questions})
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


import requests

def summarize_content(content):
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer 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"}
    
    # Directly use the content to summarize, without including the prompt in the request
    payload = {"inputs": content}
    
    response = requests.post(api_url, headers=headers, json=payload)
    summary = response.json()
    
    # Extract the summarized text
    summarized_text = summary[0]['summary_text'] if 'summary_text' in summary[0] else "Summary could not be generated."
    print("summary", summarized_text)
    
    return summarized_text




def generate_questions(user):
    os.environ["HF_TOKEN"] = "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"  

    try:
        # Retrieve the resume from the database
        resume = Resume.objects.get(user=user)
        resume_content = resume.extracted_text

        if not resume_content:
            raise ValueError("No resume content found for the user.")

        # Summarize the resume content
        summarized_content = summarize_content(resume_content)

        # Set up your model
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=os.environ["HF_TOKEN"])

        # Construct the prompt using the summarized content
        prompt = f"""
        Based on the following summarized resume content:

        {summarized_content}

        Generate four interview questions that can be asked to the candidate, along with the expected answers based on the candidate's field of knowledge and experience.
        """

        # Create a prompt template and chain for your LLM
        llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
        
        # Generate questions
        response = llm_chain.invoke({})
        questions_text = response["text"]  

        # Process the response to extract questions
        questions_and_answers = questions_text.strip().split("\n\n")
        questions = [qa.split("Expected Answer:")[0].strip() for qa in questions_and_answers]
        
        # Filter out any empty strings from the list
        questions = [q for q in questions if q.strip()]

        print(questions)

        return questions
    
    except Resume.DoesNotExist:
        raise Resume.DoesNotExist("No resume found for this user.")
    except Exception as e:
        raise Exception(f"An error occurred during question generation: {str(e)}")
