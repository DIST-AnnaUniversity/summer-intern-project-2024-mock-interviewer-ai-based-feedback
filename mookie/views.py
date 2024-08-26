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
import speech_recognition as sr
from django.http import JsonResponse
import time
from django.views.decorators.csrf import csrf_exempt
import json
import re

@csrf_exempt
def update_question_index(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        index = data.get('index', 0)
        request.session['current_question_index'] = index
        return JsonResponse({'status': 'success'})
    return JsonResponse({'error': 'Invalid request'}, status=400)

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
                questions,answers = generate_questions(user)
                request.session['answers']=answers
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

    
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Please answer the question:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_answer = recognizer.recognize_google(audio)
        print(f"User's Answer: {user_answer}")
        return user_answer
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    
def get_similarity(text1, text2):
    """
    Returns the similarity score between two texts.
    """
    payload = {
        "inputs": {
            "source_sentence": text1,
            "sentences": [text2]  # Pass text2 as a single-element list
        }
    }

    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
    headers = {"Authorization": "Bearer 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"}

    while True:
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                similarity_score = data[0]  # The similarity score for text2
                return similarity_score * 100  # Convert to percentage
        elif response.status_code == 503:
            data = response.json()
            estimated_time = data.get("estimated_time", 10)  # Default to 10 seconds if not provided
            print(f"Model is still loading. Waiting for {estimated_time} seconds...")
            time.sleep(estimated_time)  # Wait for the estimated time
        else:
            raise Exception(f"Failed to get response from API. Status code: {response.status_code}, Response: {response.text}")
    
def generate_questions(user):
    os.environ["HF_TOKEN"] = "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"  

    try:
        resume = Resume.objects.get(user=user)
        resume_content = resume.extracted_text

        if not resume_content:
            raise ValueError("No resume content found for the user.")

        summarized_content = summarize_content(resume_content)

        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=os.environ["HF_TOKEN"])

        prompt = f"""
        Based on the following summarized resume content:

        {summarized_content}

        Generate four interview questions that can be asked to the candidate, along with the expected answers based on the candidate's field of knowledge and experience.
        """

        llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
        
        response = llm_chain.invoke({})
        questions_text = response["text"]  

        print(questions_text)

        questions_and_answers = re.split(r'\n(?=\d+\.)', questions_text.strip())
        print(questions_and_answers)
        questions = [qa.split("Expected Answer:")[0].strip() for qa in questions_and_answers if "Expected Answer:" in qa]
        answers = [qa.split("Expected Answer:")[1].strip() for qa in questions_and_answers if "Expected Answer:" in qa]

        
        print(questions)
        print(answers)

        return questions, answers
    
    except Resume.DoesNotExist:
        raise Resume.DoesNotExist("No resume found for this user.")
    except Exception as e:
        raise Exception(f"An error occurred during question generation: {str(e)}")


def interview_process(request):
    user_answer = recognize_speech()

    if not user_answer:
        return JsonResponse({'error': 'No speech detected or error in recognition'}, status=400)
    
    current_question_index = request.session.get('current_question_index', 0)
    answers = request.session.get('answers', [])

    if not answers:
        return JsonResponse({'error': 'No expected answers found'}, status=400)
    
    if current_question_index >= len(answers):
        return JsonResponse({'error': 'Current question index out of range'}, status=400)
    
    expected_answer = answers[current_question_index]

    percentage = get_similarity(user_answer, expected_answer)

    request.session['best_match_percentage'] = percentage

    return JsonResponse({'percentage': percentage})