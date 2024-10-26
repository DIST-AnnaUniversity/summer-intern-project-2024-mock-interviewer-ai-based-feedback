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
from bs4 import BeautifulSoup
import requests
import speech_recognition as sr
from django.http import JsonResponse
import time
from django.views.decorators.csrf import csrf_exempt
import json
import re
from fer import FER
import numpy as np 
#import wave
import subprocess
#import parselmouth
#from parselmouth.praat import call
#import io

emotions=[]
eye_moves= {'left':0,'right':0}
MOVEMENT_THRESHOLD = 10

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
    print("home in views")
    questions = [] 

    # Set session variables only if not already set
    #if 'isSubjectWise' not in request.session:
        #request.session['isSubjectWise'] = False
    #if 'selected_topic' not in request.session:
    #    request.session['selected_topic'] = "javascript"
    
    #print("Session Variables:", request.session.get('isSubjectWise'), request.session.get('selected_topic'))

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, "Logged In")
           
            try:
                print("RESUME QUESTIONS")
                questions, answers = generate_questions(user)
                request.session['answers'] = answers
            except Resume.DoesNotExist:
                messages.error(request, "No resume found for this user.")
            except Exception as e:
                print(f"Error: {e}")  # Log the error for debugging
                messages.error(request, f"An error occurred: {str(e)}")
            
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

    predictor = dlib.shape_predictor("C:/Users/abharna shree m/Downloads/shape_predictor_68_face_landmarks.dat")
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42,48))
    prev_left_eye_center = None
    prev_right_eye_center = None
    emotion_detector = FER(mtcnn=True)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240)) 

        frame_width= frame.shape[1]
    
        if not ret or frame is None or frame.size == 0:
            print("Frame not captured correctly")
            break
        
        result = emotion_detector.detect_emotions(frame)
        if result:
            ep=result[0]['emotions']
            maxe=max(ep, key=ep.get)
            #print(f"Emotion detected: {maxe}")
            emotions.append(maxe)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray,1)
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x,p.y] for p in landmarks.parts()])

            left_eye = landmarks[LEFT_EYE]
            right_eye =landmarks[RIGHT_EYE]

            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)

            if prev_left_eye_center is not None and prev_right_eye_center is not None:
                left_eye_movement = np.abs(left_eye_center[0] - prev_left_eye_center[0])
                right_eye_movement = np.abs(right_eye_center[0] - prev_right_eye_center[0])

                if left_eye_movement > MOVEMENT_THRESHOLD:
                    if left_eye_center[0] < prev_left_eye_center[0]:
                        eye_moves['left'] += 1
                    else:
                        eye_moves['right'] += 1

                if right_eye_movement > MOVEMENT_THRESHOLD:
                    if right_eye_center[0] < prev_right_eye_center[0]:
                        eye_moves['left'] += 1
                    else:
                        eye_moves['right'] += 1
            prev_left_eye_center = left_eye_center
            prev_right_eye_center = right_eye_center
        #print(f"Eye moves left: {eye_moves['left']}, Eye moves right: {eye_moves['right']}")

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


def video_feed(request):
    return StreamingHttpResponse(face_detection(), content_type='multipart/x-mixed-replace; boundary=frame')

def video_results(request):
    print("\n INSIDE VIDEO RESULTS \n")
    eye_moved = eye_moves.get('left', 0) + eye_moves.get('right', 0)
    most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "No emotion detected"
    
    print("VIDEO RESULTS : eye moved - ",eye_moved,"most common emotion - ",most_common_emotion)

    return JsonResponse({
        'eye_moved': eye_moved,
        'most_common_emotion': most_common_emotion
    })

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
        recognizer.adjust_for_ambient_noise(source, timeout=7)
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
        
        try:
            response = llm_chain.invoke({})
            questions_text = response["text"]
            print("GENERATED CONTENT:", questions_text, "\n")
            
        except Exception as api_error:
            if "429" in str(api_error):  # Check for rate limit (HTTP 429)
                raise Exception("Rate limit reached. Please try again later.")
            else:
                raise Exception(f"An API error occurred: {api_error}")

        # Split based on question pattern
        questions_and_answers = re.split(r'(\d+\.\sQuestion:)', questions_text.strip())
        
        # Combine question parts with their respective answers
        combined = []
        for i in range(1, len(questions_and_answers), 2):
            question = questions_and_answers[i] + questions_and_answers[i + 1]
            # Find the answer that follows the question
            answer_start_index = question.index('Answer:') if 'Answer:' in question else -1
            if answer_start_index != -1:
                answer = question[answer_start_index:].strip()
                question = question[:answer_start_index].strip()
                combined.append((question, answer))

        # Extract questions and answers from combined text
        questions = [qa[0] for qa in combined]
        answers = [qa[1] for qa in combined]

        print("QUESTIONS:", questions, "\n")
        print("ANSWERS:", answers, "\n")

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

    return JsonResponse({
        'percentage': percentage,
    })


def Skill_Streams(request):
    if request.method == "POST":
        # Set session variables
        request.session['isSubjectWise'] = True
        selected_topic = request.POST.get('topic')
        request.session['selected_topic'] = selected_topic  # Store the selected topic in the session

        # Redirect to home page after selection
        return redirect('home')  # Redirect to the 'home' URL

    return render(request, "Skill_Streams.html")

def get_questions_answers(topic):
    urls = {
        "operating_systems": 'https://www.interviewbit.com/operating-system-interview-questions/',
        "oops": 'https://www.interviewbit.com/oops-interview-questions/',
        "javascript": 'https://www.interviewbit.com/javascript-interview-questions/'
    }

    url = urls.get(topic)

    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to retrieve content. Status code: {r.status_code}")
        return []

    soup = BeautifulSoup(r.content, 'html.parser')

    questions = soup.find_all('h3')

    question_list = []
    answer_list = []

    for question in questions:
        question_text = question.get_text(strip=True)

        if "code" in question_text.lower() or "download" in question_text.lower():
            continue

        answers = question.find_all_next('p')
        answer_texts = [ans.get_text(strip=True) for ans in answers if ans.get_text(strip=True)]

        if answer_texts:
            full_answer = " ".join(answer_texts[:3]) 
            question_list.append(question_text)
            answer_list.append(full_answer)

    print("QUESTIONS:", question_list, "\n")
    print("ANSWERS:", answer_list, "\n")

    return question_list, answer_list

@csrf_exempt  # Allow CSRF exempt for testing, remove in production
def scrape_questions(request):
    if request.method == 'POST':
        selected_topic = request.POST.get('topic')
        qa_pairs = get_questions_answers(selected_topic)

        return render(request, 'results.html', {'qa_pairs': qa_pairs})

    return render(request, 'index.html')  # Render the selection form again if not a POST request