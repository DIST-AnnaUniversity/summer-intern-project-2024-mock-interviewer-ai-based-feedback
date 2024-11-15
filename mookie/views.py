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
from bs4 import BeautifulSoup
import os
import requests
import speech_recognition as sr
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from django.http import JsonResponse
import time
from django.views.decorators.csrf import csrf_exempt
import json
import re
from fer import FER
import torch
import numpy as np 

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
        resume = Resume.objects.get(user=request.user)  
        has_uploaded = True
    except Resume.DoesNotExist:
        resume = None
        has_uploaded = False

    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES, instance=resume)
        if form.is_valid():
            resume = form.save(commit=False)
            resume.user = request.user

            
            reader = PdfReader(resume.resume_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            resume.extracted_text = text
            resume.custom_save()
            messages.success(request, 'Your resume was uploaded successfully!')
            return redirect('home')  
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
                
                questions,answers = generate_questions(user)
                request.session['answers']=answers
            except Resume.DoesNotExist:
                messages.error(request, "No resume found for this user.")
            except Exception as e:
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


left_eye_moves = 0
right_eye_moves = 0
emotions = []
MOVEMENT_THRESHOLD = 7

def face_detection():
    global left_eye_moves  
    global right_eye_moves
    global emotions  

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/abharna shree m/Downloads/shape_predictor_68_face_landmarks.dat")
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    prev_left_eye_center = None
    prev_right_eye_center = None
    emotion_detector = FER(mtcnn=True)

    last_left_move_time = 0
    last_right_move_time = 0
    TIME_THRESHOLD = 1.5  

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240)) 

        if not ret or frame is None or frame.size == 0:
            print("Frame not captured correctly")
            break
        
     
        result = emotion_detector.detect_emotions(frame)
        if result:
            ep = result[0]['emotions']
            maxe = max(ep, key=ep.get)
            emotions.append(maxe)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]

            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)

            current_time = time.time() 
            if prev_left_eye_center is not None and prev_right_eye_center is not None:
                left_eye_movement = np.abs(left_eye_center[0] - prev_left_eye_center[0])
                right_eye_movement = np.abs(right_eye_center[0] - prev_right_eye_center[0])

                if left_eye_movement > MOVEMENT_THRESHOLD and current_time - last_left_move_time > TIME_THRESHOLD:
                    if left_eye_center[0] < prev_left_eye_center[0]:
                        left_eye_moves += 1
                    else:
                        right_eye_moves += 1
                    last_left_move_time = current_time  

                if right_eye_movement > MOVEMENT_THRESHOLD and current_time - last_right_move_time > TIME_THRESHOLD:
                    if right_eye_center[0] < prev_right_eye_center[0]:
                        left_eye_moves += 1
                    else:
                        right_eye_moves += 1
                    last_right_move_time = current_time 

            prev_left_eye_center = left_eye_center
            prev_right_eye_center = right_eye_center

       
        print(f"Eye moves left: {left_eye_moves}, Eye moves right: {right_eye_moves}")

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def video_feed(request):
    """Streaming the live video feed."""
    return StreamingHttpResponse(face_detection(), content_type='multipart/x-mixed-replace; boundary=frame')

def video_results(request):
    """Return the eye movement and emotion results as JSON."""
    eye_moved = left_eye_moves + right_eye_moves 
    print("EYE MOVED-",{eye_moved})
    most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "No emotion detected"
    return JsonResponse({
        'eye_moved': eye_moved,
        'most_common_emotion': most_common_emotion
    })

def summarize_content(content):
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer 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"}
    
    payload = {"inputs": content}
    
    response = requests.post(api_url, headers=headers, json=payload)
    summary = response.json()
    
  
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
            "sentences": [text2] 
        }
    }

    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
    headers = {"Authorization": "Bearer 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"}

    while True:
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                similarity_score = data[0] 
                return similarity_score * 100  
        elif response.status_code == 503:
            data = response.json()
            estimated_time = data.get("estimated_time", 10)  
            print(f"Model is still loading. Waiting for {estimated_time} seconds...")
            time.sleep(estimated_time) 
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

        Generate FOUR interview questions that can be asked to the candidate, along with the expected answers based on the candidate's field of knowledge and experience.
        """

        llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
        
        response = llm_chain.invoke({})

        print(f"Full LLM Response: {response}")

        questions_text = response["text"]  

        print("GEBERATED CONTENT : ",questions_text,"\n")

        questions_and_answers = re.split(r'(\d+\.\sQuestion:)', questions_text.strip())
        print("QUESTIONS AND ANSWERS SPLIT : ",questions_and_answers,"\n")
        combined = []
        for i in range(1, len(questions_and_answers), 2):
            combined.append(questions_and_answers[i] + questions_and_answers[i + 1])

       
        questions = [qa.split("Expected Answer:")[0].strip() for qa in combined]
        answers = [qa.split("Expected Answer:")[1].strip() for qa in combined]
        
        print("QUESTIONS : ",questions,"\n")
        print("ANSWERS : ",answers,"\n")

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

def get_questions_answers(topic):
    topic_urls = {
        'operating_systems': 'https://www.interviewbit.com/operating-system-interview-questions/',
        'oops': 'https://www.interviewbit.com/oops-interview-questions/',
        'javascript': 'https://www.interviewbit.com/javascript-interview-questions/'
    }

    url = topic_urls.get(topic, 'https://www.interviewbit.com/javascript-interview-questions/')  
    r = requests.get(url)
    
    if r.status_code != 200:
        print(f"Failed to retrieve content. Status code: {r.status_code}")
        return []

    soup = BeautifulSoup(r.content, 'html.parser')
    questions = soup.find_all('h3')

    qa_pairs = []
    for question in questions:
        question_text = question.get_text(strip=True)
        if "code" in question_text.lower() or "download" in question_text.lower():
            continue

        answers = question.find_all_next('p')
        answer_texts = [ans.get_text(strip=True) for ans in answers if ans.get_text(strip=True)]
        
        if answer_texts:
            full_answer = " ".join(answer_texts[:3])  
            qa_pairs.append({"Question": question_text, "Answer": full_answer})

    return qa_pairs


def calculate_similarity(user_answer, correct_answer):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer([user_answer, correct_answer], return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        user_embedding = model(**inputs)[0][0][0].numpy() 
        correct_embedding = model(**inputs)[0][1][0].numpy()

    cosine_sim = cosine_similarity([user_embedding], [correct_embedding])
    return float(cosine_sim[0][0])


def Skill_Streams(request):
    return render(request, 'options.html')


def skill_qs(request):
    if request.method == 'POST':
     
        if 'user_answer' in request.POST:
            
            selected_topic = request.POST.get('topic', 'javascript')
            print("JUST BEFORE GET QUESTION ANSWERS")
            qa_pairs = get_questions_answers(selected_topic)

           
            user_answer = request.POST.get('user_answer', '')
            print(user_answer)
            question_index = int(request.POST.get('question_index', 0))

            if question_index < len(qa_pairs):
               
                similarity = calculate_similarity(user_answer, qa_pairs[question_index]['Answer'])
                is_correct = similarity >= 0.7

              
                result = {
                    "question": qa_pairs[question_index]['Question'],
                    "similarity": similarity,
                    "is_correct": is_correct,
                    "correct_answer": qa_pairs[question_index]['Answer'],
                    "next_question_index": question_index + 1, 
                }
                return JsonResponse(result)  

      
        else:
           
            selected_topic = request.POST.get('topic', 'javascript')  
            qa_pairs = get_questions_answers(selected_topic)
            context = {'qa_pairs': json.dumps(qa_pairs), 'selected_topic': selected_topic}
            return render(request, 'Skill_Streams.html', context)

    else:
       
        selected_topic = request.GET.get('topic', 'javascript')
        qa_pairs = get_questions_answers(selected_topic) 

    
    context = {'qa_pairs': json.dumps(qa_pairs), 'selected_topic': selected_topic}
    return render(request, 'Skill_Streams.html', context)
