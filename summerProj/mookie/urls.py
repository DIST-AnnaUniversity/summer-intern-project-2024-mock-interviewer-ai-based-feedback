from django.urls import path
from . import views

urlpatterns = [
   # path('login/', views.login_user, name='login_user'),
    path('logout/', views.logout_user, name='logout_user'),
    path('', views.home, name='home'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('upload/', views.upload_resume, name='upload_resume'),
    path('update_question_index/', views.update_question_index, name='update_question_index'),
    path('interview_process/', views.interview_process, name='interview_process'),
    path('video_results/', views.video_results, name='video_results'),
    path('Skill_Streams/', views.Skill_Streams, name='Skill_Streams'),
    path('scrape/', views.scrape_questions, name='scrape_questions'),
]