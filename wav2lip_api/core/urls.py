# your_app/urls.py

from django.urls import path
from django.shortcuts import render
from .views import (
    # Import the four new, refactored views
    CreateSpeakerProfile,
    GenerateVideoFromText,
    GenerateVideoFromLlama,
    GenerateLlamaTextAnswer
)

urlpatterns = [
    # --- HTML Page Routes ---
    # These paths will serve the 4 HTML files you created.
    # The profile creation page is set as the homepage ('').
    path('', lambda request: render(request, 'create_profile.html'), name='create-profile-page'),
    path('generate-from-text/', lambda request: render(request, 'generate_from_text.html'), name='generate-from-text-page'),
    path('generate-from-llama/', lambda request: render(request, 'generate_from_llama.html'), name='generate-from-llama-page'),
    path('ask-llama/', lambda request: render(request, 'ask_llama_text.html'), name='ask-llama-page'),

    # --- API Endpoints ---
    # These are the paths that your JavaScript/front-end will call.
    path('api/create_speaker_profile/', CreateSpeakerProfile.as_view(), name='api-create-profile'),
    path('api/generate_video_from_text/', GenerateVideoFromText.as_view(), name='api-generate-from-text'),
    path('api/generate_video_from_llama/', GenerateVideoFromLlama.as_view(), name='api-generate-from-llama'),
    path('api/generate_llama_text_answer/', GenerateLlamaTextAnswer.as_view(), name='api-generate-text-answer'),
]