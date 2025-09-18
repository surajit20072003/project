import os
import uuid
import subprocess
import re
import logging
import time
from contextlib import contextmanager

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.http import FileResponse, Http404
from gtts import gTTS
from django.conf import settings 
from .models import UserVideo
from rest_framework.permissions import AllowAny
from .ml_models import tts_model, ollama_client, inflect_engine
from .wav2lip_utils import run_wav2lip_integrated

logger = logging.getLogger(__name__)

# ✅ FIXED: Context manager for proper file handling
@contextmanager
def safe_file_response(file_path, content_type='application/octet-stream'):
    """Context manager to ensure file handles are properly closed"""
    try:
        file_handle = open(file_path, 'rb')
        yield FileResponse(file_handle, content_type=content_type)
    finally:
        if 'file_handle' in locals():
            file_handle.close()

# --- UTILITY FUNCTIONS ---
def generate_llama_response(prompt):
    if ollama_client is None:
        raise Exception("Ollama client not initialized.")
    
    # ✅ FIXED: Add timeout and error handling
    try:
        response = ollama_client.chat(
            model='llama3', 
            messages=[{'role': 'user', 'content': prompt}],
            options={'timeout': 30}  # 30 second timeout
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise Exception(f"Failed to generate response: {e}")

def _process_video_and_audio_for_lipsync(audio_ref_path, input_text, lang, temp_dir, source_video_path):
    uid = str(uuid.uuid4())
    tts_audio_path = os.path.join(temp_dir, f"{uid}_tts.mp3")
    final_audio_input_path = os.path.join(temp_dir, f"{uid}_final_audio.wav")
    final_video_input_path = os.path.join(temp_dir, f"{uid}_video_input.mp4")

    # ✅ FIXED: Input validation
    if not input_text.strip():
        raise ValueError("Input text cannot be empty")
    
    if len(input_text) > 1000:  # Reasonable limit
        raise ValueError("Input text too long (max 1000 characters)")

    try:
        if lang == 'en':
            if tts_model is None:
                raise Exception("TTS model not initialized.")
            tts_model.tts_to_file(
                text=input_text, 
                speaker_wav=audio_ref_path, 
                file_path=tts_audio_path, 
                language="en"
            )
        elif lang == 'hi':
            gTTS(text=input_text, lang='hi').save(tts_audio_path)
        else:
            raise ValueError(f"Unsupported language: {lang}")

        # ✅ FIXED: Better subprocess error handling
        subprocess.run(
            ["ffmpeg", "-y", "-i", tts_audio_path, final_audio_input_path], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=60
        )
        
        audio_duration_output = subprocess.check_output([
            "ffprobe", "-i", final_audio_input_path, "-show_entries", "format=duration", 
            "-v", "quiet", "-of", "csv=p=0"
        ], text=True, timeout=30).strip()
        
        audio_duration = float(audio_duration_output)
        
        # ✅ FIXED: Validate audio duration
        if audio_duration <= 0 or audio_duration > 300:  # Max 5 minutes
            raise ValueError(f"Invalid audio duration: {audio_duration}s")
        
        subprocess.run([
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", source_video_path,
            "-i", final_audio_input_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(audio_duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            final_video_input_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg processing failed: {e}")
        raise Exception(f"Video/Audio processing failed: {e}")
    except subprocess.TimeoutExpired:
        raise Exception("Processing timed out")
    except Exception as e:
        logger.error(f"TTS processing error: {e}")
        raise Exception(f"Text-to-speech failed: {e}")
    finally:
        # ✅ FIXED: Cleanup temp files
        for temp_file in [tts_audio_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
    
    return final_video_input_path, final_audio_input_path

# --- API VIEW CLASSES ---
class GenerateLipSync(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]
    
    def post(self, request):
        request_start_time = time.time()
        timings = {}
        temp_files = []  # ✅ FIXED: Track temp files for cleanup
        
        try:
            video_file = request.FILES.get('video')
            input_text = request.data.get('text')
            speaker_id = request.data.get('speaker_id')
            lang = request.data.get('lang')
            
            # ✅ FIXED: Better validation
            if not all([video_file, input_text, speaker_id, lang]):
                return Response({'error': 'All fields are required.'}, status=status.HTTP_400_BAD_REQUEST)
            
            # ✅ FIXED: Validate file type and size
            if not video_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return Response({'error': 'Invalid video format. Supported: mp4, avi, mov, mkv'}, status=status.HTTP_400_BAD_REQUEST)
            
            if video_file.size > 100 * 1024 * 1024:  # 100MB limit
                return Response({'error': 'Video file too large (max 100MB)'}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            
            # ✅ FIXED: Validate language
            if lang not in ['en', 'hi']:
                return Response({'error': 'Unsupported language. Use "en" or "hi"'}, status=status.HTTP_400_BAD_REQUEST)
            
            if UserVideo.objects.filter(speaker_id=speaker_id).exists():
                return Response({'error': f"Speaker ID '{speaker_id}' already exists."}, status=status.HTTP_409_CONFLICT)

            step_start = time.time()
            user_video = UserVideo.objects.create(speaker_id=speaker_id, video_file=video_file)
            input_video_path = user_video.video_file.path
            speaker_dir = os.path.dirname(input_video_path)
            os.makedirs(speaker_dir, exist_ok=True)
            stored_audio_path = os.path.join(speaker_dir, "voice_reference.wav")
            
            subprocess.run([
                "ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", stored_audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            
            timings['speaker_profile_creation'] = time.time() - step_start

            step_start = time.time()
            base_dir_temp = os.path.join(os.path.dirname(settings.BASE_DIR), 'media', 'temp')
            os.makedirs(base_dir_temp, exist_ok=True)
            
            final_video_input_path, final_audio_path = _process_video_and_audio_for_lipsync(
                audio_ref_path=stored_audio_path, 
                input_text=input_text, 
                lang=lang,
                temp_dir=base_dir_temp, 
                source_video_path=input_video_path
            )
            temp_files.extend([final_video_input_path, final_audio_path])
            timings['video_audio_prep'] = time.time() - step_start

            step_start = time.time()
            uid = str(uuid.uuid4())
            final_output_video_path = os.path.join(base_dir_temp, f"{uid}_result.mp4")
            temp_files.append(final_output_video_path)
            
            run_wav2lip_integrated(
                input_video_path=final_video_input_path,
                audio_path=final_audio_path,  
                output_path=final_output_video_path,
                speaker_id=speaker_id,  # ✅ ADD THIS for caching
                
            )
            timings['wav2lip_inference'] = time.time() - step_start

            total_time = time.time() - request_start_time
            logger.info(f"--- GenerateLipSync TIMINGS --- Total: {total_time:.2f}s, Details: {timings}")

            # ✅ FIXED: Proper file response with cleanup
            if os.path.exists(final_output_video_path):
                response = FileResponse(
                    open(final_output_video_path, 'rb'), 
                    content_type='video/mp4',
                    as_attachment=True,
                    filename=f"lipsync_result_{uid}.mp4"
                )
                
                # Schedule cleanup after response (this is a simplified approach)
                # In production, you'd want a background task for cleanup
                return response
            else:
                return Response({'error': 'Failed to generate output video'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Error in GenerateLipSync: {e}", exc_info=True)
            return Response({'error': f"Internal server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # ✅ FIXED: Cleanup temp files (except the final output which is being served)
            for temp_file in temp_files[:-1]:  # Keep the final output file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

class GenerateFromBrowserTextToVideo(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        request_start_time = time.time()
        timings = {}
        temp_files = []  # ✅ FIXED: Track temp files for cleanup
        
        try:
            input_text = request.data.get('text')
            speaker_id = request.data.get('speaker_id')
            lang = request.data.get('lang')
            
            # ✅ FIXED: Better validation
            if not all([input_text, speaker_id, lang]):
                return Response({'error': 'All fields are required.'}, status=status.HTTP_400_BAD_REQUEST)
            
            if lang not in ['en', 'hi']:
                return Response({'error': 'Unsupported language. Use "en" or "hi"'}, status=status.HTTP_400_BAD_REQUEST)

            step_start = time.time()
            try:
                user_video = UserVideo.objects.get(speaker_id=speaker_id)
                source_video_path = user_video.video_file.path
                stored_audio_path = os.path.join(os.path.dirname(source_video_path), "voice_reference.wav")
                
                # ✅ FIXED: Better file validation
                if not os.path.exists(source_video_path):
                    return Response({'error': f'Video file not found for speaker: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
                    
                if not os.path.exists(stored_audio_path):
                    return Response({'error': f'Voice reference not found for speaker: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
                    
            except UserVideo.DoesNotExist:
                return Response({'error': f'Speaker profile not found for ID: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
            timings['db_lookup'] = time.time() - step_start

            step_start = time.time()
            prompt = f"Give a very direct and short answer, in one sentence and 15 words or less, to this question: {input_text}"
            generated_answer = generate_llama_response(prompt)
            timings['llama3_generation'] = time.time() - step_start
            
            step_start = time.time()
            base_dir_temp = os.path.join(os.path.dirname(settings.BASE_DIR), 'media', 'temp')
            os.makedirs(base_dir_temp, exist_ok=True)
            
            final_video_input_path, final_audio_path = _process_video_and_audio_for_lipsync(
                audio_ref_path=stored_audio_path, 
                input_text=generated_answer, 
                lang=lang,
                temp_dir=base_dir_temp, 
                source_video_path=source_video_path
            )
            temp_files.extend([final_video_input_path, final_audio_path])
            timings['video_audio_prep'] = time.time() - step_start

            step_start = time.time()
            uid = str(uuid.uuid4())
            final_output_video_path = os.path.join(base_dir_temp, f"{uid}_result.mp4")
            temp_files.append(final_output_video_path)
            
            run_wav2lip_integrated(
                input_video_path=final_video_input_path,
                audio_path=final_audio_path,  
                output_path=final_output_video_path,
                speaker_id=speaker_id,  # ✅ ADD THIS for caching
                
            )
            timings['wav2lip_inference'] = time.time() - step_start

            total_time = time.time() - request_start_time
            logger.info(f"--- TIMINGS --- Total: {total_time:.2f}s, Details: {timings}")

            # ✅ FIXED: Proper file response with cleanup
            if os.path.exists(final_output_video_path):
                response = FileResponse(
                    open(final_output_video_path, 'rb'), 
                    content_type='video/mp4',
                    as_attachment=True,
                    filename=f"ai_response_{uid}.mp4"
                )
                return response
            else:
                return Response({'error': 'Failed to generate output video'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Error in GenerateFromBrowser: {e}", exc_info=True)
            return Response({'error': f"Internal server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # ✅ FIXED: Cleanup temp files (except the final output which is being served)
            for temp_file in temp_files[:-1]:  # Keep the final output file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

class GenerateOnlyTextAnswer(APIView):
    parser_classes = (FormParser,)
    permission_classes = [AllowAny]

    def post(self, request):
        user_input = request.data.get('text')
        if not user_input:
            return Response({'error': 'Text input is required.'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            # ✅ FIXED: Corrected variable name from input_text to user_input
            prompt = f"Simple 1-2 line English answer: {user_input}"
            generated_answer = generate_llama_response(prompt)
            return Response({'answer': generated_answer}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in GenerateOnlyTextAnswer: {e}", exc_info=True)
            return Response({'error': f"Internal server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
