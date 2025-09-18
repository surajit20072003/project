import os
import uuid
import subprocess
import logging
import json
import cv2
import requests
import time

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.http import FileResponse
from django.conf import settings 
from .models import UserVideo
from rest_framework.permissions import AllowAny
from .wav2lip_utils import run_wav2lip_integrated, face_detect, clear_face_cache
from .ml_models import ollama_client

logger = logging.getLogger(__name__)

# --- UTILITY FUNCTION FOR LLAMA3 ---
def generate_llama_response(prompt):
    if ollama_client is None:
        raise Exception("Ollama client not initialized.")
    try:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant. Provide a very direct and short answer, in one sentence and 15 words or less.'},
            {'role': 'user', 'content': prompt}
        ]
        response = ollama_client.chat(model='llama3', messages=messages)
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise Exception(f"Failed to generate Llama3 response: {e}")

# --- API 1: Create a Speaker Profile ---
class CreateSpeakerProfile(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]
    
    def post(self, request):
        request_start_time = time.time()
        timings = {}
        
        try:
            # Input validation timing
            step_start = time.time()
            video_file = request.FILES.get('video')
            speaker_id = request.data.get('speaker_id')
            
            if not video_file or not speaker_id:
                return Response({'error': 'Video file and speaker_id are required.'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate file type and size
            if not video_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return Response({'error': 'Invalid video format. Supported: mp4, avi, mov, mkv'}, status=status.HTTP_400_BAD_REQUEST)
            
            if video_file.size > 100 * 1024 * 1024:  # 100MB limit
                return Response({'error': 'Video file too large (max 100MB)'}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            
            if UserVideo.objects.filter(speaker_id=speaker_id).exists():
                return Response({'error': f"Speaker ID '{speaker_id}' already exists."}, status=status.HTTP_409_CONFLICT)
            
            timings['validation'] = time.time() - step_start
            
            # Database and file operations
            step_start = time.time()
            user_video = UserVideo.objects.create(speaker_id=speaker_id, video_file=video_file)
            input_video_path = user_video.video_file.path
            speaker_dir = os.path.dirname(input_video_path)
            stored_audio_path = os.path.join(speaker_dir, "voice_reference.wav")
            
            # Extract audio from video
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", stored_audio_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
            )
            timings['file_operations'] = time.time() - step_start
            
            # Face detection and caching
            step_start = time.time()
            logger.info(f"Starting face detection for speaker: {speaker_id}")
            video_stream = cv2.VideoCapture(input_video_path)
            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading: 
                    break
                full_frames.append(frame)
            video_stream.release()
            
            if not full_frames: 
                raise Exception("Could not read frames from video.")
            
            # Use face detection with caching
            face_det_results = face_detect(full_frames)
            coordinates = [coords for face, coords in face_det_results if coords is not None]
            
            if not coordinates: 
                logger.warning(f"No faces detected for speaker_id: {speaker_id}")
                return Response({'error': f"No faces detected in video for speaker: {speaker_id}"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Convert numpy int64 to regular Python int for JSON serialization
            serializable_coordinates = []
            for coord in coordinates:
                if isinstance(coord, (list, tuple)):
                    # Convert each coordinate value to regular Python int
                    serializable_coord = [int(c) for c in coord]
                    serializable_coordinates.append(serializable_coord)
                else:
                    # Handle case where coord might be a single value or numpy array
                    serializable_coordinates.append(int(coord) if hasattr(coord, 'item') else coord)
            
            # Save coordinates for backward compatibility
            coordinates_path = os.path.join(speaker_dir, "face_coordinates.json")
            with open(coordinates_path, 'w') as f:
                json.dump(serializable_coordinates, f)
                
            timings['face_detection'] = time.time() - step_start
            
            # Total processing time
            total_time = time.time() - request_start_time
            logger.info(f"=== CreateSpeakerProfile TIMINGS === Total: {total_time:.2f}s | Validation: {timings.get('validation', 0):.2f}s | File Ops: {timings.get('file_operations', 0):.2f}s | Face Detection: {timings.get('face_detection', 0):.2f}s")
            
            return Response({
                'status': 'Speaker profile created successfully', 
                'speaker_id': speaker_id,
                'faces_detected': len(serializable_coordinates),
                'processing_time': f"{total_time:.2f}s"
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error in CreateSpeakerProfile: {e}", exc_info=True)
            if 'user_video' in locals() and user_video.pk: 
                user_video.delete()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- API 2: Generate Video From Your Own Text (FAST VERSION) ---
class GenerateVideoFromText(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        request_start_time = time.time()
        timings = {}
        temp_files = []
        
        try:
            # Input validation
            step_start = time.time()
            input_text = request.data.get('text')
            speaker_id = request.data.get('speaker_id')
            lang = request.data.get('lang')

            if not all([input_text, speaker_id, lang]):
                return Response({'error': 'text, speaker_id, and lang are required.'}, status=status.HTTP_400_BAD_REQUEST)
            
            if not input_text.strip():
                return Response({'error': 'Input text cannot be empty'}, status=status.HTTP_400_BAD_REQUEST)
            
            if len(input_text) > 1000:
                return Response({'error': 'Input text too long (max 1000 characters)'}, status=status.HTTP_400_BAD_REQUEST)
            
            if lang not in ['en', 'hi']:
                return Response({'error': 'Unsupported language. Use "en" or "hi"'}, status=status.HTTP_400_BAD_REQUEST)
            
            timings['validation'] = time.time() - step_start
            
            # Database lookup
            step_start = time.time()
            try:
                user_video = UserVideo.objects.get(speaker_id=speaker_id)
                source_video_path = user_video.video_file.path
                speaker_dir = os.path.dirname(source_video_path)
                stored_audio_path = os.path.join(speaker_dir, "voice_reference.wav")
                coordinates_path = os.path.join(speaker_dir, "face_coordinates.json")

                if not all(os.path.exists(p) for p in [source_video_path, stored_audio_path, coordinates_path]):
                    return Response({'error': f'Profile files missing for speaker: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
                
            except UserVideo.DoesNotExist:
                return Response({'error': f'Speaker profile not found for ID: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
            
            timings['db_lookup'] = time.time() - step_start
            
            # TTS Generation via Chatterbox service
            step_start = time.time()
            uid = str(uuid.uuid4())
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            chatterbox_url = "http://127.0.0.1:5001/generate"
            payload = {
                "text": input_text,
                "lang": lang,
                "audio_prompt": stored_audio_path,
                "output_dir": temp_dir
            }
            
            logger.info(f"Sending request to Chatterbox service...")
            response = requests.post(chatterbox_url, json=payload, timeout=180)
            response.raise_for_status()
            response_data = response.json()
            tts_output_path = response_data.get('output_path')
            
            if not tts_output_path or not os.path.exists(tts_output_path):
                raise Exception("Chatterbox service did not return a valid file path.")
            
            temp_files.append(tts_output_path)
            timings['tts_generation'] = time.time() - step_start
            
            # Load face coordinates for backward compatibility
            step_start = time.time()
            with open(coordinates_path, 'r') as f:
                saved_coords = json.load(f)
            if not saved_coords:
                return Response({'error': f'No face data for speaker: {speaker_id}.'}, status=status.HTTP_400_BAD_REQUEST)
            timings['load_coords'] = time.time() - step_start
            
            # Wav2Lip inference with caching
            step_start = time.time()
            final_output_video_path = os.path.join(temp_dir, f"{uid}_result.mp4")
            temp_files.append(final_output_video_path)
            
            # Use speaker_id for caching, fallback to saved_coords
            run_wav2lip_integrated(
                input_video_path=source_video_path,
                audio_path=tts_output_path,  
                output_path=final_output_video_path,
                saved_coords=saved_coords,
                speaker_id=speaker_id  # Enable face detection caching
            )
            timings['wav2lip_inference'] = time.time() - step_start
            
            # Total processing time
            total_time = time.time() - request_start_time
            logger.info(f"=== GenerateVideoFromText TIMINGS === Total: {total_time:.2f}s | Validation: {timings.get('validation', 0):.2f}s | DB Lookup: {timings.get('db_lookup', 0):.2f}s | TTS: {timings.get('tts_generation', 0):.2f}s | Load Coords: {timings.get('load_coords', 0):.2f}s | Wav2Lip: {timings.get('wav2lip_inference', 0):.2f}s")
            
            if os.path.exists(final_output_video_path):
                response = FileResponse(open(final_output_video_path, 'rb'), content_type='video/mp4')
                response['Content-Disposition'] = f'attachment; filename="result_{uid}.mp4"'
                return response
            else:
                raise Exception('Wav2Lip failed to generate the final video.')

        except UserVideo.DoesNotExist:
            return Response({'error': f'Speaker profile not found for ID: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error in GenerateVideoFromText: {e}", exc_info=True)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # Cleanup temp files (except final output being served)
            for temp_file in temp_files[:-1]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

# --- API 3: Generate Video From Llama3's Answer (FAST VERSION) ---
class GenerateVideoFromLlama(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        request_start_time = time.time()
        timings = {}
        temp_files = []
        
        try:
            # Input validation
            step_start = time.time()
            prompt = request.data.get('prompt')
            speaker_id = request.data.get('speaker_id')
            lang = request.data.get('lang', 'en')

            if not all([prompt, speaker_id]):
                return Response({'error': 'prompt and speaker_id are required.'}, status=status.HTTP_400_BAD_REQUEST)
            
            if not prompt.strip():
                return Response({'error': 'Prompt cannot be empty'}, status=status.HTTP_400_BAD_REQUEST)
            
            if len(prompt) > 500:
                return Response({'error': 'Prompt too long (max 500 characters)'}, status=status.HTTP_400_BAD_REQUEST)
            
            if lang not in ['en', 'hi']:
                return Response({'error': 'Unsupported language. Use "en" or "hi"'}, status=status.HTTP_400_BAD_REQUEST)
            
            timings['validation'] = time.time() - step_start
            
            # Generate Llama3 response
            step_start = time.time()
            llama_answer = generate_llama_response(prompt)
            timings['llama_generation'] = time.time() - step_start
            
            # Database lookup
            step_start = time.time()
            try:
                user_video = UserVideo.objects.get(speaker_id=speaker_id)
                source_video_path = user_video.video_file.path
                speaker_dir = os.path.dirname(source_video_path)
                stored_audio_path = os.path.join(speaker_dir, "voice_reference.wav")
                coordinates_path = os.path.join(speaker_dir, "face_coordinates.json")

                if not all(os.path.exists(p) for p in [source_video_path, stored_audio_path, coordinates_path]):
                    return Response({'error': f'Profile files missing for speaker: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
                
            except UserVideo.DoesNotExist:
                return Response({'error': f'Speaker profile not found for ID: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
            
            timings['db_lookup'] = time.time() - step_start
            
            # TTS Generation via Chatterbox service
            step_start = time.time()
            uid = str(uuid.uuid4())
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            chatterbox_url = "http://127.0.0.1:5001/generate"
            payload = {
                "text": llama_answer,  # Use Llama's answer here
                "lang": lang,
                "audio_prompt": stored_audio_path,
                "output_dir": temp_dir
            }
            
            logger.info(f"Sending Llama request to Chatterbox service...")
            response = requests.post(chatterbox_url, json=payload, timeout=180)
            response.raise_for_status()
            response_data = response.json()
            tts_output_path = response_data.get('output_path')
            
            if not tts_output_path or not os.path.exists(tts_output_path):
                raise Exception("Chatterbox service did not return a valid file path.")
            
            temp_files.append(tts_output_path)
            timings['tts_generation'] = time.time() - step_start
            
            # Load face coordinates
            step_start = time.time()
            with open(coordinates_path, 'r') as f:
                saved_coords = json.load(f)
            if not saved_coords:
                return Response({'error': f'No face data for speaker: {speaker_id}.'}, status=status.HTTP_400_BAD_REQUEST)
            timings['load_coords'] = time.time() - step_start
            
            # Wav2Lip inference with caching
            step_start = time.time()
            final_output_video_path = os.path.join(temp_dir, f"{uid}_result.mp4")
            temp_files.append(final_output_video_path)
            
            run_wav2lip_integrated(
                input_video_path=source_video_path,
                audio_path=tts_output_path,  
                output_path=final_output_video_path,
                saved_coords=saved_coords,
                speaker_id=speaker_id  # Enable face detection caching
            )
            timings['wav2lip_inference'] = time.time() - step_start
            
            # Total processing time
            total_time = time.time() - request_start_time
            logger.info(f"=== GenerateVideoFromLlama TIMINGS === Total: {total_time:.2f}s | Validation: {timings.get('validation', 0):.2f}s | Llama: {timings.get('llama_generation', 0):.2f}s | DB Lookup: {timings.get('db_lookup', 0):.2f}s | TTS: {timings.get('tts_generation', 0):.2f}s | Load Coords: {timings.get('load_coords', 0):.2f}s | Wav2Lip: {timings.get('wav2lip_inference', 0):.2f}s")
            
            if os.path.exists(final_output_video_path):
                response = FileResponse(open(final_output_video_path, 'rb'), content_type='video/mp4')
                response['Content-Disposition'] = f'attachment; filename="llama_video_{uid}.mp4"'
                return response
            else:
                raise Exception('Wav2Lip failed to generate the final video.')

        except UserVideo.DoesNotExist:
            return Response({'error': f'Speaker profile not found for ID: {speaker_id}.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error in GenerateVideoFromLlama: {e}", exc_info=True)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # Cleanup temp files (except final output being served)
            for temp_file in temp_files[:-1]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

# --- API 4: Generate Text-Only Answer from Llama3 ---
class GenerateLlamaTextAnswer(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        request_start_time = time.time()
        timings = {}
        
        try:
            # Input validation
            step_start = time.time()
            prompt = request.data.get('prompt')
            if not prompt:
                return Response({'error': 'A "prompt" is required.'}, status=status.HTTP_400_BAD_REQUEST)
            
            if not prompt.strip():
                return Response({'error': 'Prompt cannot be empty'}, status=status.HTTP_400_BAD_REQUEST)
            
            if len(prompt) > 1000:
                return Response({'error': 'Prompt too long (max 1000 characters)'}, status=status.HTTP_400_BAD_REQUEST)
            
            timings['validation'] = time.time() - step_start
            
            # Generate Llama3 response
            step_start = time.time()
            generated_answer = generate_llama_response(prompt)
            timings['llama_generation'] = time.time() - step_start
            
            # Total processing time
            total_time = time.time() - request_start_time
            logger.info(f"=== GenerateLlamaTextAnswer TIMINGS === Total: {total_time:.2f}s | Validation: {timings.get('validation', 0):.2f}s | Llama Generation: {timings.get('llama_generation', 0):.2f}s")
            
            return Response({
                'answer': generated_answer,
                'processing_time': f"{total_time:.2f}s"
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in GenerateLlamaTextAnswer: {e}", exc_info=True)
            return Response({'error': f"Internal server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- BONUS API: Cache Management ---
class CacheManagement(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Get cache information"""
        try:
            from .wav2lip_utils import get_cache_info
            cache_info = get_cache_info()
            return Response(cache_info, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request):
        """Clear cache for specific speaker or all"""
        try:
            speaker_id = request.data.get('speaker_id')
            clear_face_cache(speaker_id)
            
            if speaker_id:
                message = f"Cache cleared for speaker: {speaker_id}"
            else:
                message = "All face detection cache cleared"
                
            logger.info(message)
            return Response({'status': message}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)