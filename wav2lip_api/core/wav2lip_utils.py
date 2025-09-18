import os
import uuid
import cv2
import numpy as np
import torch
from tqdm import tqdm
import subprocess
from Wav2Lip import audio
from .ml_models import wav2lip_model, face_detector, wav2lip_device
import logging
import json

logger = logging.getLogger(__name__)

# Global cache for face detections per speaker
FACE_DETECTION_CACHE = {}

def get_smoothened_boxes(boxes, T):
    """Smooth face detection boxes over time"""
    if len(boxes) == 0:
        return boxes
    
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect_with_cache(images, speaker_id=None):
    """Face detection with caching per speaker"""
    
    # Check if we have cached face detection for this speaker
    if speaker_id and speaker_id in FACE_DETECTION_CACHE:
        cached_data = FACE_DETECTION_CACHE[speaker_id]
        logger.info(f"Using cached face detection for speaker {speaker_id}")
        return apply_cached_detections(images, cached_data)
    
    # Original face detection code
    batch_size = 128
    while True:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(face_detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big for face detection. Consider resizing the video.')
            batch_size //= 2
            logger.warning(f'Recovering from OOM error during face detection; New batch size: {batch_size}')
            continue
        break

    results = []
    pads = [0, 10, 0, 0] 
    for rect, image in zip(predictions, images):
        if rect is None:
            results.append([None, None])
            continue

        y1 = max(0, rect[1] - pads[0])
        y2 = min(image.shape[0], rect[3] + pads[1])
        x1 = max(0, rect[0] - pads[2])
        x2 = min(image.shape[1], rect[2] + pads[3])
        
        results.append([image[y1: y2, x1:x2], (int(x1), int(y1), int(x2), int(y2))])

    # Apply smoothing and cache results
    detected_faces = [res[1] for res in results if res[1] is not None]
    if detected_faces and len(detected_faces) > 1:
        smoothed_boxes = get_smoothened_boxes(np.array(detected_faces), T=3)
        
        smoothed_results = []
        box_idx = 0
        for i, res in enumerate(results):
            if res[1] is not None:
                x1, y1, x2, y2 = smoothed_boxes[box_idx].astype(int)
                smoothed_results.append([images[i][y1:y2, x1:x2], (x1, y1, x2, y2)])
                box_idx += 1
            else:
                smoothed_results.append([None, None])
        
        # Cache the face detection results for this speaker
        if speaker_id and detected_faces:
            FACE_DETECTION_CACHE[speaker_id] = {
                'boxes': smoothed_boxes,
                'image_shapes': [img.shape for img in images]
            }
            logger.info(f"Cached face detection for speaker {speaker_id} - {len(smoothed_boxes)} face boxes")
        
        return smoothed_results
    else:
        return results

def apply_cached_detections(images, cached_data):
    """Apply cached face detections to current images"""
    results = []
    boxes = cached_data['boxes']
    
    for i, image in enumerate(images):
        if i < len(boxes):
            x1, y1, x2, y2 = boxes[i].astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image.shape[1]-1))
            y1 = max(0, min(y1, image.shape[0]-1))
            x2 = max(x1+1, min(x2, image.shape[1]))
            y2 = max(y1+1, min(y2, image.shape[0]))
            
            face = image[y1:y2, x1:x2]
            results.append([face, (x1, y1, x2, y2)])
        else:
            # If we have more images than cached boxes, cycle through cached boxes
            box_idx = i % len(boxes)
            x1, y1, x2, y2 = boxes[box_idx].astype(int)
            x1 = max(0, min(x1, image.shape[1]-1))
            y1 = max(0, min(y1, image.shape[0]-1))
            x2 = max(x1+1, min(x2, image.shape[1]))
            y2 = max(y1+1, min(y2, image.shape[0]))
            
            face = image[y1:y2, x1:x2]
            results.append([face, (x1, y1, x2, y2)])
    
    return results

# Original face_detect function for backward compatibility
def face_detect(images):
    """Original face detect function for profile creation"""
    return face_detect_with_cache(images, speaker_id=None)

def datagen(frames, mels, batch_size, saved_coords=None, speaker_id=None):
    """
    FIXED: Generator that prioritizes saved_coords (FAST PATH) over face detection
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # FAST PATH: Use saved_coords if available - NO FACE DETECTION!
    if saved_coords and len(saved_coords) > 0:
        logger.info(f"⚡ FAST PATH: Using saved_coords for {len(saved_coords)} coordinates - NO face detection!")
        
        num_saved_coords = len(saved_coords)
        
        for i, m in enumerate(mels):
            frame_idx = i % len(frames)
            frame_to_save = frames[frame_idx].copy()
            
            coord_idx = frame_idx % num_saved_coords
            coords = saved_coords[coord_idx]
            
            if not coords or len(coords) != 4:
                logger.warning(f"Invalid coordinates at index {coord_idx}: {coords}")
                continue
                
            x1, y1, x2, y2 = coords
            
            h, w = frame_to_save.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            face = frame_to_save[y1:y2, x1:x2]
            
            if face.size == 0:
                logger.warning(f"Face crop is empty for frame {frame_idx} with coords {coords}. Skipping.")
                continue

            face = cv2.resize(face, (96, 96))
            
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append((x1, y1, x2, y2))

            if len(img_batch) >= batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                img_masked = img_batch.copy()
                img_masked[:, 96//2:] = 0
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, 96//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
    
    else:
        # SLOW PATH: Use face detection only when saved_coords is NOT available
        logger.warning("🐌 SLOW PATH: No saved_coords available, using face detection")
        
        if speaker_id:
            face_det_results = face_detect_with_cache(frames, speaker_id)
        else:
            raise ValueError("Either saved_coords or speaker_id must be provided.")
        
        for i, m in enumerate(mels):
            frame_idx = i % len(frames)
            frame_to_save = frames[frame_idx].copy()
            
            if frame_idx < len(face_det_results):
                face, coords = face_det_results[frame_idx]
            else:
                face, coords = None, None
            
            if face is not None and coords is not None:
                if face.size == 0:
                    logger.warning(f"Face crop is empty for frame {frame_idx}. Skipping.")
                    continue

                face = cv2.resize(face, (96, 96))
                
                img_batch.append(face)
                mel_batch.append(m)
                frame_batch.append(frame_to_save)
                coords_batch.append(coords)

                if len(img_batch) >= batch_size:
                    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                    img_masked = img_batch.copy()
                    img_masked[:, 96//2:] = 0
                    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                    yield img_batch, mel_batch, frame_batch, coords_batch
                    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, 96//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch

def run_wav2lip_integrated(input_video_path, audio_path, output_path, saved_coords=None, speaker_id=None):
    if wav2lip_model is None:
        raise Exception("Wav2Lip model not pre-loaded.")

    # Enable CUDA optimizations if available
    if wav2lip_device == 'cuda':
        torch.backends.cudnn.benchmark = True

    temp_dir = os.path.dirname(output_path)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. Get audio duration
        logger.info("Getting audio duration...")
        audio_duration_output = subprocess.check_output([
            "ffprobe", "-i", audio_path, "-show_entries", "format=duration", 
            "-v", "quiet", "-of", "csv=p=0"
        ], text=True, timeout=30).strip()
        audio_duration = float(audio_duration_output)

        # 2. Loop video to match audio duration
        logger.info(f"Looping source video to match {audio_duration:.2f}s audio duration...")
        looped_video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_looped.mp4")
        
        subprocess.run([
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", input_video_path,
            "-t", str(audio_duration), "-c:v", "copy", "-c:a", "copy", looped_video_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
        
        # 3. Read frames from looped video
        logger.info("Reading frames from looped video...")
        video_stream = cv2.VideoCapture(looped_video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 25.0
            logger.warning("Could not determine FPS, using default 25.0")
            
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            full_frames.append(frame)
        video_stream.release()

        if not full_frames:
            raise Exception("Could not read frames from the looped input video.")
        
        logger.info(f"Read {len(full_frames)} frames at {fps} FPS")
        
        # 4. Process audio into mel spectrogram
        logger.info("Processing audio into mel spectrogram...")
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        # 5. Create mel chunks aligned with video frames
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        
        for i in range(len(full_frames)):
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > mel.shape[1]:
                mel_chunks.append(mel[:, -16:])
            else:
                mel_chunks.append(mel[:, start_idx:start_idx + 16])
        
        min_length = min(len(full_frames), len(mel_chunks))
        full_frames = full_frames[:min_length]
        mel_chunks = mel_chunks[:min_length]
        
        logger.info(f"Generated {len(mel_chunks)} mel chunks for {len(full_frames)} frames")
        
        # 6. Dynamic batch sizing based on video duration
        video_duration = len(full_frames) / fps
        if video_duration < 10:
            batch_size = min(256, len(full_frames))
        elif video_duration < 30:
            batch_size = min(128, len(full_frames))
        else:
            batch_size = min(64, len(full_frames))
            
        logger.info(f"Using batch size: {batch_size} for {video_duration:.1f}s video")
        
        # 7. FIXED: Prioritize saved_coords over speaker_id for fast processing
        if saved_coords and len(saved_coords) > 0:
            logger.info("⚡ Using FAST PATH with saved_coords - skipping face detection")
        else:
            logger.warning("🐌 Using SLOW PATH with face detection")
        
        # Run Wav2Lip inference
        logger.info("Starting Wav2Lip inference...")
        gen = datagen(full_frames, mel_chunks, batch_size, saved_coords=saved_coords, speaker_id=speaker_id)
        
        # Create temporary output video
        temp_avi_path = os.path.join(temp_dir, f"{uuid.uuid4()}_result.avi")
        frame_width, frame_height = full_frames[0].shape[1], full_frames[0].shape[0]
        
        out = cv2.VideoWriter(
            temp_avi_path, 
            cv2.VideoWriter_fourcc(*'XVID'),
            fps, 
            (frame_width, frame_height)
        )
        
        processed_frames = 0
        try:
            for img_batch, mel_batch, frames, coords in tqdm(gen, 
                total=int(np.ceil(float(len(mel_chunks))/batch_size)), 
                desc="Wav2Lip Inference"):
                
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(wav2lip_device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(wav2lip_device)
                
                with torch.no_grad():
                    pred = wav2lip_model(mel_batch, img_batch)
                
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                
                for p, f, c in zip(pred, frames, coords):
                    x1, y1, x2, y2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    out.write(f)
                    processed_frames += 1
                
                # Memory cleanup for CUDA
                if wav2lip_device == 'cuda' and processed_frames % (batch_size * 2) == 0:
                    torch.cuda.empty_cache()
                    
        finally:
            out.release()
            if wav2lip_device == 'cuda':
                torch.cuda.empty_cache()
            
        logger.info(f"Processed {processed_frames} frames")
        
        # 8. Merge with optimized settings
        logger.info("Merging video with audio...")
        preset = "veryfast" if video_duration < 15 else "ultrafast"
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_avi_path, "-i", audio_path,
            "-c:v", "libx264", "-preset", preset,
            "-c:a", "aac", "-strict", "experimental", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
        
        logger.info(f"Final video saved to: {output_path}")
        
    finally:
        # 9. Cleanup temporary files
        for temp_file in [looped_video_path, temp_avi_path]:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")
                    
    return True

# Utility functions for cache management
def clear_face_cache(speaker_id=None):
    """Clear face detection cache for specific speaker or all"""
    global FACE_DETECTION_CACHE
    if speaker_id:
        if speaker_id in FACE_DETECTION_CACHE:
            del FACE_DETECTION_CACHE[speaker_id]
            logger.info(f"Cleared cache for speaker: {speaker_id}")
    else:
        FACE_DETECTION_CACHE.clear()
        logger.info("Cleared all face detection cache")

def get_cache_info():
    """Get cache information and statistics"""
    return {
        'cached_speakers': list(FACE_DETECTION_CACHE.keys()),
        'total_cached_speakers': len(FACE_DETECTION_CACHE)
    }