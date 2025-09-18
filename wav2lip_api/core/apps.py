# core/apps.py
from django.apps import AppConfig
import logging
import torch
import os
from Wav2Lip.models import Wav2Lip as Wav2LipModel
from Wav2Lip import face_detection

# REMOVED the import for Chatterbox from here

logger = logging.getLogger(__name__)

# Helper function to load the Wav2Lip model
def _load(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device)

def load_wav2lip_model(path, device):
    model = Wav2LipModel()
    logger.info(f"Loading Wav2Lip checkpoint from: {path}")
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        logger.info("SERVER STARTUP: Initializing all AI models...")
        
        from . import ml_models
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ml_models.wav2lip_device = device
        logger.info(f"Using device: {device} for all models.")

        # --- Load Wav2Lip Model ---
        try:
            checkpoint_path = "/workspace/project/Wav2Lip/checkpoints/wav2lip_gan.pth"
            if os.path.exists(checkpoint_path):
                ml_models.wav2lip_model = load_wav2lip_model(checkpoint_path, device)
                logger.info("--> Wav2Lip model loaded successfully.")
            else:
                logger.error(f"Wav2Lip checkpoint not found at: {checkpoint_path}")
        except Exception as e:
            logger.error(f"FATAL: Failed to load Wav2Lip model: {e}", exc_info=True)

        # --- Load Face Detector Model ---
        try:
            ml_models.face_detector = face_detection.FaceAlignment(
                face_detection.LandmarksType._2D,
                flip_input=False,
                device=device
            )
            logger.info("--> Face detector model loaded successfully.")
        except Exception as e:
            logger.error(f"FATAL: Failed to load face detector model: {e}", exc_info=True)

        # --- REMOVED THE CHATTERBOX LOADING SECTION ---
        # The model will be loaded on-demand by the chatterbox_service.py script

        # --- Load Ollama Client ---
        try:
            import ollama as ol
            ol.chat(model='llama3', messages=[{'role': 'user', 'content': 'Test'}])
            ml_models.ollama_client = ol
            logger.info("--> Ollama client connection successful.")
        except Exception as e:
            logger.error(f"FATAL: Failed to connect to Ollama/LLaMA3: {e}")

        logger.info("SERVER STARTUP: All AI model initialization complete.")