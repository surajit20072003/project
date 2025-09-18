# chatterbox_api.py
import torch
from flask import Flask, request, jsonify
import torchaudio as ta
import uuid
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError as e:
    logger.error(f"Failed to import ChatterboxMultilingualTTS: {e}")
    raise

# --- Model is loaded only ONCE when the server starts ---
print("Starting Chatterbox Microservice...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox model onto device: {DEVICE}")

try:
    CHATTERBOX_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    print("✅ Chatterbox model loaded successfully. Service is ready.")
except Exception as e:
    print(f"❌ Failed to load Chatterbox model: {e}")
    raise

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'model_loaded': CHATTERBOX_MODEL is not None
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    required_fields = ['text', 'lang', 'audio_prompt', 'output_dir']
    
    # Validate required fields
    if not all(field in data for field in required_fields):
        missing_fields = [field for field in required_fields if field not in data]
        return jsonify({
            'error': f'Missing required fields: {missing_fields}'
        }), 400
    
    # Validate audio prompt file exists
    if not os.path.exists(data['audio_prompt']):
        return jsonify({
            'error': f'Audio prompt file not found: {data["audio_prompt"]}'
        }), 400
    
    # Validate output directory
    output_dir = data['output_dir']
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return jsonify({
                'error': f'Could not create output directory: {str(e)}'
            }), 400

    try:
        logger.info(f"Generating TTS for text: '{data['text'][:50]}...' in language: {data['lang']}")
        
        uid = str(uuid.uuid4())
        output_path = os.path.join(output_dir, f"{uid}_tts.wav")
        
        # ✅ IMPROVED: Better parameter handling
        cfg_weight = float(data.get('cfg', 0.3))
        exaggeration = float(data.get('exaggeration', 0.5))
        
        # Clamp values to reasonable ranges
        cfg_weight = max(0.0, min(1.0, cfg_weight))
        exaggeration = max(0.0, min(1.0, exaggeration))
        
        # Use the pre-loaded model for fast generation
        wav = CHATTERBOX_MODEL.generate(
            data['text'],
            language_id=data['lang'],
            audio_prompt_path=data['audio_prompt'],
            cfg_weight=cfg_weight,
            exaggeration=exaggeration
        )
        
        # Save the generated audio
        ta.save(output_path, wav, CHATTERBOX_MODEL.sr)
        
        logger.info(f"✅ TTS generated successfully: {output_path}")
        
        return jsonify({
            'output_path': output_path,
            'status': 'success',
            'duration': len(wav[0]) / CHATTERBOX_MODEL.sr if len(wav.shape) > 1 else len(wav) / CHATTERBOX_MODEL.sr
        })
        
    except Exception as e:
        logger.error(f"❌ Error during generation: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # This starts the API server on port 5001
    print("🚀 Starting Chatterbox API server on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)