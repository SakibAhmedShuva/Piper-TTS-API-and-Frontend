import asyncio
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import io
import logging
import wave
import os
from pathlib import Path

from piper import PiperVoice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes.

# In-memory cache for PiperVoice instances
tts_instances = {}

# Directory where voice models are stored
VOICES_DIR = Path(__file__).parent / "voices"  # Adjust this path as needed

def get_tts_instance(voice):
    """
    Retrieves a cached PiperVoice instance or creates a new one.
    """
    if voice not in tts_instances:
        logger.info(f"Creating new PiperVoice instance for voice: {voice}")
        try:
            # Try multiple possible paths
            possible_paths = [
                VOICES_DIR / f"{voice}.onnx",  # voices/ subdirectory
                Path(__file__).parent / f"{voice}.onnx",  # same directory as app.py
                Path(f"{voice}.onnx"),  # current working directory
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    logger.info(f"Found model at: {model_path}")
                    break
            
            if not model_path:
                logger.error(f"Voice model not found. Searched in: {[str(p) for p in possible_paths]}")
                return None
            
            tts_instances[voice] = PiperVoice.load(model_path)
        except Exception as e:
            logger.error(f"Failed to create PiperVoice instance for voice {voice}: {e}")
            return None
    return tts_instances[voice]

@app.route('/')
def index():
    """
    Serves the index.html frontend.
    """
    return render_template('index.html')

@app.route('/api/tts', methods=['GET'])
def synthesize_audio():
    """
    Synthesizes audio from text using Piper TTS.
    """
    text = request.args.get('text')
    voice = request.args.get('voice', 'en_GB-alba-medium') # Default voice

    if not text:
        return jsonify({"error": "Text to synthesize is required."}), 400

    tts_instance = get_tts_instance(voice)
    if not tts_instance:
        return jsonify({"error": f"Could not load voice model for '{voice}'."}), 500

    try:
        # Synthesize audio
        audio_data = bytearray()
        
        # Option 1: Try synthesize_stream (most common method)
        if hasattr(tts_instance, 'synthesize_stream'):
            for audio_bytes in tts_instance.synthesize_stream(text):
                audio_data.extend(audio_bytes)
        
        # Option 2: Try synthesize (returns complete audio)
        elif hasattr(tts_instance, 'synthesize'):
            audio_data = tts_instance.synthesize(text)
        
        # Option 3: Try with wav_file parameter
        else:
            wav_io = io.BytesIO()
            tts_instance.synthesize(text, wav_io)
            wav_io.seek(0)
            return send_file(
                wav_io,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='output.wav'
            )
        
        # Create WAV file in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(tts_instance.config.sample_rate)
            wav_file.writeframes(audio_data)
        
        wav_io.seek(0)
        
        return send_file(
            wav_io,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='output.wav'
        )

    except Exception as e:
        logger.error(f"Error during synthesis: {e}", exc_info=True)
        return jsonify({"error": f"Failed to synthesize audio: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)