import asyncio
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import io
import logging
import wave

from piper import PiperVoice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes.

# In-memory cache for PiperVoice instances
tts_instances = {}

def get_tts_instance(voice):
    """
    Retrieves a cached PiperVoice instance or creates a new one.
    """
    if voice not in tts_instances:
        logger.info(f"Creating new PiperVoice instance for voice: {voice}")
        try:
            tts_instances[voice] = PiperVoice.load(f"{voice}.onnx")
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
    voice = request.args.get('voice', 'en_US-lessac-medium') # Default voice

    if not text:
        return jsonify({"error": "Text to synthesize is required."}), 400

    tts_instance = get_tts_instance(voice)
    if not tts_instance:
        return jsonify({"error": f"Could not load voice model for '{voice}'."}), 500

    try:
        # Synthesize audio
        audio_data = bytearray()
        
        # The synthesize_stream_raw method yields audio chunks
        for audio_bytes in tts_instance.synthesize_stream_raw(text):
            audio_data.extend(audio_bytes)
        
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
        return jsonify({"error": "Failed to synthesize audio."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)