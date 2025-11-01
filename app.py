import asyncio
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import io
import logging

from piper_tts import PiperTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes.

# In-memory cache for PiperTTS instances
tts_instances = {}

def get_tts_instance(voice):
    """
    Retrieves a cached PiperTTS instance or creates a new one.
    """
    if voice not in tts_instances:
        logger.info(f"Creating new PiperTTS instance for voice: {voice}")
        try:
            tts_instances[voice] = PiperTTS(voice=f"{voice}.onnx")
        except Exception as e:
            logger.error(f"Failed to create PiperTTS instance for voice {voice}: {e}")
            return None
    return tts_instances[voice]

@app.route('/')
def index():
    """
    Serves the index.html frontend.
    """
    return render_template('index.html')

@app.route('/api/tts', methods=['GET'])
async def synthesize_audio():
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
        synthesizer = tts_instance.synthesize(text, request_id="local")
        async with synthesizer as stream:
            audio_data = bytearray()
            async for audio in stream:
                audio_data.extend(audio.frame.data)

        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='output.wav'
        )

    except Exception as e:
        logger.error(f"Error during synthesis: {e}", exc_info=True)
        return jsonify({"error": "Failed to synthesize audio."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)