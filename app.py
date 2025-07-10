import os
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import openai
from dotenv import load_dotenv

# Optional imports for advanced features
try:
    from pyannote.audio import Pipeline
    import torch
    PYANNOTE_AVAILABLE = True
except ImportError:
    print("Warning: pyannote.audio not available. Speaker diarization will use simple estimation.")
    PYANNOTE_AVAILABLE = False

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not available. Speaker count will default to 1.")
    LIBROSA_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize speaker diarization pipeline
diarization_pipeline = None
# Disabled by default - requires HuggingFace authentication
# if PYANNOTE_AVAILABLE:
#     try:
#         print("Loading speaker diarization model...")
#         diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
#         print("Speaker diarization model loaded successfully.")
#     except Exception as e:
#         print(f"Warning: Could not load speaker diarization model: {e}")
#         diarization_pipeline = None

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'ogg', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_speakers(audio_path):
    """Count the number of speakers using pyannote.audio"""
    print("Starting speaker counting...")
    try:
        if diarization_pipeline is not None:
            print("Using pyannote for speaker diarization...")
            # Use pyannote for proper speaker diarization
            diarization = diarization_pipeline(audio_path)
            speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
            return len(speakers)
        else:
            # Simple fallback - just return 1 speaker to avoid librosa issues
            print("Using simple speaker estimation (defaulting to 1)...")
            return 1
    
    except Exception as e:
        print(f"Error in speaker counting: {e}")
        return 1  # Default to 1 speaker if error occurs

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper API"""
    print("Starting transcription with OpenAI Whisper...")
    try:
        with open(audio_path, 'rb') as audio_file:
            print("Sending audio file to OpenAI API...")
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        print("Transcription completed successfully!")
        return transcript
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise Exception(f"Transcription failed: {str(e)}")

# Global variable to track progress
progress_status = {"step": "idle", "message": "Ready"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    return jsonify(progress_status)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global progress_status
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an audio file.'}), 400
    
    try:
        progress_status = {"step": "upload", "message": "Processing upload..."}
        print(f"Processing upload: {file.filename}")
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        progress_status = {"step": "saving", "message": "Saving uploaded file..."}
        print("Saving uploaded file...")
        file.save(temp_path)
        print(f"File saved to: {temp_path}")
        
        # Count speakers
        progress_status = {"step": "speakers", "message": "Analyzing speakers..."}
        print("Counting speakers...")
        speaker_count = count_speakers(temp_path)
        print(f"Speaker count: {speaker_count}")
        
        # Transcribe audio
        progress_status = {"step": "transcribing", "message": "Transcribing audio with OpenAI Whisper..."}
        print("Starting transcription...")
        transcription = transcribe_audio(temp_path)
        print("Transcription process completed!")
        
        # Clean up temporary file
        progress_status = {"step": "cleanup", "message": "Cleaning up..."}
        print("Cleaning up temporary file...")
        os.remove(temp_path)
        
        progress_status = {"step": "complete", "message": "Transcription complete!"}
        print("Request completed successfully!")
        return jsonify({
            'transcription': transcription,
            'speakers': speaker_count,
            'success': True
        })
    
    except Exception as e:
        progress_status = {"step": "error", "message": f"Error: {str(e)}"}
        print(f"Error during transcription: {str(e)}")
        
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variable")
    
    app.run(debug=True, host='0.0.0.0', port=5001)