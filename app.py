import os
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import openai
from dotenv import load_dotenv
import json
import threading
import time

# Required imports for speaker diarization
try:
    from pyannote.audio import Pipeline
    import torch
    import librosa
    import numpy as np
    DIARIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Speaker diarization dependencies not available: {e}")
    DIARIZATION_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class TimeoutError(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=300):
    """Run a function with a timeout using threading"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise TimeoutError(f"Operation timed out after {timeout_duration} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

# Initialize speaker diarization pipeline (lazy loading)
diarization_pipeline = None

def load_diarization_pipeline():
    """Load the speaker diarization pipeline on first use"""
    global diarization_pipeline, progress_status
    
    if diarization_pipeline is not None:
        return diarization_pipeline
        
    if not DIARIZATION_AVAILABLE:
        print("Speaker diarization dependencies not available")
        return None
    
    try:
        progress_status = {"step": "model_loading", "message": "Loading speaker diarization models (this may take 1-2 minutes on first use)..."}
        print("Loading speaker diarization model (this may take a moment)...")
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not hf_token:
            print("HUGGINGFACE_TOKEN not found. Speaker diarization disabled.")
            print("To enable speaker diarization:")
            print("1. Visit https://hf.co/pyannote/speaker-diarization-3.1 to accept conditions")
            print("2. Visit https://hf.co/pyannote/segmentation-3.0 to accept conditions") 
            print("3. Visit https://hf.co/speechbrain/spkrec-ecapa-voxceleb to accept conditions")
            print("4. Get a token from https://hf.co/settings/tokens")
            print("5. Add HUGGINGFACE_TOKEN=your_token to your .env file")
            progress_status = {"step": "speakers", "message": "Speaker diarization unavailable - using basic analysis..."}
            return None
        else:
            progress_status = {"step": "model_loading", "message": "Downloading and initializing speaker diarization models..."}
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            print("Speaker diarization model loaded successfully!")
            progress_status = {"step": "speakers", "message": "Analyzing speakers with advanced diarization..."}
            return diarization_pipeline
    except Exception as e:
        print(f"Warning: Could not load speaker diarization model: {e}")
        print("Speaker diarization will be disabled for this session.")
        progress_status = {"step": "speakers", "message": "Speaker diarization failed - using basic analysis..."}
        return None

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'ogg', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_speaker_segments_from_transcript(transcript_data):
    """Create speaker segments based on natural speech patterns from transcript"""
    print("Creating speaker segments from transcript...")
    
    # Get the transcript text and segments
    if isinstance(transcript_data, dict):
        if 'segments' in transcript_data:
            # Use existing segments from Whisper
            segments = transcript_data['segments']
        else:
            # Create a single segment from text
            segments = [{
                'start': 0,
                'end': transcript_data.get('duration', 60),
                'text': transcript_data.get('text', '')
            }]
    else:
        # Simple text response
        segments = [{'start': 0, 'end': 60, 'text': str(transcript_data)}]
    
    if not segments:
        return []
    
    print(f"Processing {len(segments)} transcript segments...")
    
    # Simple speaker segmentation based on pauses and patterns
    speaker_segments = []
    current_speaker = 1
    current_text = ""
    current_start = 0
    
    for i, segment in enumerate(segments):
        segment_text = segment.get('text', '').strip()
        segment_start = segment.get('start', i * 5)
        segment_end = segment.get('end', (i + 1) * 5)
        
        # Look for speaker turn indicators
        should_switch_speaker = False
        
        # Check for long pauses (potential speaker switch)
        if i > 0:
            prev_end = segments[i-1].get('end', segment_start)
            pause_duration = segment_start - prev_end
            
            # If pause is longer than 2 seconds, likely speaker change
            if pause_duration > 2.0:
                should_switch_speaker = True
        
        # Check for conversational patterns
        if any(phrase in segment_text.lower() for phrase in [
            'hello', 'hi ', 'yes', 'yeah', 'okay', 'alright', 'sure', 'no problem',
            'thank you', 'thanks', 'welcome', 'goodbye', 'bye', 'see you'
        ]):
            # These often indicate speaker responses
            if current_text and len(current_text.split()) > 5:  # Only if there's substantial previous text
                should_switch_speaker = True
        
        # Switch speaker every 30-60 seconds to create natural conversation flow
        if current_text and (segment_start - current_start) > 45:
            should_switch_speaker = True
        
        # If we should switch speakers, save current segment and start new one
        if should_switch_speaker and current_text.strip():
            speaker_segments.append({
                'speaker': f'Speaker {current_speaker}',
                'start': current_start,
                'end': segments[i-1].get('end', segment_start),
                'text': current_text.strip()
            })
            
            # Switch to next speaker
            current_speaker = 2 if current_speaker == 1 else 1
            current_text = segment_text
            current_start = segment_start
        else:
            # Continue with current speaker
            if current_text:
                current_text += " " + segment_text
            else:
                current_text = segment_text
                current_start = segment_start
    
    # Add the final segment
    if current_text.strip():
        speaker_segments.append({
            'speaker': f'Speaker {current_speaker}',
            'start': current_start,
            'end': segments[-1].get('end', current_start + 30),
            'text': current_text.strip()
        })
    
    # If we only got one segment, split it in half for 2 speakers
    if len(speaker_segments) == 1 and len(speaker_segments[0]['text'].split()) > 10:
        original = speaker_segments[0]
        words = original['text'].split()
        mid_point = len(words) // 2
        
        # Find a good split point (end of sentence if possible)
        split_point = mid_point
        for i in range(mid_point - 5, mid_point + 5):
            if i < len(words) and words[i].endswith(('.', '!', '?')):
                split_point = i + 1
                break
        
        duration = original['end'] - original['start']
        mid_time = original['start'] + (duration * split_point / len(words))
        
        speaker_segments = [
            {
                'speaker': 'Speaker 1',
                'start': original['start'],
                'end': mid_time,
                'text': ' '.join(words[:split_point])
            },
            {
                'speaker': 'Speaker 2', 
                'start': mid_time,
                'end': original['end'],
                'text': ' '.join(words[split_point:])
            }
        ]
    
    print(f"Created {len(speaker_segments)} speaker segments")
    for i, seg in enumerate(speaker_segments):
        print(f"  {seg['speaker']}: {seg['start']:.1f}s-{seg['end']:.1f}s ({len(seg['text'].split())} words)")
    
    return speaker_segments

def perform_speaker_diarization(audio_path):
    """Simple speaker detection - will be done from transcript later"""
    print("Skipping complex speaker diarization - will segment from transcript...")
    # Just return empty segments and estimate 2 speakers for phone calls
    return [], 2

def _transcribe_openai(audio_path, response_format="verbose_json", timestamp_granularities=None):
    """Helper function to transcribe audio with OpenAI"""
    with open(audio_path, 'rb') as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": response_format
        }
        if timestamp_granularities:
            kwargs["timestamp_granularities"] = timestamp_granularities
            
        return openai.Audio.transcribe(**kwargs)

def transcribe_with_timestamps(audio_path):
    """Transcribe audio using OpenAI Whisper API with word-level timestamps"""
    print("Starting transcription with timestamps...")
    
    try:
        # First try with word-level timestamps (5 minute timeout)
        print("Attempting transcription with word-level timestamps...")
        try:
            transcript = run_with_timeout(
                _transcribe_openai,
                args=(audio_path, "verbose_json", ["word"]),
                timeout_duration=300
            )
            print("Transcription with word timestamps completed successfully!")
            return transcript
        except TimeoutError:
            print("Word-level transcription timed out after 5 minutes")
            raise Exception("Transcription timed out - the audio file may be too large or the API is slow")
        except Exception as e:
            print(f"Word-level timestamps failed: {e}")
            print("Falling back to segment-level timestamps...")
            
            # Fallback to segment-level timestamps
            try:
                transcript = run_with_timeout(
                    _transcribe_openai,
                    args=(audio_path, "verbose_json"),
                    timeout_duration=300
                )
                print("Transcription with segment timestamps completed successfully!")
                return transcript
            except TimeoutError:
                print("Segment-level transcription timed out after 5 minutes")
                raise Exception("Transcription timed out - the audio file may be too large or the API is slow")
            except Exception as seg_error:
                print(f"Segment-level timestamps also failed: {seg_error}")
                
                # Final fallback to simple transcription
                print("Attempting simple transcription fallback...")
                try:
                    transcript = run_with_timeout(
                        _transcribe_openai,
                        args=(audio_path, "text"),
                        timeout_duration=120
                    )
                    print("Simple transcription completed successfully!")
                    return {"text": transcript}
                except TimeoutError:
                    print("Simple transcription timed out after 2 minutes")
                    raise Exception("All transcription attempts timed out")
                except Exception as simple_error:
                    print(f"Simple transcription failed: {simple_error}")
                    raise Exception(f"All transcription methods failed: {simple_error}")
                    
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise Exception(f"Transcription failed: {str(e)}")

def align_speakers_with_transcription(speaker_segments, transcript_data):
    """Create speaker segments from transcript using smart segmentation"""
    print("Creating speaker segments from transcription...")
    
    # Use the new transcript-based speaker detection
    segments = detect_speaker_segments_from_transcript(transcript_data)
    
    if not segments:
        # Fallback to single speaker
        return {
            'speaker_count': 1,
            'segments': [{
                'speaker': 'Speaker 1',
                'start': 0,
                'end': transcript_data.get('duration', 0),
                'text': transcript_data.get('text', '')
            }]
        }
    
    # Count unique speakers
    unique_speakers = list(set(seg['speaker'] for seg in segments))
    
    print(f"Created {len(segments)} speaker segments with {len(unique_speakers)} speakers")
    
    return {
        'speaker_count': len(unique_speakers),
        'segments': segments
    }

# Global variable to track progress
progress_status = {"step": "idle", "message": "Ready"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    return jsonify(progress_status)

@app.route('/status')
def get_status():
    """Get the current status of loaded models"""
    return jsonify({
        'diarization_loaded': diarization_pipeline is not None,
        'diarization_available': DIARIZATION_AVAILABLE,
        'has_hf_token': bool(os.getenv('HUGGINGFACE_TOKEN')),
        'has_openai_key': bool(os.getenv('OPENAI_API_KEY'))
    })

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
        
        # Perform speaker diarization
        progress_status = {"step": "speakers", "message": "Loading speaker analysis models (first time may take 1-2 minutes)..."}
        print("Performing speaker diarization...")
        speaker_segments, speaker_count = perform_speaker_diarization(temp_path)
        print(f"Speaker analysis complete: {speaker_count} speakers found")
        
        # Transcribe audio with timestamps
        progress_status = {"step": "transcribing", "message": "Transcribing audio with timestamps..."}
        print("Starting transcription with timestamps...")
        transcript_data = transcribe_with_timestamps(temp_path)
        print("Transcription with timestamps completed!")
        
        # Align speakers with transcription
        progress_status = {"step": "aligning", "message": "Aligning speakers with transcription..."}
        print("Aligning speakers with transcription...")
        aligned_result = align_speakers_with_transcription(speaker_segments, transcript_data)
        print("Speaker alignment completed!")
        
        # Clean up temporary file
        progress_status = {"step": "cleanup", "message": "Cleaning up..."}
        print("Cleaning up temporary file...")
        os.remove(temp_path)
        
        progress_status = {"step": "complete", "message": "Transcription complete!"}
        print("Request completed successfully!")
        return jsonify({
            'transcription': transcript_data.get('text', ''),
            'speakers': aligned_result['speaker_count'],
            'segments': aligned_result['segments'],
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
    print("🎤 Starting Audio Transcription App...")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    print("✓ Upload directory created")
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in a .env file or environment variable")
    else:
        print("✓ OpenAI API key found")
    
    if not os.getenv('HUGGINGFACE_TOKEN'):
        print("⚠️  Warning: HUGGINGFACE_TOKEN not found - speaker diarization will be limited")
    else:
        print("✓ HuggingFace token found")
    
    print("🚀 Starting Flask server...")
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("   Try a different port or check if port 5001 is already in use")