# CLAUDE.md - Audio Transcription App

## Project Overview
A Python Flask web application that transcribes audio files using OpenAI's Whisper API and identifies speakers in the conversation. Features drag-and-drop upload, real-time progress tracking, and speaker-labeled transcription output.

## Key Learnings & Solutions

### 1. Speaker Diarization Challenges
**Problem**: Complex speaker diarization models (pyannote.audio) consistently failed due to:
- Dependency conflicts (torchaudio backend issues)
- Model authentication requirements (HuggingFace tokens)
- Heavy computational requirements causing hangs

**Solution**: Implemented transcript-based speaker segmentation:
- Uses Whisper's natural speech segments and timestamps
- Detects speaker changes based on pause duration (>2 seconds)
- Identifies conversational patterns ("hello", "yes", "thanks", etc.)
- Automatically switches speakers every 45 seconds for natural flow
- Splits at sentence boundaries when possible

### 2. API Timeout Issues
**Problem**: OpenAI Whisper API calls would hang indefinitely, especially with larger audio files.

**Solution**: Multi-level timeout system:
- Threading-based timeouts (signal doesn't work in Flask threads)
- 5-minute timeout for primary transcription
- 2-minute timeout for fallback transcription
- Progressive fallbacks: word-level → segment-level → simple text

### 3. Audio File Loading Problems
**Problem**: librosa failed to load certain audio formats (M4A files).

**Solution**: Multiple audio loading backends:
- Primary: librosa with original sample rate
- Fallback: soundfile library
- Graceful error handling with meaningful defaults

### 4. Frontend Progress Feedback
**Problem**: Users had no visibility into long-running transcription processes.

**Solution**: Real-time progress system:
- Backend progress tracking with global status variable
- Frontend polling every 500ms for smooth updates
- Visual progress bar with step-by-step feedback
- Time estimates and warnings for long processes
- Special handling for first-time model loading

## Technical Architecture

### Backend (Flask)
- **Lazy model loading**: Models load on first use, not at startup
- **Timeout management**: Threading-based approach for API calls
- **Smart speaker detection**: Transcript analysis instead of audio processing
- **Progressive fallbacks**: Multiple strategies for robust operation

### Frontend (HTML/JavaScript)
- **Drag-and-drop interface**: Native file upload with visual feedback
- **Real-time progress**: Polling-based status updates
- **Segmented display**: Speaker-labeled blocks with timestamps
- **Dual download options**: Plain text or speaker-labeled format

### Key Dependencies
```
# Essential
flask==2.3.3
openai==0.28.1
python-dotenv==1.0.0

# Audio processing
librosa==0.10.1
soundfile==0.12.1
numpy==1.24.3

# Optional (for advanced diarization)
pyannote.audio==3.1.1
torch==2.1.0
```

## Environment Setup

### Required Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here  # Optional
```

### SSH Configuration for GitHub
```bash
# Add to ~/.ssh/config
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_ajm339
  IdentitiesOnly yes
```

## Running the Application

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run application
python app.py
# Access at http://localhost:5001
```

### Common Issues & Solutions

1. **Port 5001 in use**:
   - macOS AirPlay Receiver uses port 5001
   - Disable in System Preferences → Sharing
   - Or change port in app.py

2. **Audio loading fails**:
   - Install soundfile: `pip install soundfile`
   - Check audio file format is supported

3. **Transcription hangs**:
   - Check OpenAI API key is valid
   - Verify internet connection
   - Try smaller audio files first

4. **Speaker detection inaccurate**:
   - Algorithm optimized for phone calls (2 speakers)
   - Works best with clear pauses between speakers
   - Falls back to time-based segmentation

## Performance Optimizations

### Startup Time
- Lazy loading prevents slow startup
- Models only load when first needed
- Clear progress feedback during loading

### Transcription Speed
- Word-level timestamps when possible
- Fallback to segment-level for speed
- Simple text as last resort

### Memory Usage
- Temporary files cleaned up immediately
- Models loaded once and reused
- Threading prevents blocking

## Future Improvements

1. **Enhanced Speaker Detection**:
   - Voice fingerprinting for better accuracy
   - Machine learning-based speaker classification
   - Support for 3+ speakers

2. **Audio Processing**:
   - Audio preprocessing (noise reduction)
   - Support for video files
   - Batch processing multiple files

3. **User Experience**:
   - User accounts and file history
   - Export to multiple formats (SRT, VTT)
   - Integration with cloud storage

## Testing Notes

### Successful Test Cases
- Phone call recordings (2 speakers)
- Interview formats
- M4A, MP3, WAV file formats
- Files up to 25MB tested successfully

### Known Limitations
- Optimized for English language
- Best with 2-speaker conversations
- Requires clear audio quality
- 100MB file size limit (Flask setting)

## Deployment Considerations

### Production Setup
- Use production WSGI server (gunicorn, uWSGI)
- Configure proper SSL/TLS
- Set up rate limiting for API protection
- Monitor OpenAI API usage and costs

### Security
- Validate file uploads strictly
- Sanitize filenames
- Implement user authentication for production
- Store API keys securely (environment variables)

## Command Reference

### Git Operations
```bash
# Initial setup
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin git@github.com:ajm339/audio-transcription-app.git
git push -u origin main
```

### SSH Troubleshooting
```bash
# Test GitHub connection
ssh -T git@github.com

# Add SSH key to agent
ssh-add ~/.ssh/github_ajm339

# List loaded keys
ssh-add -l
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Test with sample file
curl -X POST -F "audio=@sample.mp3" http://localhost:5001/transcribe
```

---

*This file serves as a knowledge base for future development and troubleshooting of the audio transcription application.*