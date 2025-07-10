# Audio Transcription App

A Python web application that transcribes audio files using OpenAI's Whisper API and identifies the number of speakers in the audio.

## Features

- Drag-and-drop audio file upload
- Audio transcription using OpenAI Whisper API
- Speaker identification and counting
- Download transcription as a text file
- Support for multiple audio formats (MP3, WAV, FLAC, M4A, OGG, AAC)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file

3. Run the application:
```bash
python app.py
```

4. Open your browser and go to `http://localhost:5000`

## Usage

1. Drag and drop an audio file onto the upload area or click to browse
2. Click "Transcribe Audio" to start processing
3. Wait for the transcription to complete
4. Download the transcription as a text file

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## Note

The speaker identification uses pyannote.audio for accurate speaker diarization. If the model fails to load, it falls back to a simple estimation method.