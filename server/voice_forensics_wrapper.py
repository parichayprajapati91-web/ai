#!/usr/bin/env python3
"""
Wrapper script for voice forensics analysis
Reads JSON from stdin, performs analysis, writes JSON to stdout
Supports language-specific analysis for Hindi, Tamil, Telugu, Malayalam, English
"""
import sys
import json
import base64
import io
from voice_forensics_research import VoiceForensicsAnalyzer
import librosa
import numpy as np

def main():
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Decode audio
        audio_base64 = input_data.get('audioBase64', '')
        audio_bytes = base64.b64decode(audio_base64)
        
        # Get language (default to English)
        language = input_data.get('language', 'English')
        
        # Validate language
        supported_languages = ['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Bengali', 'English']
        if language not in supported_languages:
            language = 'English'
        
        # Load audio using librosa
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Run forensics analysis with language context
        analyzer = VoiceForensicsAnalyzer(language=language)
        result = analyzer.analyze_voice(audio)
        
        # Format output
        output = {
            'status': 'success',
            'classification': {
                'classification': result['classification'],
                'confidence': result['confidence'],
                'accuracy_estimate': result['accuracy_estimate'],
                'analysis_reasons': [
                    f"Spectral: {result['results']['spectral']['classification']} ({result['results']['spectral']['confidence']:.1f}%)",
                    f"Prosodic: {result['results']['prosodic']['classification']} ({result['results']['prosodic']['confidence']:.1f}%)",
                    f"Waveform: {result['results']['waveform']['classification']} ({result['results']['waveform']['confidence']:.1f}%)",
                    f"VAD: {result['results']['vad']['classification']} ({result['results']['vad']['confidence']:.1f}%)"
                ]
            }
        }
        
        print(json.dumps(output))
        sys.exit(0)
        
    except Exception as e:
        output = {
            'status': 'error',
            'error': str(e)
        }
        print(json.dumps(output))
        sys.exit(1)

if __name__ == '__main__':
    main()
