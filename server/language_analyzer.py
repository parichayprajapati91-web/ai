
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language-Aware Voice Forensics Analyzer
Enhanced detection for 5 Indian languages: Hindi, Tamil, Telugu, Malayalam, English

Each language has unique acoustic characteristics that affect AI detection:
- Hindi/Bengali: Aspirated consonants, nasal sounds
- Tamil/Malayalam: Retroflex consonants, vowel length variations  
- Telugu: Geminated consonants, rhythmic patterns
- English: Baseline for comparison
"""

import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.stats import skew, kurtosis
import librosa
import warnings
warnings.filterwarnings('ignore')

class LanguageAwareAnalyzer:
    """Language-specific voice forensics analyzer"""
    
    # Language-specific acoustic characteristics
    LANGUAGE_FEATURES = {
        'Hindi': {
            'name': 'Hindi (Indo-Aryan)',
            'key_phonemes': ['aspirated_consonants', 'nasalization', 'retroflex'],
            'typical_syllable_rate': 5.5,  # syllables per second
            'vowel_duration_range': (0.15, 0.35),  # seconds
        },
        'Tamil': {
            'name': 'Tamil (Dravidian)',
            'key_phonemes': ['retroflex', 'laterals', 'gemination'],
            'typical_syllable_rate': 6.2,
            'vowel_duration_range': (0.12, 0.30),
        },
        'Telugu': {
            'name': 'Telugu (Dravidian)',
            'key_phonemes': ['gemination', 'nasalization', 'retroflex'],
            'typical_syllable_rate': 6.0,
            'vowel_duration_range': (0.13, 0.32),
        },
        'Malayalam': {
            'name': 'Malayalam (Dravidian)',
            'key_phonemes': ['retroflex', 'laterals', 'nasalization'],
            'typical_syllable_rate': 5.8,
            'vowel_duration_range': (0.14, 0.33),
        },
        'English': {
            'name': 'English',
            'key_phonemes': ['fricatives', 'stops', 'nasals'],
            'typical_syllable_rate': 5.0,
            'vowel_duration_range': (0.10, 0.30),
        }
    }
    
    # Language-optimized ensemble weights
    LANGUAGE_WEIGHTS = {
        'Hindi': {
            'spectral': 0.30,
            'prosodic': 0.30,
            'waveform': 0.25,
            'vad': 0.15,
            'language_specific': 0.0
        },
        'Tamil': {
            'spectral': 0.40,
            'prosodic': 0.20,
            'waveform': 0.20,
            'vad': 0.15,
            'language_specific': 0.05  # Retroflex detection
        },
        'Telugu': {
            'spectral': 0.35,
            'prosodic': 0.25,
            'waveform': 0.25,
            'vad': 0.10,
            'language_specific': 0.05  # Gemination detection
        },
        'Malayalam': {
            'spectral': 0.45,
            'prosodic': 0.20,
            'waveform': 0.20,
            'vad': 0.10,
            'language_specific': 0.05  # Retroflex detection
        },
        'English': {
            'spectral': 0.40,
            'prosodic': 0.30,
            'waveform': 0.20,
            'vad': 0.10,
            'language_specific': 0.0
        }
    }
    
    # Language-specific classification thresholds
    LANGUAGE_THRESHOLDS = {
        'Hindi': {
            'min_confidence': 0.65,
            'aspiration_sensitivity': 0.8,
            'nasalization_factor': 0.3
        },
        'Tamil': {
            'min_confidence': 0.70,
            'retroflex_sensitivity': 0.85,
            'vowel_length_variance': 0.25
        },
        'Telugu': {
            'min_confidence': 0.68,
            'gemination_sensitivity': 0.8,
            'rhythm_factor': 0.35
        },
        'Malayalam': {
            'min_confidence': 0.72,
            'retroflex_sensitivity': 0.85,
            'lateral_sensitivity': 0.75
        },
        'English': {
            'min_confidence': 0.70,
            'fricative_sensitivity': 0.7,
            'vowel_quality': 0.3
        }
    }
    
    def __init__(self, language='English', sr=16000):
        self.language = language if language in self.LANGUAGE_FEATURES else 'English'
        self.sr = sr
        self.threshold = self.LANGUAGE_THRESHOLDS.get(self.language, self.LANGUAGE_THRESHOLDS['English'])
        self.weights = self.LANGUAGE_WEIGHTS.get(self.language, self.LANGUAGE_WEIGHTS['English'])
        
    def extract_language_specific_features(self, audio):
        """Extract language-specific acoustic features"""
        features = {}
        
        # Basic audio properties
        features['rms_energy'] = np.sqrt(np.mean(audio ** 2))
        features['zero_crossing_rate'] = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        
        if self.language in ['Tamil', 'Malayalam']:
            # Dravidian languages - retroflex detection
            features['retroflex_ratio'] = self._detect_retroflex_sounds(audio)
            features['vowel_length_variation'] = self._analyze_vowel_duration(audio)
            
        elif self.language in ['Hindi', 'Bengali']:
            # Indo-Aryan languages - aspiration detection
            features['aspiration_presence'] = self._detect_aspirated_consonants(audio)
            features['nasalization_level'] = self._analyze_nasal_sounds(audio)
            
        elif self.language == 'Telugu':
            # Telugu - gemination detection
            features['gemination_presence'] = self._detect_geminated_consonants(audio)
            features['rhythm_regularity'] = self._analyze_syllable_rhythm(audio)
        
        # Common features for all languages
        features['pitch_stability'] = self._analyze_pitch_contour(audio)
        features['syllable_rate'] = self._calculate_syllables_per_second(audio)
        
        return features
    
    def _detect_retroflex_sounds(self, audio):
        """Detect retroflex consonants (characteristic of Tamil/Malayalam)"""
        # Retroflex sounds have unique spectral characteristics
        # Extract high-resolution spectrogram
        D = librosa.stft(audio)
        S = np.abs(D)
        
        # Retroflex consonants typically show energy in 2-4 kHz range
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=D.shape[0])
        retroflex_band = np.where((freq_bins > 2000) & (freq_bins < 4000))[0]
        
        if len(retroflex_band) > 0:
            retroflex_energy = np.mean(S[retroflex_band, :])
            total_energy = np.mean(S)
            return retroflex_energy / (total_energy + 1e-10)
        return 0.0
    
    def _analyze_vowel_duration(self, audio):
        """Analyze vowel length variation (important for Dravidian languages)"""
        # Find stable energy regions (vowels)
        frame_energy = np.abs(signal.hilbert(audio)) ** 2
        frame_energy_smooth = signal.savgol_filter(frame_energy, min(11, len(frame_energy) | 1), 3)
        
        # Detect vowel regions (high energy, stable)
        vowel_threshold = np.mean(frame_energy_smooth) * 0.7
        vowel_regions = frame_energy_smooth > vowel_threshold
        
        # Calculate duration variance
        if np.any(vowel_regions):
            durations = []
            in_vowel = False
            start_idx = 0
            
            for i, is_vowel in enumerate(vowel_regions):
                if is_vowel and not in_vowel:
                    in_vowel = True
                    start_idx = i
                elif not is_vowel and in_vowel:
                    in_vowel = False
                    durations.append(i - start_idx)
            
            if durations:
                return np.std(durations) / (np.mean(durations) + 1e-10)
        return 0.0
    
    def _detect_aspirated_consonants(self, audio):
        """Detect aspirated consonants (Hindi/Bengali characteristic)"""
        # Aspirated sounds have characteristic high-frequency noise
        D = librosa.stft(audio)
        S = np.abs(D)
        
        # Aspiration typically in 4-8 kHz range
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=D.shape[0])
        aspiration_band = np.where((freq_bins > 4000) & (freq_bins < 8000))[0]
        
        if len(aspiration_band) > 0:
            aspiration_energy = np.mean(S[aspiration_band, :])
            total_energy = np.mean(S)
            return aspiration_energy / (total_energy + 1e-10)
        return 0.0
    
    def _analyze_nasal_sounds(self, audio):
        """Analyze nasal resonances (present in Hindi/Bengali)"""
        # Nasal sounds have distinctive spectral peaks (formants)
        D = librosa.stft(audio)
        S = np.abs(D)
        
        # Nasal formants typically in 0.5-2 kHz range
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=D.shape[0])
        nasal_band = np.where((freq_bins > 500) & (freq_bins < 2000))[0]
        
        if len(nasal_band) > 0:
            nasal_energy = np.mean(S[nasal_band, :])
            high_freq_energy = np.mean(S[nasal_band[-len(nasal_band)//2:], :])
            
            # Nasal sounds have peak in lower frequencies
            if nasal_energy > 0:
                return nasal_energy / (high_freq_energy + 1e-10)
        return 0.0
    
    def _detect_geminated_consonants(self, audio):
        """Detect geminated (doubled) consonants (Telugu characteristic)"""
        # Geminated consonants show repeated strong peaks in energy
        frame_energy = np.abs(signal.hilbert(audio)) ** 2
        frame_energy_smooth = signal.savgol_filter(frame_energy, min(51, len(frame_energy) | 1), 3)
        
        # Find peaks
        peaks, properties = signal.find_peaks(frame_energy_smooth, distance=50, prominence=np.std(frame_energy_smooth))
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            # Gemination shows consistent intervals
            return 1.0 / (np.std(peak_intervals) + 0.1) if np.std(peak_intervals) > 0 else 0.0
        return 0.0
    
    def _analyze_syllable_rhythm(self, audio):
        """Analyze syllable rate regularity (Telugu has strong rhythmic patterns)"""
        frame_energy = np.abs(signal.hilbert(audio)) ** 2
        frame_energy_smooth = signal.savgol_filter(frame_energy, min(51, len(frame_energy) | 1), 3)
        
        # Find syllable peaks
        peaks, _ = signal.find_peaks(frame_energy_smooth, distance=100)
        
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            # Regularity: inverse of coefficient of variation
            return 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-10))
        return 0.0
    
    def _analyze_pitch_contour(self, audio):
        """Analyze pitch stability (different for each language)"""
        # Use autocorrelation for pitch estimation
        frame_length = int(self.sr * 0.05)  # 50ms frames
        hop_length = frame_length // 2
        
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            frames.append(frame)
        
        if len(frames) < 2:
            return 0.0
        
        # Estimate pitch for each frame
        pitches = []
        for frame in frames:
            # Autocorrelation pitch detection
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak after zero-lag
            min_lag = int(self.sr / 400)  # Max pitch ~400Hz
            max_lag = int(self.sr / 80)   # Min pitch ~80Hz
            
            if max_lag < len(autocorr):
                peak_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
                if peak_lag > 0:
                    pitches.append(self.sr / peak_lag)
        
        if len(pitches) > 1:
            # Pitch stability: inverse of variation coefficient
            return 1.0 - (np.std(pitches) / (np.mean(pitches) + 1e-10))
        return 0.0
    
    def _calculate_syllables_per_second(self, audio):
        """Calculate speech rate in syllables/second"""
        # Use energy-based syllable detection
        frame_energy = np.abs(signal.hilbert(audio)) ** 2
        frame_energy_smooth = signal.savgol_filter(frame_energy, min(51, len(frame_energy) | 1), 3)
        
        # Find peaks (syllable nuclei)
        peaks, _ = signal.find_peaks(frame_energy_smooth, distance=100, prominence=np.std(frame_energy_smooth)*0.5)
        
        duration = len(audio) / self.sr
        if duration > 0:
            return len(peaks) / duration
        return 0.0
    
    def score_language_specific_features(self, features):
        """Score language-specific features for AI detection"""
        score = 0.0
        weight = self.weights.get('language_specific', 0.0)
        
        if weight == 0:
            return 0.0
        
        if self.language in ['Tamil', 'Malayalam']:
            # AI struggles with retroflex consonants
            if features.get('retroflex_ratio', 0) < 0.15:
                score += 0.5  # AI tends to skip retroflex
            
            # AI has less vowel length variation
            if features.get('vowel_length_variation', 0) < 0.3:
                score += 0.5
                
        elif self.language == 'Hindi':
            # AI lacks natural aspiration
            if features.get('aspiration_presence', 0) < 0.2:
                score += 0.5
            
            # AI has less nasalization
            if features.get('nasalization_level', 0) < 0.5:
                score += 0.5
                
        elif self.language == 'Telugu':
            # AI struggles with gemination
            if features.get('gemination_presence', 0) < 0.3:
                score += 0.5
            
            # AI has less rhythmic regularity
            if features.get('rhythm_regularity', 0) < 0.4:
                score += 0.5
        
        # Pitch stability: very high = possibly AI (too perfect)
        if features.get('pitch_stability', 0) > 0.85:
            score += 0.3
        
        # Syllable rate: too regular = possibly AI
        syllable_rate = features.get('syllable_rate', 0)
        expected_rate = self.LANGUAGE_FEATURES[self.language]['typical_syllable_rate']
        if abs(syllable_rate - expected_rate) > expected_rate * 0.3:
            score += 0.2
        
        return min(score / 2.0, 1.0) * weight  # Normalized and weighted
    
    def get_supported_languages(self):
        """Return list of supported languages"""
        return list(self.LANGUAGE_FEATURES.keys())
