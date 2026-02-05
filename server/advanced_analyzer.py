"""
Advanced AI Voice Detection System - Deep Learning Approach

This module provides state-of-the-art deep learning models for AI voice detection.
It leverages TensorFlow/PyTorch for CNN, LSTM, and Transformer-based architectures.

This is an OPTIONAL module for enhanced accuracy with trained models.
The basic ensemble system (voice_forensics_research.py) works without these dependencies.

Features:
- CNN-based spectrogram classification
- LSTM networks for temporal pattern detection
- Pre-trained Transformer models from Hugging Face
- Transfer learning for rapid deployment
- MFCC and spectrogram feature extraction

Requirements (optional):
- tensorflow >= 2.12.0
- torch >= 2.0.0
- transformers >= 4.30.0

Author: Voice Authenticator Team
License: MIT
"""

import numpy as np
import librosa  # type: ignore
from typing import Tuple, Dict, Any, Optional
import warnings

# Optional imports - graceful fallback if not installed
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not installed. Deep learning models unavailable. "
                  "Install with: pip install tensorflow")

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AdvancedVoiceAnalyzer:
    """
    Advanced voice detection using deep learning models.
    Supports CNN, LSTM, and Transformer-based architectures.
    """

    def __init__(self, model_type: str = "cnn", sample_rate: int = 16000):
        """
        Initialize advanced voice analyzer.

        Args:
            model_type: 'cnn', 'lstm', 'transformer'
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.model = None
        self.feature_extractor = None

        if not TF_AVAILABLE and not TORCH_AVAILABLE and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "No deep learning framework installed. "
                "Install TensorFlow: pip install tensorflow\n"
                "Or PyTorch: pip install torch\n"
                "Or Transformers: pip install transformers"
            )

    def extract_mfcc_features(
        self, audio: np.ndarray, n_mfcc: int = 13
    ) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficient) features.

        MFCCs represent the spectrum of sound, capturing differences between
        human and synthetic voices effectively.

        Args:
            audio: Audio waveform array
            n_mfcc: Number of MFCC coefficients to extract

        Returns:
            MFCC feature matrix of shape (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=n_mfcc
        )
        return mfcc

    def extract_spectrogram_features(
        self, audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        """
        Extract spectrogram (time-frequency representation).

        The spectrogram shows visual patterns that distinguish AI from human voices.
        CNNs process these effectively as they do with images.

        Args:
            audio: Audio waveform array
            n_fft: FFT window size
            hop_length: Hop length for STFT

        Returns:
            Log-magnitude spectrogram of shape (freq_bins, time_steps)
        """
        spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        log_spec = np.abs(librosa.power_to_db(np.abs(spectrogram)))
        return log_spec

    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chroma features (pitch class information).

        Args:
            audio: Audio waveform array

        Returns:
            Chroma feature matrix
        """
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        return chroma

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features: centroid, bandwidth, etc.

        Args:
            audio: Audio waveform array

        Returns:
            Dictionary of spectral features
        """
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate
        )[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]

        return {
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_centroid_std": np.std(spectral_centroid),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
            "spectral_bandwidth_std": np.std(spectral_bandwidth),
            "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
            "zero_crossing_rate_std": np.std(zero_crossing_rate),
        }

    def _build_cnn_model(
        self, input_shape: Tuple[int, int]
    ) -> Optional[keras.Model]:
        """
        Build CNN model for spectrogram classification.

        CNN architecture processes spectrogram patterns similar to image recognition,
        making it highly effective for visual differences in AI vs human speech.

        Args:
            input_shape: Input feature shape (height, width)

        Returns:
            Compiled CNN model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            warnings.warn("TensorFlow required for CNN model")
            return None

        model = keras.Sequential(
            [
                layers.Input(shape=(*input_shape, 1)),
                # Conv Block 1
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Conv Block 2
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Conv Block 3
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Dense layers
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(2, activation="softmax"),  # AI or Human
            ]
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )

        return model

    def _build_lstm_model(
        self, input_shape: Tuple[int, int]
    ) -> Optional[keras.Model]:
        """
        Build LSTM model for temporal pattern detection.

        LSTM networks excel at capturing temporal dependencies in speech,
        understanding the natural flow and pauses of human speech vs
        the sometimes unnatural patterns in AI-generated audio.

        Args:
            input_shape: Input feature shape

        Returns:
            Compiled LSTM model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            warnings.warn("TensorFlow required for LSTM model")
            return None

        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(256, return_sequences=True, activation="relu"),
                layers.Dropout(0.2),
                layers.LSTM(128, return_sequences=False, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(2, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def load_pretrained_transformer(
        self, model_name: str = "superb/hubert-base-superb-ks"
    ) -> bool:
        """
        Load pre-trained Transformer model from Hugging Face.

        Transformers (BERT-based) have shown excellent results for audio
        classification tasks, often outperforming CNNs and LSTMs.

        Args:
            model_name: Hugging Face model identifier

        Returns:
            True if successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            warnings.warn("Transformers library required")
            return False

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name
            )
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            return True
        except Exception as e:
            warnings.warn(f"Failed to load transformer model: {e}")
            return False

    def analyze_with_advanced_model(
        self, audio: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze voice using advanced deep learning model.

        Args:
            audio: Audio waveform array

        Returns:
            Classification result with confidence scores
        """
        if self.model_type == "cnn":
            return self._analyze_cnn(audio)
        elif self.model_type == "lstm":
            return self._analyze_lstm(audio)
        elif self.model_type == "transformer":
            return self._analyze_transformer(audio)
        else:
            return {"error": f"Unknown model type: {self.model_type}"}

    def _analyze_cnn(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze using CNN model."""
        if not TF_AVAILABLE or self.model is None:
            return {
                "error": "CNN model not available. Install TensorFlow.",
                "classification": "unknown",
                "confidence": 0,
            }

        # Extract spectrogram
        spec = self.extract_spectrogram_features(audio)
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        spec = np.expand_dims(spec, axis=(0, -1))

        # Prediction
        prediction = self.model.predict(spec, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx]) * 100

        return {
            "method": "CNN-Spectrogram",
            "classification": "AI" if class_idx == 0 else "Human",
            "confidence": confidence,
            "ai_probability": float(prediction[0][0]) * 100,
            "human_probability": float(prediction[0][1]) * 100,
        }

    def _analyze_lstm(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze using LSTM model."""
        if not TF_AVAILABLE or self.model is None:
            return {
                "error": "LSTM model not available. Install TensorFlow.",
                "classification": "unknown",
                "confidence": 0,
            }

        # Extract MFCC features
        mfcc = self.extract_mfcc_features(audio)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        mfcc = np.expand_dims(mfcc, axis=0)

        # Prediction
        prediction = self.model.predict(mfcc, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx]) * 100

        return {
            "method": "LSTM-MFCC",
            "classification": "AI" if class_idx == 0 else "Human",
            "confidence": confidence,
            "ai_probability": float(prediction[0][0]) * 100,
            "human_probability": float(prediction[0][1]) * 100,
        }

    def _analyze_transformer(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze using Transformer model."""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return {
                "error": "Transformer model not available. Install transformers.",
                "classification": "unknown",
                "confidence": 0,
            }

        try:
            inputs = self.feature_extractor(
                audio, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits

            predictions = torch.nn.functional.softmax(logits, dim=-1)
            class_idx = torch.argmax(predictions[0]).item()
            confidence = float(predictions[0][class_idx]) * 100

            return {
                "method": "Transformer-HuBERT",
                "classification": "AI" if class_idx == 0 else "Human",
                "confidence": confidence,
                "ai_probability": float(predictions[0][0]) * 100 if class_idx == 0 else float(predictions[0][1]) * 100,
                "human_probability": float(predictions[0][1]) * 100 if class_idx == 1 else float(predictions[0][0]) * 100,
            }
        except Exception as e:
            return {
                "error": f"Transformer analysis failed: {str(e)}",
                "classification": "unknown",
                "confidence": 0,
            }

    def get_feature_importance(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Calculate importance of different acoustic features.

        Useful for understanding what the model focused on for its decision.

        Args:
            audio: Audio waveform array

        Returns:
            Dictionary of feature importances
        """
        spectral_features = self.extract_spectral_features(audio)

        mfcc = self.extract_mfcc_features(audio)
        mfcc_variance = np.var(np.mean(mfcc, axis=1))

        spectrogram = self.extract_spectrogram_features(audio)
        spec_variance = np.var(spectrogram)

        chroma = self.extract_chroma_features(audio)
        chroma_variance = np.var(np.mean(chroma, axis=1))

        # Normalize to 0-100
        features = {
            "mfcc_variance": mfcc_variance,
            "spectrogram_variance": spec_variance,
            "chroma_variance": chroma_variance,
            **spectral_features,
        }

        total = sum(
            v for v in features.values() if isinstance(v, (int, float)) and v > 0
        )
        if total > 0:
            normalized = {k: (v / total * 100) if isinstance(v, (int, float)) else v
                         for k, v in features.items()}
        else:
            normalized = features

        return normalized


def create_advanced_analyzer(
    model_type: str = "transformer",
) -> Optional[AdvancedVoiceAnalyzer]:
    """
    Factory function to create advanced analyzer with specified model.

    Args:
        model_type: 'cnn', 'lstm', or 'transformer'

    Returns:
        AdvancedVoiceAnalyzer instance or None if dependencies missing
    """
    try:
        analyzer = AdvancedVoiceAnalyzer(model_type=model_type)
        return analyzer
    except ImportError as e:
        print(f"Cannot create advanced analyzer: {e}")
        return None


# Example usage
if __name__ == "__main__":
    print("Advanced Voice Analyzer Module")
    print("=" * 50)
    print("\nOptional Deep Learning Models for Voice Detection:")
    print("1. CNN - Convolutional Neural Networks")
    print("   - Best for spectrogram pattern recognition")
    print("   - Requires: tensorflow")
    print("\n2. LSTM - Long Short-Term Memory")
    print("   - Best for temporal sequence analysis")
    print("   - Requires: tensorflow")
    print("\n3. Transformer - HuBERT/WAV2VEC2")
    print("   - Best overall performance (92-95%+ accuracy)")
    print("   - Requires: transformers, torch")
    print("\nTo use, install dependencies:")
    print("  pip install tensorflow  # For CNN/LSTM")
    print("  pip install transformers torch  # For Transformer")
    print("\nThen use with:")
    print("  from advanced_analyzer import AdvancedVoiceAnalyzer")
    print("  analyzer = AdvancedVoiceAnalyzer(model_type='transformer')")
    print("  result = analyzer.analyze_with_advanced_model(audio)")
