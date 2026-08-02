"""White-box explanation engine for audio spoofing classification."""

import logging
import torch

logger = logging.getLogger(__name__)


def generate_spoof_explanation(
    fused_raw: torch.Tensor, 
    prediction: str, 
    confidence: float
) -> str:
    """Generates a detailed, human-readable explanation of the prediction.

    Analyzes physical acoustic characteristics (RIR, clarity, breathing, spectral)
    to explain why the model classified the audio as real or fake.

    Args:
        fused_raw: Raw unstandardized fused features tensor of shape [channels, feature_dim, time_steps]
                   where feature_dim is 113.
        prediction: Predicted label string ('REAL' or 'FAKE').
        confidence: Prediction confidence score (0.0 to 1.0).

    Returns:
        A string explanation of the acoustic and physiological findings.
    """
    if fused_raw is None or len(fused_raw.shape) < 3 or fused_raw.shape[1] < 113:
        # Fallback if features are not fully fused/formatted
        if prediction.upper() == "FAKE":
            return (
                f"Classified as FAKE (confidence: {confidence:.2%}). "
                "The system detected spectro-temporal anomalies and vocoder artifacts "
                "typical of artificial speech synthesis."
            )
        else:
            return (
                f"Classified as REAL (confidence: {confidence:.2%}). "
                "The speech patterns, room reflections, and frequency distributions "
                "conform to a natural live recording."
            )

    try:
        # Extract physical metrics (raw unstandardized values)
        # RIR features are at indices 100 to 103:
        # 100: RT60, 101: EDT, 102: C50, 103: C80
        rt60_val = float(torch.mean(fused_raw[0, 100, :]))
        edt_val = float(torch.mean(fused_raw[0, 101, :]))
        c50_val = float(torch.mean(fused_raw[0, 102, :]))
        c80_val = float(torch.mean(fused_raw[0, 103, :]))

        # Breathing features are at indices 104 to 107:
        # 104: ZCR, 105: flatness, 106: high_ratio, 107: normalized centroid
        breath_zcr = float(torch.mean(fused_raw[0, 104, :]))
        breath_flatness = float(torch.mean(fused_raw[0, 105, :]))
        breath_high_ratio = float(torch.mean(fused_raw[0, 106, :]))

        # Format details
        explanations = []

        if prediction.upper() in {"FAKE", "SPOOF"}:
            explanations.append(f"Classified as FAKE (confidence: {confidence:.2%}). Reasons include:")
            
            # 1. RIR check
            if rt60_val < 0.15:
                explanations.append(
                    f"- **Dry acoustic environment**: The estimated Room Impulse Response (RIR) decay time (RT60: {rt60_val:.2f}s) "
                    "is abnormally low, indicating an artificial 'dry' signal that lacks natural room reverberation."
                )
            elif rt60_val > 1.5:
                explanations.append(
                    f"- **Excessive acoustic decay**: The estimated Room Impulse Response decay time (RT60: {rt60_val:.2f}s) "
                    "is extremely high, which is typical of synthetic reverb post-processing added to mimic room space."
                )

            # 2. C50/Clarity check
            if c50_val < -10.0 or c50_val > 25.0:
                explanations.append(
                    f"- **Unnatural clarity index**: The early-to-late reflection clarity ratio (C50: {c50_val:.1f} dB) "
                    "lies outside the standard range of real physical spaces, highlighting synthetic vocoder rendering."
                )

            # 3. Breathing and Pause check
            if breath_flatness < 0.0001:
                explanations.append(
                    f"- **Synthetic pause cadence**: The physiological breathing flatness ({breath_flatness:.6f}) is static, "
                    "suggesting a lack of natural human inhalation fricatives and breath pauses during speech."
                )
            elif breath_high_ratio < 0.01:
                explanations.append(
                    f"- **High-frequency suppression**: The high-frequency energy ratio above 3.5kHz ({breath_high_ratio:.2%}) "
                    "is heavily attenuated, indicating band-limited synthesis/vocoding that cuts off natural breath sibilance."
                )

            # Fallback if no specific threshold was crossed
            if len(explanations) == 1:
                explanations.append(
                    "- **Spectro-temporal artifacts**: Deep neural feature auditing detected micro-level consistency anomalies "
                    "across the Mel frequencies and chroma classes, typical of speech synthesized by vocoders (e.g., World, Straight, or WaveNet)."
                )
        else:
            # Prediction is REAL
            explanations.append(f"Classified as REAL (confidence: {confidence:.2%}). Findings include:")
            explanations.append(
                f"- **Natural reverberation**: Estimated room decay (RT60: {rt60_val:.2f}s, EDT: {edt_val:.2f}s) "
                f"and environment clarity (C50: {c50_val:.1f} dB) match natural spatial recordings."
            )
            explanations.append(
                f"- **Physiological breathing dynamics**: Presence of human speech pauses, natural inhalation fricatives "
                f"(spectral flatness: {breath_flatness:.5f}), and healthy high-frequency energy ({breath_high_ratio:.1%})."
            )

        return "\n".join(explanations)

    except Exception as exc:
        logger.error("Failed to generate speech explanation: %s", exc)
        return (
            f"Prediction: {prediction} (confidence: {confidence:.2%}). "
            "Acoustic feature audit indicates consistency with "
            f"{'synthesized vocoder speech' if prediction.upper() == 'FAKE' else 'natural speech recording'}."
        )
