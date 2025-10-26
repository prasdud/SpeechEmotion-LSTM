'''
compute MFCC features from audio frames.
'''
import librosa
import numpy as np
import logging
from src.api.utils.utils import log_function, send_update

logging.basicConfig(level=logging.INFO)


@log_function
async def compute_mfcc(websocket, frames, sample_rate=16000, num_mfcc=13):
    '''
    Compute MFCC features from framed audio
    Extracts 13 MFCCs + 13 deltas + 13 delta-deltas = 39 features total
    
    CRITICAL: Must match training exactly:
    1. Flatten frames back to continuous audio
    2. Compute MFCCs on full audio (not per-frame!)
    3. Compute deltas across time dimension
    4. Transpose to (num_frames, 39)
    '''
    try:
        # Step 1: Reconstruct continuous audio from overlapping frames
        # frames shape: (frame_length, num_frames)
        # We need to flatten back to continuous audio for MFCC computation
        frame_length = frames.shape[0]
        num_frames = frames.shape[1]
        
        # For simplicity and matching training: take center of each frame
        # This gives us num_frames audio samples, one per frame
        # Better approach: reconstruct with overlap-add, but this matches training better
        audio = frames[frame_length // 2, :]  # Take center sample from each frame
        
        # Actually, let's reconstruct properly
        # Use the first sample of each frame (hop_length apart)
        # This matches the framing: each frame starts hop_length samples after the previous
        audio = frames[0, :]  # shape: (num_frames,)
        
        # Even better: flatten all frames back to continuous audio
        # Since frames overlap, we need to be smart about reconstruction
        # For RAVDESS: frame_size=25ms=400 samples, hop=10ms=160 samples
        # Let's just concatenate non-overlapping portions
        hop_length_samples = int(0.010 * sample_rate)  # 160 samples (10ms hop)
        audio_length = (num_frames - 1) * hop_length_samples + frame_length
        audio = np.zeros(audio_length)
        
        # Overlap-add reconstruction
        for i in range(num_frames):
            start = i * hop_length_samples
            end = start + frame_length
            if end > audio_length:
                end = audio_length
            audio[start:end] += frames[:end-start, i]
        
        # Normalize (sum of overlaps = frame_length / hop_length_samples)
        overlap_count = frame_length / hop_length_samples
        audio = audio / overlap_count
        
        logging.info(f"Reconstructed audio: {len(audio)} samples from {num_frames} frames")
        
        # Step 2: Compute MFCCs on full audio (EXACTLY like training)
        await send_update(websocket, "processing", {
            "stage": "MFCC_EXTRACTION",
            "progress": 25,
            "message": f"Computing MFCCs for {num_frames} frames..."
        })
        
        # Compute MFCCs for all frames at once (vectorized operation - MATCHES TRAINING!)
        mfcc_features = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=num_mfcc,
            n_fft=frame_length,
            hop_length=hop_length_samples
        )
        # Shape: (13, num_frames)
        
        logging.info(f"MFCC features computed: shape={mfcc_features.shape}")
        
        await send_update(websocket, "processing", {
            "stage": "MFCC_EXTRACTION",
            "progress": 50,
            "message": f"Computing delta features..."
        })
        
        # Step 3: Compute delta (velocity) and delta-delta (acceleration) features
        # This requires multiple time steps, which is why we process all frames at once!
        mfcc_delta = librosa.feature.delta(mfcc_features)
        mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
        
        await send_update(websocket, "processing", {
            "stage": "MFCC_EXTRACTION",
            "progress": 75,
            "message": f"Stacking features..."
        })
        
        # Step 4: Stack: 13 MFCCs + 13 deltas + 13 delta-deltas = 39 features
        mfcc_combined = np.concatenate([mfcc_features, mfcc_delta, mfcc_delta2], axis=0)
        # Shape: (39, num_frames)
        
        # Step 5: Transpose to (num_frames, 39) - MATCHES TRAINING!
        mfcc_combined = mfcc_combined.T.astype(np.float32)
        
        # Step 6: Normalize MFCCs to mean=0, std=1 (CRITICAL - must match training!)
        mean = mfcc_combined.mean(axis=0, keepdims=True)
        std = mfcc_combined.std(axis=0, keepdims=True) + 1e-8  # Add epsilon to prevent division by zero
        mfcc_combined = (mfcc_combined - mean) / std
        
        logging.info(f"Final MFCC features: shape={mfcc_combined.shape} (num_frames={num_frames}, features=39)")
        logging.info(f"MFCC normalization applied: original mean={mean.mean():.4f}, std={std.mean():.4f}")
        logging.info(f"After normalization: mean={mfcc_combined.mean():.4f}, std={mfcc_combined.std():.4f}, min={mfcc_combined.min():.4f}, max={mfcc_combined.max():.4f}")
        
    except Exception as e:
        logging.error(f"Error computing MFCCs: {e}")
        await send_update(websocket, "error", {
            "stage": "MFCC_EXTRACTION",
            "message": f"Error computing MFCCs: {e}"
        })
        return None
    
    if mfcc_combined.shape[0] == 0:
        logging.error("No MFCC features were computed from the audio frames.")
        if websocket:
            await send_update(websocket, "error", {
                "stage": "MFCC_EXTRACTION",
                "message": "No MFCC features were computed from the audio frames."
            })
        return None
    
    mfcc_features = mfcc_combined
    logging.info(f"Computed MFCC features (13 MFCCs + 13 deltas + 13 delta-deltas = 39) for {num_frames} frames, final shape={mfcc_features.shape}")

    # Send final update for frontend to trigger next step
    if websocket:
        await send_update(websocket, "MFCC_EXTRACTION", {
            "stage": "MFCC_EXTRACTION",
            "message": "MFCC extraction completed"
        })
    
    return mfcc_features