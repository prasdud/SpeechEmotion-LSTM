'''
compute MFCC features from audio frames.
'''
from src.api.websocket_handler import send_update
import librosa
import numpy as np
import logging

async def compute_mfcc(websocket, frames, sample_rate=16000, num_mfcc=13):
    '''
    Compute MFCC features from framed audio
    '''
    mfcc_features = []
    total_frames = frames.shape[1]

    for i in range(total_frames):
        frame = frames[:, i]
        if np.all(frame == 0):
            logging.warning(f"Frame {i} is silent, skipping MFCC computation.")
            continue

        mfccs = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=num_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)  # shape = (num_mfcc,)
        mfcc_features.append(mfcc_mean)

        if websocket and i % 10 == 0: # send update every 10 frames
            progress = round((i / total_frames) * 100, 2)
            await send_update(websocket, "processing", {
                "stage": "MFCC_EXTRACTION",
                "progress": progress,
                "message": f"Computed MFCCs for {i}/{total_frames} frames ({progress}%)"
            })

    mfcc_features = np.vstack(mfcc_features)  # shape = (num_frames, num_mfcc)
    logging.info(f"Computed MFCC features for {total_frames} frames, shape={mfcc_features.shape}")
    
    return mfcc_features