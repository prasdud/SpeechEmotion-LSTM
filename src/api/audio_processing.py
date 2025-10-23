'''
prepare audio for feature extraction
'''
import librosa
import logging
import io
import numpy as np
from src.api.websocket_handler import send_update

SAMPLE_RATE = 16000
FRAME_SIZE = 0.025  # 25ms
HOP_LENGTH = 0.010  # 10ms

async def process_audio(file_bytes):
    '''
    Process raw audio bytes
    '''
    audio, sample_rate = load_audio(file_bytes)
    rs_audio = resample_audio(audio, sample_rate, target_rate=SAMPLE_RATE)
    norm_audio = normalize_audio(rs_audio)
    frames = await frame_audio(norm_audio, sample_rate=SAMPLE_RATE, frame_size=0.025, hop_length=0.010)
    logging.info(f"Processed audio into {frames.shape[1]} frames at {SAMPLE_RATE} Hz")
    return frames, SAMPLE_RATE
    

def load_audio(file_bytes):
    '''
    Load audio bytes
    converts from bytes to waveform
    '''
    audio_stream = io.BytesIO(file_bytes)
    waveform, sample_rate = librosa.load(audio_stream, sr=None)
    logging.info(f"Loaded audio with sample rate: {sample_rate}, waveform length: {len(waveform)}")
    logging.info(f"Waveform shape: {waveform.shape}")
    return waveform, sample_rate
    

def resample_audio(audio, current_rate, target_rate):
    '''
    Resample audio to target rate
    ensures all audio has the same sampling rate for consistent MFCC extraction
    '''
    if current_rate != target_rate:
        resampled_waveform = librosa.resample(audio, orig_sr=current_rate, target_sr=target_rate)
        logging.info(f"Resampled audio from {current_rate} to {target_rate}")
        return resampled_waveform
    return audio
    

def normalize_audio(audio):
    '''
    Normalize audio data
    Scales amplitude to range [-1, 1], so loudness differences do not skew feature extraction
    '''
    norm_audio = audio / np.max(np.abs(audio) + 1e-9) # prevent division by zero
    logging.info(f"Normalized audio. Max amplitude after normalization: {np.max(np.abs(norm_audio))}")
    return norm_audio



async def frame_audio(audio, sr, websocket=None, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH):
    '''
    Frame audio into overlapping frames
    '''
    frame_length = int(frame_size * sr)         # 25 ms frames
    hop_length = int(hop_length * sr)   # 10 ms overlap
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    logging.info(f"Framed audio into {frames.shape[1]} frames of length {frame_length}")

    if websocket:
        await send_update(websocket, "processing", {
            "stage": "audio_processing",
            "message": f"Framed into {frames.shape[1]} frames of length {frame_length}"
        })
    
    logging.info(f"Framed audio into {frames.shape[1]} frames of length {frame_length}")

    return frames


'''
TO ADD
Optional WebSocket in all steps

You may eventually want to send updates after:

Load → “Audio loaded”

Resample → “Audio resampled to 16 kHz”

Normalize → “Audio normalized”

Currently updates only happen during framing. Consider adding small async calls after each step.
'''