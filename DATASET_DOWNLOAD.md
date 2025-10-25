# 📥 RAVDESS Dataset Download Instructions

## Which File to Download?

Download: **Audio_Speech_Actors_01-24.zip**
- Size: 208.5 MB
- MD5: bc696df654c87fed845eb13823edef8a
- Contains: Audio-only speech recordings from all 24 actors

## Why This File?

✅ **Audio-only** - No video (lighter download, faster processing)
✅ **Speech** - Better for emotion recognition than songs
✅ **All 24 actors** - Complete dataset in one file
✅ **Small** - Only 208 MB (vs 10+ GB for all videos)

## Download Steps

### 1. Download from Zenodo

Go to: **https://zenodo.org/record/1188976**

Find and download:
```
Audio_Speech_Actors_01-24.zip (208.5 MB)
```

### 2. Extract to Project

```bash
# Navigate to your project root
cd /home/prasdud/playground/python/SpeechEmotion-LSTM

# Create data directory
mkdir -p data

# Extract (adjust path to where you downloaded)
unzip ~/Downloads/Audio_Speech_Actors_01-24.zip -d data/

# Verify extraction
ls data/
```

### 3. Verify Structure

You should see 24 Actor folders:
```bash
data/
├── Actor_01/
├── Actor_02/
├── Actor_03/
...
├── Actor_23/
└── Actor_24/
```

Each folder contains ~60 .wav files:
```bash
ls data/Actor_01/
```

Output should look like:
```
03-01-01-01-01-01-01.wav
03-01-01-01-01-02-01.wav
03-01-01-01-02-01-01.wav
...
```

### 4. Verify File Count

```bash
# Count total .wav files (should be ~1440)
find data/ -name "*.wav" | wc -l
```

Expected: **1440 files** (60 per actor × 24 actors)

## ✅ Ready to Train!

Once you have 1440 .wav files in 24 Actor folders, you're ready to train:

```bash
cd training
python verify_setup.py  # Should pass dataset check ✅
python train.py         # Start training!
```

## 🚫 Don't Download These

You **don't need**:
- ❌ Audio_Song_Actors_01-24.zip (songs, not speech)
- ❌ Video_Song_Actor_*.zip (large video files, not needed)
- ❌ Video_Speech_Actor_*.zip (large video files, not needed)

## 📊 Dataset Details

### File Naming Convention
`03-01-06-01-02-01-12.wav`

- Position 1-2: Modality (03 = audio-only)
- Position 3-4: Vocal channel (01 = speech)
- **Position 5-6: Emotion (01-08)** ← What we're predicting!
- Position 7-8: Intensity (01 = normal, 02 = strong)
- Position 9-10: Statement (01-02)
- Position 11-12: Repetition (01-02)
- Position 13-14: Actor (01-24)

### Emotion Labels
| Code | Emotion | Class |
|------|---------|-------|
| 01 | Neutral | 0 |
| 02 | Calm | 1 |
| 03 | Happy | 2 |
| 04 | Sad | 3 |
| 05 | Angry | 4 |
| 06 | Fearful | 5 |
| 07 | Disgust | 6 |
| 08 | Surprised | 7 |

### Actor Details
- Actors 01-24
- Odd numbers (01, 03, ...) = Male actors
- Even numbers (02, 04, ...) = Female actors

## 🎯 What's Next?

After downloading and extracting:

1. ✅ Run verification:
   ```bash
   cd training
   python verify_setup.py
   ```

2. ✅ Start training:
   ```bash
   python train.py
   ```

That's it! The training code will automatically load all 1440 audio files. 🚀
