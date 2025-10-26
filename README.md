# SpeechEmotion-LSTM
A web-based application that analyzes audio recordings to automatically detect the speaker’s emotional state. It extracts MFCC features from speech and uses an LSTM network to classify emotions such as happy, sad, angry, or neutral. The app provides interactive feedback, showing processing progress and final predictions in real time.

## Running Frontend
- goto src/site
- $npm run dev
- visit http://localhost:5173/

## Running backend
- from project root
- uvicorn src.api.app:app --reload > logs.txt 2>&1

## TODO
- Add docs