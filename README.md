# SpeechEmotion-LSTM
A web-based application that analyzes audio recordings to automatically detect the speaker’s emotional state. It extracts MFCC features from speech and uses an LSTM network to classify emotions such as happy, sad, angry, or neutral. The app provides interactive feedback, showing processing progress and final predictions in real time.

# Status, Dev Notes
Completed project. Not actively looking to work on this. If anyone wants to continue this, maybe a better dataset instead of RAVDESS.

## Running Frontend
- goto src/site
- $npm install
- $npm run dev
- visit http://localhost:5173/

## Running backend
- from project root
- create virtual env
- pip3 install -r requirements.txt
- uvicorn src.api.app:app --reload > logs.txt 2>&1

## TODO
- Add docs
