# SpeechEmotion-LSTM
A web-based application that analyzes audio recordings to automatically detect the speaker’s emotional state. It extracts MFCC features from speech and uses an LSTM network to classify emotions such as happy, sad, angry, or neutral. The app provides interactive feedback, showing processing progress and final predictions in real time.


## TODO
- add exception handling for websocket recieve / send
- ensure the uploaded file is .wav and matches expected sample rate / channels
- add better websocket updates
- add better logging
- add LSTM model
- check async await consistency satisfied
- there is a circular import issue in audio_processing.py, handle it
- work on the frontend
- dockerize the entire app