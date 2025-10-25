/**
 * AudioInput.tsx
 * Component for audio input: upload or record
 * Sends audio file to backend via WebSocket Base64 encoded
 * Optional : show a waveform preview of the uploaded audio
 */
import React, { useState, useRef } from 'react';
import { usePipeline } from '../context/PipelineContext';


const AudioInput: React.FC = () => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { uploadAudio, state } = usePipeline();

  function handleFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files ? event.target.files[0] : null;
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.wav')) {
      setAudioFile(null);
      if (inputRef.current) {
        inputRef.current.value = '';
      }
      alert('Please upload a .wav file');
      return;
    }
    setAudioFile(file);
    uploadAudio(file);
  }

  return (
    <div>
      <h2>Audio Input</h2>
      <label htmlFor="upload-audio">
        <input
          type="file"
          accept=".wav"
          id="upload-audio"
          onChange={handleFile}
          ref={inputRef}
        />
        <span>Upload Audio File</span>
      </label>
      {audioFile && <p>Selected file: {audioFile.name}</p>}
      {state.error && <p style={{ color: 'red' }}>{state.error}</p>}
    </div>
  );
};

export default AudioInput;
