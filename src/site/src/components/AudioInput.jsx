/**
 * AudioInput.jsx
 * Component for audio input: upload or record
 * Sends audio file to backend via WebSocket Base64 encoded
 * Optional : show a waveform preview of the uploaded audio
 */
import React, { useRef, useState, useContext } from 'react';
import { PipelineContext } from '../context/PipelineContext';

function AudioInput({ onAudioSelected }) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const { state } = useContext(PipelineContext);

  // Consider processing if not idle and not errored
  const isProcessing = state.stage && state.stage !== 'idle' && !state.error;

  const handleFileChange = (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) {
      setSelectedFile(file);
      onAudioSelected && onAudioSelected(file);
    }
  };

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="audio-input-container">
      <input
        type="file"
        accept="audio/wav"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={isProcessing}
      />
      <button onClick={handleButtonClick} disabled={isProcessing}>
        {selectedFile ? selectedFile.name : 'Select Audio File'}
      </button>
    </div>
  );
}

export default AudioInput;
