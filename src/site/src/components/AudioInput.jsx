/**
 * AudioInput.jsx
 * Component for audio input: upload or record
 * Sends audio file to backend via WebSocket Base64 encoded
 * Optional : show a waveform preview of the uploaded audio
 */
import React, { useRef, useState, useContext } from 'react';
import { PipelineContext } from '../context/PipelineContext';

function AudioInput({ onAudioSelected }) {
  // [frontend] Log component render
  console.log('[frontend] AudioInput component rendered');
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  const { state, ws } = useContext(PipelineContext);

  // Consider processing if not idle and not errored
  const isProcessing = state.stage && state.stage !== 'idle' && !state.error;

  const handleFileChange = (event) => {
    try {
      const file = event.target.files && event.target.files[0];
      if (file) {
        // [frontend] Log file selection
        console.log('[frontend] AudioInput file selected:', file.name);
        // Check MIME type and extension
        const isWav = file.type === 'audio/wav' || file.name.toLowerCase().endsWith('.wav');
        if (!isWav) {
          setError('Please upload a .wav audio file.');
          setSelectedFile(null);
          if (onAudioSelected) onAudioSelected(null);
          return;
        }
        setError(null);
        setSelectedFile(file);
        if (onAudioSelected) onAudioSelected(file);

        const reader = new FileReader();
        reader.onload = function(e) {
          try {
            const base64String = arrayBufferToBase64(e.target.result);
            // [frontend] Log file read and send
            console.log('[frontend] AudioInput sending file to backend');
            if (ws && ws.current && ws.current.readyState === 1) {
              ws.current.send(JSON.stringify({
                action: "upload_audio",
                data: { content: base64String }
              }));
            } else {
              setError('WebSocket is not connected.');
              console.error('[frontend] AudioInput WebSocket not connected');
            }
          } catch (err) {
            setError('Failed to process audio file.');
            console.error('[frontend] AudioInput error processing file:', err);
          }
        };
        reader.onerror = function(e) {
          setError('Failed to read audio file.');
          console.error('[frontend] AudioInput FileReader error:', e);
        };
        reader.readAsArrayBuffer(file);
      }
    } catch (err) {
      setError('Unexpected error during file selection.');
      console.error('[frontend] AudioInput handleFileChange error:', err);
    }
  };

  const handleButtonClick = () => {
    try {
      if (fileInputRef.current) {
        fileInputRef.current.click();
      }
    } catch (err) {
      setError('Failed to open file dialog.');
      console.error('[frontend] AudioInput handleButtonClick error:', err);
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
      {error && <div className="audio-input-error" style={{ color: 'red' }}>{error}</div>}
    </div>
  );
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

export default AudioInput;
