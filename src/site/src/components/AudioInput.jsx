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
    <div className="card audio-input-container">
      <div style={{textAlign: 'center', marginBottom: '2rem'}}>
        <div style={{fontSize: '3rem', marginBottom: '1rem'}}>🎙️</div>
        <h2 style={{marginBottom: '0.5rem', textAlign: 'center'}}>Upload Audio File</h2>
        <p style={{color: 'var(--text-muted)', fontSize: '0.95rem'}}>
          Select a .wav file to begin emotion analysis
        </p>
      </div>
      
      <input
        type="file"
        accept="audio/wav"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={isProcessing}
      />
      
      <div style={{display: 'flex', justifyContent: 'center', marginBottom: '1rem'}}>
        <button 
          onClick={handleButtonClick} 
          disabled={isProcessing}
          style={{
            maxWidth: '320px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.75rem',
            fontSize: '1.1rem'
          }}
        >
          <span style={{fontSize: '1.25rem'}}>📁</span>
          {selectedFile ? selectedFile.name : 'Choose Audio File'}
        </button>
      </div>
      
      {selectedFile && !error && (
        <div style={{
          marginTop: '1.5rem',
          padding: '1rem',
          background: 'var(--bg-accent)',
          borderRadius: '12px',
          border: '1px solid var(--green-light)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <span style={{fontSize: '1.5rem'}}>✅</span>
          <div>
            <div style={{fontWeight: '600', color: 'var(--text-primary)', marginBottom: '0.25rem'}}>File Selected</div>
            <div style={{fontSize: '0.9rem', color: 'var(--text-secondary)'}}>{selectedFile.name}</div>
          </div>
        </div>
      )}
      
      {error && (
        <div className="audio-input-error" style={{
          marginTop: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <span style={{fontSize: '1.5rem'}}>⚠️</span>
          <div>{error}</div>
        </div>
      )}
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
