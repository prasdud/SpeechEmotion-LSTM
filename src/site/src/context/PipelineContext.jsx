/**
 * PipelineContext.tsx
 * Global context is managed here
 */


import React, { createContext, useContext, useState, useRef, useEffect } from 'react';

export const PipelineContext = createContext(undefined);


export const usePipeline = () => {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error('usePipeline must be used within PipelineProvider');
  return ctx;
};


export const PipelineProvider = ({ children }) => {
  const [state, setState] = useState({
    wsConnected: false,
    error: null,
    stage: null,
    message: '',
    progress: 0,
    partialPredictions: [],
    finalPrediction: null,
    lastMessage: null,
  });

  const ws = useRef(null);

  // Initialize WebSocket connection
  useEffect(() => {
    try {
      ws.current = new WebSocket('ws://localhost:8000/ws');

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setState((prev) => ({
          ...prev,
          wsConnected: true,
          error: null,
          stage: 'idle',
        }));
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setState((prev) => ({
          ...prev,
          error: 'WebSocket connection error',
          wsConnected: false,
        }));
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setState((prev) => ({ ...prev, wsConnected: false }));
      };

      ws.current.onmessage = (event) => {
        try {
          const response = JSON.parse(event.data);
          console.log('Backend message received:', response);

          // Extract fields from backend response (matches test.py logic)
          const data = response.data || {};
          const stage = data.stage || response.stage;
          const message = data.message || response.message || '';
          const finalPrediction =
            data.final_prediction || response.final_prediction;
          const progress = data.progress || 0;
          const partialPrediction =
            data.partial_prediction || response.partial_prediction;

          // Update state with backend message
          setState((prev) => {
            const updatedPartialPredictions = [...prev.partialPredictions];
            if (partialPrediction && !updatedPartialPredictions.includes(partialPrediction)) {
              updatedPartialPredictions.push(partialPrediction);
            }

            const newState = {
              ...prev,
              stage: stage || prev.stage,
              message,
              progress,
              partialPredictions: updatedPartialPredictions,
              finalPrediction: finalPrediction || prev.finalPrediction,
              lastMessage: response,
              wsConnected: true,
              error: null,
            };

            console.log('State updated:', {
              stage: newState.stage,
              message: newState.message,
              progress: newState.progress,
              finalPrediction: newState.finalPrediction,
            });

            return newState;
          });
        } catch (err) {
          console.error('Error parsing message:', err);
          setState((prev) => ({
            ...prev,
            error: 'Failed to parse backend message',
          }));
        }
      };

      return () => {
        if (ws.current) ws.current.close();
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setState((prev) => ({
        ...prev,
        error: 'Failed to connect to WebSocket',
      }));
    }
  }, []);

  // Send action to backend
  const sendAction = (action, data = {}) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      setState((prev) => ({
        ...prev,
        error: 'WebSocket not connected',
      }));
      return;
    }

    const message = { action, data };
    console.log(`Sending "${action}":`, message);
    ws.current.send(JSON.stringify(message));
  };

  // Upload audio file
  const uploadAudio = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = (e.target && e.target.result ? e.target.result : '').split(',')[1];
      setState((prev) => ({
        ...prev,
        stage: 'AUDIO_UPLOAD',
        progress: 0,
        message: 'Uploading audio...',
      }));
      sendAction('upload_audio', {
        filename: file.name,
        content: base64,
      });
    };
    reader.readAsDataURL(file);
  };

  // Trigger MFCC extraction (backend already has frames)
  const triggerMFCCExtraction = () => {
    setState((prev) => ({
      ...prev,
      message: 'Extracting MFCC features...',
    }));
    sendAction('mfcc_extraction', {});
  };

  // Trigger model inference (backend already has MFCCs)
  const triggerModelInference = () => {
    setState((prev) => ({
      ...prev,
      message: 'Running model inference...',
    }));
    sendAction('model_inference', {});
  };

  // Reset pipeline to initial state
  const reset = () => {
    setState({
      wsConnected: ws.current && ws.current.readyState === WebSocket.OPEN,
      error: null,
      stage: 'idle',
      message: '',
      progress: 0,
      partialPredictions: [],
      finalPrediction: null,
      lastMessage: null,
    });
  };

  return (
    <PipelineContext.Provider
      value={{
        state,
        setState,
        ws,
        sendAction,
        uploadAudio,
        triggerMFCCExtraction,
        triggerModelInference,
        reset,
      }}
    >
      {children}
    </PipelineContext.Provider>
  );
};
