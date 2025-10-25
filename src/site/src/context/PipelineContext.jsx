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
  // [frontend] Log context provider render
  console.log('[frontend] PipelineProvider rendered');
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
        console.log('[frontend] WebSocket connected');
        setState((prev) => ({
          ...prev,
          wsConnected: true,
          error: null,
          stage: 'idle',
        }));
      };

      ws.current.onerror = (error) => {
        console.error('[frontend] WebSocket error:', error);
        setState((prev) => ({
          ...prev,
          error: 'WebSocket connection error',
          wsConnected: false,
        }));
      };

      ws.current.onclose = () => {
        console.log('[frontend] WebSocket disconnected');
        setState((prev) => ({ ...prev, wsConnected: false }));
      };

      ws.current.onmessage = (event) => {
        try {
          const response = JSON.parse(event.data);
          console.log('[frontend] Backend message received:', response);

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

            console.log('[frontend] State updated:', {
              stage: newState.stage,
              message: newState.message,
              progress: newState.progress,
              finalPrediction: newState.finalPrediction,
            });

            return newState;
          });
        } catch (err) {
          console.error('[frontend] Error parsing backend message:', err);
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
      console.error('[frontend] Failed to create WebSocket:', err);
      setState((prev) => ({
        ...prev,
        error: 'Failed to connect to WebSocket',
      }));
    }
  }, []);

  // Send action to backend
  const sendAction = (action, data = {}) => {
    try {
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
        console.error('[frontend] WebSocket not connected');
        setState((prev) => ({
          ...prev,
          error: 'WebSocket not connected',
        }));
        return;
      }

      const message = { action, data };
      console.log(`[frontend] Sending "${action}":`, message);
      ws.current.send(JSON.stringify(message));
    } catch (err) {
      console.error('[frontend] sendAction error:', err);
      setState((prev) => ({
        ...prev,
        error: 'Failed to send action to backend',
      }));
    }
  };

  // Upload audio file
  const uploadAudio = (file) => {
    try {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const base64 = (e.target && e.target.result ? e.target.result : '').split(',')[1];
          setState((prev) => ({
            ...prev,
            stage: 'AUDIO_UPLOAD',
            progress: 0,
            message: 'Uploading audio...',
          }));
          console.log('[frontend] uploadAudio: sending audio to backend');
          sendAction('upload_audio', {
            filename: file.name,
            content: base64,
          });
        } catch (err) {
          setState((prev) => ({ ...prev, error: 'Failed to process audio file' }));
          console.error('[frontend] uploadAudio error:', err);
        }
      };
      reader.onerror = (e) => {
        setState((prev) => ({ ...prev, error: 'Failed to read audio file' }));
        console.error('[frontend] uploadAudio FileReader error:', e);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setState((prev) => ({ ...prev, error: 'Unexpected error during audio upload' }));
      console.error('[frontend] uploadAudio unexpected error:', err);
    }
  };

  // Trigger MFCC extraction (backend already has frames)
  const triggerMFCCExtraction = () => {
    try {
      setState((prev) => ({
        ...prev,
        message: 'Extracting MFCC features...',
      }));
      console.log('[frontend] triggerMFCCExtraction: sending mfcc_extraction');
      sendAction('mfcc_extraction', {});
    } catch (err) {
      setState((prev) => ({ ...prev, error: 'Failed to trigger MFCC extraction' }));
      console.error('[frontend] triggerMFCCExtraction error:', err);
    }
  };

  // Trigger model inference (backend already has MFCCs)
  const triggerModelInference = () => {
    try {
      setState((prev) => ({
        ...prev,
        message: 'Running model inference...',
      }));
      console.log('[frontend] triggerModelInference: sending model_inference');
      sendAction('model_inference', {});
    } catch (err) {
      setState((prev) => ({ ...prev, error: 'Failed to trigger model inference' }));
      console.error('[frontend] triggerModelInference error:', err);
    }
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
    console.log('[frontend] Pipeline reset to initial state');
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
