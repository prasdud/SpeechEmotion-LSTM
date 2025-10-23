import React, { createContext, useContext, useState, useRef, useEffect } from 'react';
import type { ReactNode } from 'react';

// Define the shape of the global state
export interface PipelineState {
  audioData: any;
  preprocessingStatus: any;
  mfccData: any;
  inferenceStatus: any;
  intermediatePredictions: any;
  finalPrediction: any;
  pipelineStage: string;
  wsConnected: boolean;
  wsMessage: any;
}

const PipelineContext = createContext<{
  state: PipelineState;
  setState: React.Dispatch<React.SetStateAction<PipelineState>>;
  ws: React.RefObject<WebSocket | null>;
} | undefined>(undefined);

export const usePipeline = () => {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error('usePipeline must be used within PipelineProvider');
  return ctx;
};

export const PipelineProvider = ({ children }: { children: ReactNode }) => {
  const [state, setState] = useState<PipelineState>({
    audioData: null,
    preprocessingStatus: null,
    mfccData: null,
    inferenceStatus: null,
    intermediatePredictions: null,
    finalPrediction: null,
    pipelineStage: 'idle',
    wsConnected: false,
    wsMessage: null,
  });
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.onopen = () => {
      setState((prev) => ({ ...prev, wsConnected: true }));
    };
    ws.current.onclose = () => {
      setState((prev) => ({ ...prev, wsConnected: false }));
    };
    ws.current.onmessage = (event) => {
      setState((prev) => ({ ...prev, wsMessage: event.data }));
      // TODO: Parse event.data and update relevant state fields
    };
    return () => {
      ws.current?.close();
    };
  }, []);

  return (
    <PipelineContext.Provider value={{ state, setState, ws }}>
      {children}
    </PipelineContext.Provider>
  );
};
