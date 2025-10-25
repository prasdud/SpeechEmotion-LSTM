/**
 * ModelInference.tsx
 * Shows LSTM inference progress per frame
 * Displays status message ex. "processing frame 10 of 250"
 * Optional: progress bar for overall inference completion
 */


import React, { useContext, useEffect, useRef } from 'react';
import { PipelineContext } from '../context/PipelineContext';

function ModelInference() {
  try {
    const { state, triggerMFCCExtraction, triggerModelInference } = useContext(PipelineContext);
    const mfccStarted = useRef(false);
    const inferenceStarted = useRef(false);



    // Auto-advance pipeline: trigger next step as soon as backend signals previous step is done
    useEffect(() => {
      if (!mfccStarted.current && (state.stage === 'AUDIO_UPLOAD' || state.stage === 'audio_processing')) {
        mfccStarted.current = true;
        console.log('[frontend] Auto-advance: triggering MFCC extraction');
        triggerMFCCExtraction();
      }
    }, [state.stage, triggerMFCCExtraction]);

    useEffect(() => {
      if (!inferenceStarted.current && state.stage === 'MFCC_EXTRACTION') {
        inferenceStarted.current = true;
        console.log('[frontend] Auto-advance: triggering model inference');
        triggerModelInference();
      }
    }, [state.stage, triggerModelInference]);

    console.log('[frontend] ModelInference component rendered');
    // Show all pipeline progress, status, and final prediction
    return (
      <div>
        <h2>Model Inference & Prediction</h2>
        <div style={{marginBottom:8}}>
          <b>Status:</b> {state.message || 'Waiting...'}
        </div>
        <div style={{marginBottom:8}}>
          <b>Stage:</b> {state.stage}
        </div>
        <div style={{marginBottom:8}}>
          <b>Progress:</b> {state.progress ? `${state.progress}%` : '0%'}
          <div style={{background:'#eee',height:10,borderRadius:5,marginTop:4}}>
            <div style={{width:`${state.progress||0}%`,height:10,background:'#4caf50',borderRadius:5}}></div>
          </div>
        </div>
        {state.finalPrediction && (
          <div style={{marginTop:16}}>
            <b>Final Prediction:</b>
            <div>Class: {state.finalPrediction.class}</div>
            <div>Confidence: {Array.isArray(state.finalPrediction.confidence) ? state.finalPrediction.confidence.map((c,i) => (
              <span key={i}>{c.toFixed(2)}{i < state.finalPrediction.confidence.length-1 ? ', ' : ''}</span>
            )) : state.finalPrediction.confidence}
            </div>
          </div>
        )}
        {state.error && <div style={{color:'red',marginTop:8}}>{state.error}</div>}
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering ModelInference:', err);
    return <div style={{color:'red'}}>Error rendering ModelInference</div>;
  }
}

export default ModelInference;
