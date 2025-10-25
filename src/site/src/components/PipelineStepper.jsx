/**
 * PipelineStepper.tsx
 * Shows pipleine stages in a vertical stepper visually 
 * Stages: Audio Upload → Preprocessing → MFCC Extraction → Model Inference → Final Prediction
 * Highlights current active stage
 */
import React, { useContext } from 'react';
import { PipelineContext } from '../context/PipelineContext';

const STAGES = [
  { key: 'AUDIO_UPLOAD', label: 'Audio Upload' },
  { key: 'audio_processing', label: 'Preprocessing' },
  { key: 'MFCC_EXTRACTION', label: 'MFCC Extraction' },
  { key: 'LSTM_INFERENCE', label: 'Model Inference' },
  { key: 'completed', label: 'Completed' },
];

function PipelineStepper() {
  try {
    const { state } = useContext(PipelineContext);
    console.log('[frontend] PipelineStepper component rendered');
    const currentStage = state.stage;
    return (
      <div style={{marginBottom:16}}>
        <h2>Pipeline Stepper</h2>
        <ol style={{listStyle:'none',padding:0,margin:0}}>
          {STAGES.map((stage, idx) => (
            <li key={stage.key} style={{
              display:'flex',alignItems:'center',marginBottom:8,
              fontWeight: currentStage === stage.key ? 'bold' : 'normal',
              color: currentStage === stage.key ? '#2196f3' : '#333',
            }}>
              <span style={{width:18,height:18,borderRadius:'50%',background:currentStage === stage.key ? '#2196f3' : '#ccc',display:'inline-block',marginRight:8}}></span>
              {stage.label}
            </li>
          ))}
        </ol>
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering PipelineStepper:', err);
    return <div style={{color:'red'}}>Error rendering PipelineStepper</div>;
  }
}

export default PipelineStepper;
