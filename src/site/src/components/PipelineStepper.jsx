/**
 * PipelineStepper.tsx
 * Shows pipleine stages in a vertical stepper visually 
 * Stages: Audio Upload → Preprocessing → MFCC Extraction → Model Inference → Final Prediction
 * Highlights current active stage
 */
import React, { useContext } from 'react';
import { PipelineContext } from '../context/PipelineContext';

const STAGES = [
  { key: 'AUDIO_UPLOAD', label: 'Audio Upload', icon: '📤' },
  { key: 'audio_processing', label: 'Preprocessing', icon: '⚙️' },
  { key: 'MFCC_EXTRACTION', label: 'MFCC Extraction', icon: '🔊' },
  { key: 'LSTM_INFERENCE', label: 'Model Inference', icon: '🧠' },
  { key: 'completed', label: 'Completed', icon: '✅' },
];

function PipelineStepper() {
  try {
    const { state } = useContext(PipelineContext);
    console.log('[frontend] PipelineStepper component rendered');
    const currentStage = state.stage;
    const currentIndex = STAGES.findIndex(s => s.key === currentStage);
    
    return (
      <div className="card" style={{marginBottom: '2rem'}}>
        <h2 style={{textAlign: 'center'}}>Pipeline Progress</h2>
        <div style={{display: 'flex', justifyContent: 'space-between', position: 'relative', padding: '1rem 0'}}>
          {/* Progress line */}
          <div style={{
            position: 'absolute',
            top: '2.5rem',
            left: '10%',
            right: '10%',
            height: '3px',
            background: 'var(--border-color)',
            zIndex: 0
          }}>
            <div style={{
              width: currentIndex >= 0 ? `${(currentIndex / (STAGES.length - 1)) * 100}%` : '0%',
              height: '100%',
              background: 'linear-gradient(90deg, var(--green-primary), var(--green-hover))',
              transition: 'width 0.5s ease'
            }}></div>
          </div>
          
          {STAGES.map((stage, idx) => {
            const isActive = currentStage === stage.key;
            const isCompleted = currentIndex > idx;
            
            return (
              <div key={stage.key} style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                flex: 1,
                position: 'relative',
                zIndex: 1
              }}>
                <div style={{
                  width: '3rem',
                  height: '3rem',
                  borderRadius: '50%',
                  background: isActive 
                    ? 'linear-gradient(135deg, var(--green-primary), var(--green-hover))'
                    : isCompleted 
                    ? 'var(--green-primary)'
                    : 'var(--bg-secondary)',
                  border: `3px solid ${isActive || isCompleted ? 'var(--green-primary)' : 'var(--border-color)'}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.25rem',
                  transition: 'all 0.3s ease',
                  boxShadow: isActive ? 'var(--shadow-md)' : 'var(--shadow-sm)',
                  transform: isActive ? 'scale(1.1)' : 'scale(1)'
                }}>
                  {stage.icon}
                </div>
                <div style={{
                  marginTop: '0.75rem',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? '600' : '500',
                  color: isActive ? 'var(--green-primary)' : isCompleted ? 'var(--text-secondary)' : 'var(--text-muted)',
                  textAlign: 'center',
                  maxWidth: '100px'
                }}>
                  {stage.label}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering PipelineStepper:', err);
    return <div style={{color:'red'}}>Error rendering PipelineStepper</div>;
  }
}

export default PipelineStepper;
