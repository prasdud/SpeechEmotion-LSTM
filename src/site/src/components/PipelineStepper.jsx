/**
 * PipelineStepper.tsx
 * Shows pipleine stages in a vertical stepper visually 
 * Stages: Audio Upload → Preprocessing → MFCC Extraction → Model Inference → Final Prediction
 * Highlights current active stage
 */
import React from 'react';

function PipelineStepper() {
  try {
    console.log('[frontend] PipelineStepper component rendered');
    // TODO: Show vertical stepper for pipeline stages
    return (
      <div>
        <h2>Pipeline Stepper</h2>
        {/* Stepper: Upload → Preprocessing → MFCC → Model → Prediction */}
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering PipelineStepper:', err);
    return <div style={{color:'red'}}>Error rendering PipelineStepper</div>;
  }
}

export default PipelineStepper;
