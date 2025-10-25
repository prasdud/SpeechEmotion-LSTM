/**
 * OverallProgressBar.tsx
 * Shows overall progress bar for the entire pipeline 0 - 100% as the backend sends the updates
 */
import React, { useContext } from 'react';
import { PipelineContext } from '../context/PipelineContext';

function OverallProgressBar() {
  try {
    const { state } = useContext(PipelineContext);
    console.log('[frontend] OverallProgressBar component rendered');
    // Always show lastGoodProgress if pipeline is completed, or if error and lastGoodProgress exists
    let progress = state.progress || 0;
    if ((state.pipelineCompleted || (state.error && state.lastGoodProgress)) && state.lastGoodProgress) {
      progress = state.lastGoodProgress;
    }
    return (
      <div style={{marginBottom:16}}>
        <h2>Overall Progress</h2>
        <div style={{background:'#eee',height:16,borderRadius:8,marginTop:4}}>
          <div style={{width:`${progress}%`,height:16,background:'#2196f3',borderRadius:8,transition:'width 0.3s'}}></div>
        </div>
        <div style={{fontSize:12,marginTop:2}}>{progress}%</div>
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering OverallProgressBar:', err);
    return <div style={{color:'red'}}>Error rendering OverallProgressBar</div>;
  }
}

export default OverallProgressBar;
