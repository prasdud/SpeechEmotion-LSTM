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
    
    // Use current progress, which should stay at 100 once pipeline completes
    const progress = state.progress || 0;
    
    return (
      <div className="card" style={{marginBottom: '2rem'}}>
        <h2 style={{textAlign: 'center'}}>Overall Progress</h2>
        <div style={{
          background: 'var(--bg-secondary)',
          height: '2rem',
          borderRadius: '12px',
          overflow: 'hidden',
          border: '2px solid var(--border-color)',
          position: 'relative'
        }}>
          {progress > 0 && (
            <div style={{
              width: `${progress}%`,
              height: '100%',
              background: progress === 100 
                ? 'linear-gradient(90deg, var(--green-primary), var(--green-hover))'
                : 'linear-gradient(90deg, var(--green-light), var(--green-primary))',
              transition: 'width 0.5s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              paddingRight: '0.75rem',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
                animation: progress < 100 ? 'shimmer 2s infinite' : 'none'
              }}></div>
            </div>
          )}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            fontWeight: '700',
            fontSize: '1rem',
            color: progress > 50 ? 'white' : 'var(--text-primary)',
            textShadow: progress > 50 ? '0 1px 2px rgba(0,0,0,0.2)' : 'none',
            pointerEvents: 'none'
          }}>
            {Math.round(progress)}%
          </div>
        </div>
        <style>{`
          @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
        `}</style>
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering OverallProgressBar:', err);
    return <div style={{color:'red'}}>Error rendering OverallProgressBar</div>;
  }
}

export default OverallProgressBar;
