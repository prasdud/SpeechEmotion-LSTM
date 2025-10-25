/**
 * ModelInference.tsx
 * Shows LSTM inference progress per frame
 * Displays status message ex. "processing frame 10 of 250"
 * Optional: progress bar for overall inference completion
 */


import React, { useContext, useEffect, useRef, useState } from 'react';
import { PipelineContext } from '../context/PipelineContext';

// Emotion mapping
const EMOTIONS = {
  0: { name: 'Happy', emoji: '😊', color: '#FFD700' },
  1: { name: 'Sad', emoji: '😢', color: '#4A90E2' },
  2: { name: 'Angry', emoji: '😠', color: '#E74C3C' },
  3: { name: 'Neutral', emoji: '😐', color: '#95A5A6' }
};

function ModelInference() {
  try {
    const { state, triggerMFCCExtraction, triggerModelInference } = useContext(PipelineContext);
    const mfccStarted = useRef(false);
    const inferenceStarted = useRef(false);
    const [showCelebration, setShowCelebration] = useState(false);
    const prevPrediction = useRef(null);



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

    // Trigger celebration when prediction arrives
    useEffect(() => {
      if (state.finalPrediction && state.finalPrediction !== prevPrediction.current) {
        prevPrediction.current = state.finalPrediction;
        setShowCelebration(true);
        setTimeout(() => setShowCelebration(false), 3000);
      }
    }, [state.finalPrediction]);

    console.log('[frontend] ModelInference component rendered');
    
    const getStageEmoji = (stage) => {
      switch(stage) {
        case 'AUDIO_UPLOAD': return '📤';
        case 'audio_processing': return '⚙️';
        case 'MFCC_EXTRACTION': return '🔊';
        case 'LSTM_INFERENCE': return '🧠';
        case 'completed': return '✅';
        default: return '⏳';
      }
    };
    
    const getStageLabel = (stage) => {
      switch(stage) {
        case 'AUDIO_UPLOAD': return 'AUDIO_UPLOAD';
        case 'audio_processing': return 'audio_processing';
        case 'MFCC_EXTRACTION': return 'MFCC_EXTRACTION';
        case 'LSTM_INFERENCE': return 'LSTM_INFERENCE';
        case 'completed': return 'Full pipeline completed';
        default: return stage || 'Idle';
      }
    };
    
    // Show all pipeline progress, status, and final prediction
    return (
      <div className="card">
        <h2>Pipeline Status</h2>
        
        <div style={{
          display: 'grid',
          gap: '1rem',
          marginBottom: '1.5rem'
        }}>
          <div style={{
            background: 'var(--bg-secondary)',
            padding: '1rem 1.25rem',
            borderRadius: '12px',
            border: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem'
          }}>
            <span style={{fontSize: '1.5rem'}}>{getStageEmoji(state.stage)}</span>
            <div style={{flex: 1}}>
              <div style={{fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px'}}>Current Stage</div>
              <div style={{fontSize: '1.1rem', fontWeight: '600', color: 'var(--text-primary)', marginTop: '0.25rem'}}>{getStageLabel(state.stage)}</div>
            </div>
          </div>
          
          <div style={{
            background: 'var(--bg-secondary)',
            padding: '1rem 1.25rem',
            borderRadius: '12px',
            border: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem'
          }}>
            <span style={{fontSize: '1.5rem'}}>💬</span>
            <div style={{flex: 1}}>
              <div style={{fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px'}}>Status</div>
              <div style={{fontSize: '1rem', color: 'var(--text-secondary)', marginTop: '0.25rem'}}>{state.message || 'Waiting for input...'}</div>
            </div>
          </div>
        </div>
        
        {state.finalPrediction && (
          <div style={{
            background: 'linear-gradient(135deg, var(--green-light) 0%, var(--bg-accent) 100%)',
            padding: '1.5rem',
            borderRadius: '16px',
            marginTop: '1.5rem',
            border: '2px solid var(--green-primary)',
            boxShadow: 'var(--shadow-md)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            {/* Celebration confetti */}
            {showCelebration && (
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                pointerEvents: 'none',
                zIndex: 10
              }}>
                {[...Array(20)].map((_, i) => (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      fontSize: '1.5rem',
                      animation: `confetti-pop 2s ease-out forwards`,
                      animationDelay: `${i * 0.05}s`,
                      opacity: 0
                    }}
                  >
                    {['🎉', '✨', '🌟', '💫', '⭐'][i % 5]}
                  </div>
                ))}
              </div>
            )}
            
            <h3 style={{
              fontSize: '1.2rem',
              fontWeight: '700',
              color: 'var(--text-primary)',
              marginBottom: '1rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              🎯 Final Prediction
            </h3>
            
            {/* Big Emoji Display */}
            <div style={{
              textAlign: 'center',
              margin: '1.5rem 0',
              animation: showCelebration ? 'emoji-bounce 0.6s ease-out' : 'none'
            }}>
              <div style={{
                fontSize: '5rem',
                lineHeight: 1,
                marginBottom: '1rem',
                filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.15))'
              }}>
                {EMOTIONS[state.finalPrediction.class]?.emoji || '❓'}
              </div>
              <div style={{
                fontSize: '2rem',
                fontWeight: '700',
                color: EMOTIONS[state.finalPrediction.class]?.color || 'var(--text-primary)',
                textShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                {EMOTIONS[state.finalPrediction.class]?.name || `Class ${state.finalPrediction.class}`}
              </div>
            </div>
            
            <div style={{
              background: 'var(--bg-secondary)',
              padding: '1rem',
              borderRadius: '12px'
            }}>
              <div style={{fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: '600', marginBottom: '0.75rem'}}>Confidence Distribution</div>
              {Array.isArray(state.finalPrediction.confidence) && state.finalPrediction.confidence.map((conf, i) => (
                <div key={i} style={{marginBottom: '0.5rem'}}>
                  <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem'}}>
                    <span style={{color: 'var(--text-secondary)', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                      <span style={{fontSize: '1.25rem'}}>{EMOTIONS[i]?.emoji}</span>
                      {EMOTIONS[i]?.name || `Class ${i}`}
                    </span>
                    <span style={{color: 'var(--text-primary)', fontWeight: '700'}}>{(conf * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{
                    height: '8px',
                    background: 'var(--border-color)',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${conf * 100}%`,
                      height: '100%',
                      background: i === state.finalPrediction.class 
                        ? `linear-gradient(90deg, ${EMOTIONS[i]?.color || 'var(--green-primary)'}, ${EMOTIONS[i]?.color || 'var(--green-hover)'})80`
                        : 'var(--text-muted)',
                      transition: 'width 0.5s ease'
                    }}></div>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Animation styles */}
            <style>{`
              @keyframes confetti-pop {
                0% {
                  transform: translate(-50%, -50%) translate(0, 0) rotate(0deg) scale(0);
                  opacity: 1;
                }
                50% {
                  opacity: 1;
                }
                100% {
                  transform: translate(-50%, -50%) 
                             translate(${Math.random() * 400 - 200}px, ${Math.random() * 400 - 200}px) 
                             rotate(${Math.random() * 720}deg) 
                             scale(${Math.random() * 0.5 + 0.5});
                  opacity: 0;
                }
              }
              
              @keyframes emoji-bounce {
                0%, 100% { transform: scale(1); }
                25% { transform: scale(1.2); }
                50% { transform: scale(0.95); }
                75% { transform: scale(1.05); }
              }
            `}</style>
          </div>
        )}
        
        {state.error && (
          <div style={{
            background: '#ffebee',
            color: '#c62828',
            padding: '1rem 1.25rem',
            borderRadius: '12px',
            marginTop: '1rem',
            border: '2px solid #ef5350',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem'
          }}>
            <span style={{fontSize: '1.5rem'}}>⚠️</span>
            <div>{state.error}</div>
          </div>
        )}
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering ModelInference:', err);
    return <div style={{color:'red'}}>Error rendering ModelInference</div>;
  }
}

export default ModelInference;
