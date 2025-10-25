/**
 * PreprocessingStatus.jsx
 * Displays status messages from the backend during audio processing
 * Ex. Audio loaded, Normalized, Framed, etc.
 */
import React from 'react';

function PreprocessingStatus() {
  try {
    console.log('[frontend] PreprocessingStatus component rendered');
    // TODO: Show real-time preprocessing status, duration, frames
    return (
      <div>
        <h2>Preprocessing Status</h2>
        {/* Status, duration, frames */}
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering PreprocessingStatus:', err);
    return <div style={{color:'red'}}>Error rendering PreprocessingStatus</div>;
  }
}

export default PreprocessingStatus;
