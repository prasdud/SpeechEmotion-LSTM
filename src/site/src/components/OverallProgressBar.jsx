/**
 * OverallProgressBar.tsx
 * Shows overall progress bar for the entire pipeline 0 - 100% as the backend sends the updates
 */
import React from 'react';

function OverallProgressBar() {
  try {
    console.log('[frontend] OverallProgressBar component rendered');
    // TODO: Show overall progress bar for pipeline
    return (
      <div>
        <h2>Overall Progress</h2>
        {/* Progress bar for whole pipeline */}
      </div>
    );
  } catch (err) {
    console.error('[frontend] Error rendering OverallProgressBar:', err);
    return <div style={{color:'red'}}>Error rendering OverallProgressBar</div>;
  }
}

export default OverallProgressBar;
