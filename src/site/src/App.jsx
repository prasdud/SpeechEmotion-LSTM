import { useContext } from 'react';
import { PipelineProvider, PipelineContext } from './context/PipelineContext';
import AudioInput from './components/AudioInput';
import ModelInference from './components/ModelInference';
import PipelineStepper from './components/PipelineStepper';
import OverallProgressBar from './components/OverallProgressBar';

function PipelineStages() {
  try {
    const { state, triggerMFCCExtraction, triggerModelInference } = useContext(PipelineContext);
    console.log('[frontend] PipelineStages rendered, state:', state);
    switch (state.stage) {
      case null:
      case 'idle':
        return <AudioInput />;
      case 'AUDIO_UPLOAD':
      case 'audio_processing':
      case 'MFCC_EXTRACTION':
      case 'LSTM_INFERENCE':
      case 'completed':
        return <ModelInference />;
      default:
        return <AudioInput />;
    }
  } catch (err) {
    console.error('[frontend] Error rendering PipelineStages:', err);
    return <div style={{color:'red'}}>Error rendering pipeline stages</div>;
  }
}

function App() {
  try {
    console.log('[frontend] App component rendered');
    return (
      <PipelineProvider>
        <div style={{ maxWidth: 800, margin: '0 auto', padding: 24 }}>
          <h1>Speech Emotion Recognition Pipeline</h1>
          <PipelineStepper />
          <OverallProgressBar />
          <PipelineStages />
        </div>
      </PipelineProvider>
    );
  } catch (err) {
    console.error('[frontend] Error rendering App:', err);
    return <div style={{color:'red'}}>Error rendering App</div>;
  }
}

export default App;
