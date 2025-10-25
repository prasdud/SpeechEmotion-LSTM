import { useContext } from 'react';
import { PipelineProvider, PipelineContext } from './context/PipelineContext';
import AudioInput from './components/AudioInput';
import ModelInference from './components/ModelInference';
import PipelineStepper from './components/PipelineStepper';
import OverallProgressBar from './components/OverallProgressBar';
import './App.css';

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
        <div className="app-container">
          <div className="app-header">
            <h1>🎙️ Speech Emotion Recognition</h1>
            <p>Real-time emotion detection from audio using deep learning</p>
          </div>
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
