import { useContext } from 'react';
import { PipelineProvider, PipelineContext } from './context/PipelineContext';
import AudioInput from './components/AudioInput';
import PreprocessingStatus from './components/PreprocessingStatus';
import MFCCExtraction from './components/MFCCExtraction';
import ModelInference from './components/ModelInference';
import IntermediatePredictions from './components/IntermediatePredictions';
import FinalPredictions from './components/FinalPredictions';
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
        return (
          <>
            <PreprocessingStatus />
            <button style={{marginTop:16}} onClick={triggerMFCCExtraction}>
              Next: Extract MFCC Features
            </button>
          </>
        );
      case 'MFCC_EXTRACTION':
        return (
          <>
            <MFCCExtraction />
            <button style={{marginTop:16}} onClick={triggerModelInference}>
              Next: Run Model Inference
            </button>
          </>
        );
      case 'LSTM_INFERENCE':
        return <ModelInference />;
      case 'INTERMEDIATE_PREDICTIONS':
        return <IntermediatePredictions />;
      case 'completed':
        return <FinalPredictions />;
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
