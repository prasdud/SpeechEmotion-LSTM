
import { PipelineProvider } from './context/PipelineContext';
import AudioInput from './components/AudioInput';
import PreprocessingStatus from './components/PreprocessingStatus';
import MFCCExtraction from './components/MFCCExtraction';
import ModelInference from './components/ModelInference';
import IntermediatePredictions from './components/IntermediatePredictions';
import FinalPredictions from './components/FinalPredictions';
import PipelineStepper from './components/PipelineStepper';
import OverallProgressBar from './components/OverallProgressBar';

function App() {
  return (
    <PipelineProvider>
      <div style={{ maxWidth: 800, margin: '0 auto', padding: 24 }}>
        <h1>Speech Emotion Recognition Pipeline</h1>
        <PipelineStepper />
        <OverallProgressBar />
        <AudioInput />
        <PreprocessingStatus />
        <MFCCExtraction />
        <ModelInference />
        <IntermediatePredictions />
        <FinalPredictions />
      </div>
    </PipelineProvider>
  );
}

export default App;
