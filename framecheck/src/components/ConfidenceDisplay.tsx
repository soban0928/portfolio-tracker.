import { scoreToPercent } from '../lib/summarize';

interface ConfidenceDisplayProps {
  modelScore: number | null;
  modelVersion: string;
}

export function ConfidenceDisplay({ modelScore, modelVersion }: ConfidenceDisplayProps) {
  if (modelScore === null) {
    return (
      <div className="confidence">
        <p className="confidence__label">Model estimate</p>
        <p className="confidence__value confidence__value--muted">No estimate for this image</p>
        <p className="confidence__note">
          The image analysis model couldn’t produce a usable estimate, often because the image is very small or heavily
          compressed.
        </p>
        <p className="confidence__version">Model version: {modelVersion}</p>
      </div>
    );
  }

  const percent = scoreToPercent(modelScore);
  return (
    <div className="confidence">
      <p className="confidence__label" id="confidence-label">
        Model estimate
      </p>
      <p className="confidence__value">
        <span className="confidence__number">{percent}%</span> AI-like patterns
      </p>
      <div
        className="scale"
        role="meter"
        aria-labelledby="confidence-label"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={`${percent}% AI-like patterns`}
      >
        <div className="scale__track">
          <div className="scale__fill" style={{ width: `${percent}%` }} />
          <div className="scale__marker" style={{ left: `${percent}%` }} />
        </div>
        <div className="scale__ends" aria-hidden="true">
          <span>Fewer AI-like patterns</span>
          <span>More AI-like patterns</span>
        </div>
      </div>
      <p className="confidence__note">
        This is how strongly the image matches patterns the model has learned from AI-generated images. It is the
        model’s estimate, <strong>not the probability that the image is fake</strong>, and it can be wrong.
      </p>
      <p className="confidence__version">Model version: {modelVersion}</p>
    </div>
  );
}
