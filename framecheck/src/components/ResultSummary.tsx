import { forwardRef } from 'react';
import { formatBytes } from '../lib/fileValidation';
import { summarizeResult } from '../lib/summarize';
import type { AnalysisResult } from '../types/analysis';
import { ConfidenceDisplay } from './ConfidenceDisplay';
import { Icon } from './Icon';
import { LimitationsNotice } from './LimitationsNotice';
import { ObservationsList } from './ObservationsList';
import { ProvenanceStatus } from './ProvenanceStatus';

interface ResultSummaryProps {
  result: AnalysisResult;
  previewUrl: string;
  previewAlt: string;
  onCheckAnother: () => void;
}

export const ResultSummary = forwardRef<HTMLHeadingElement, ResultSummaryProps>(function ResultSummary(
  { result, previewUrl, previewAlt, onCheckAnother },
  headingRef,
) {
  const summary = summarizeResult(result);
  const { imageInfo } = result;

  return (
    <section className="panel result" aria-labelledby="result-heading">
      {result.isIllustrative && (
        <div className="notice notice--sample">
          <Icon name="info" size={20} />
          <div>
            <p className="notice__title">Illustrative sample result</p>
            <p>
              This prototype isn’t connected to a detection model or credential checker. Everything below is mock data
              chosen to demonstrate the layout — it says nothing about your image.
            </p>
          </div>
        </div>
      )}

      <div className="result__top">
        <figure className="result__thumb">
          <img src={previewUrl} alt={previewAlt} />
          <figcaption>
            <span className="result__filename" title={imageInfo.fileName}>
              {imageInfo.fileName}
            </span>
            <span className="muted">
              {imageInfo.width} × {imageInfo.height} px · {formatBytes(imageInfo.sizeBytes)}
            </span>
          </figcaption>
        </figure>

        <div className="result__summary">
          <p className="eyebrow">{result.isIllustrative ? 'Sample result' : 'Result'}</p>
          <h2 id="result-heading" className="result__headline" ref={headingRef} tabIndex={-1}>
            {summary.headline}
          </h2>
          <p className="result__body">{summary.body}</p>
          <ConfidenceDisplay modelScore={result.modelScore} modelVersion={result.modelVersion} />
        </div>
      </div>

      <div className="result__grid">
        <ProvenanceStatus status={result.provenanceStatus} details={result.provenanceDetails} />
        <ObservationsList observations={result.observations} />
      </div>

      <LimitationsNotice specific={result.limitations} />

      <div className="result__actions">
        <button type="button" className="button button--primary" onClick={onCheckAnother}>
          <Icon name="refresh" size={18} />
          Check another image
        </button>
      </div>
    </section>
  );
});
