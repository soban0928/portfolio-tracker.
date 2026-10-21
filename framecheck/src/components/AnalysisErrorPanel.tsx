import { forwardRef } from 'react';
import { Icon } from './Icon';

interface AnalysisErrorPanelProps {
  message: string;
  onRetry: () => void;
  onChooseAnother: () => void;
}

export const AnalysisErrorPanel = forwardRef<HTMLHeadingElement, AnalysisErrorPanelProps>(function AnalysisErrorPanel(
  { message, onRetry, onChooseAnother },
  headingRef,
) {
  return (
    <section className="panel analysis-error" aria-labelledby="analysis-error-heading" role="alert">
      <span className="analysis-error__icon">
        <Icon name="alert" size={22} />
      </span>
      <div>
        <h2 id="analysis-error-heading" className="panel__title" ref={headingRef} tabIndex={-1}>
          The check couldn’t be completed
        </h2>
        <p>{message}</p>
        <p className="muted">No result was produced, so nothing should be read into this image from this attempt.</p>
        <div className="analysis-error__actions">
          <button type="button" className="button button--primary" onClick={onRetry}>
            Try again
          </button>
          <button type="button" className="button button--secondary" onClick={onChooseAnother}>
            Choose a different image
          </button>
        </div>
      </div>
    </section>
  );
});
