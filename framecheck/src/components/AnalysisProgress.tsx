import { forwardRef } from 'react';
import type { AnalysisStepId } from '../types/analysis';
import { Icon } from './Icon';

export type StepStatus = 'pending' | 'active' | 'done';

const STEPS: { id: AnalysisStepId; label: string; mockNote: string }[] = [
  { id: 'reading', label: 'Reading image details', mockNote: 'Read on this device' },
  { id: 'credentials', label: 'Checking credentials', mockNote: 'Simulated' },
  { id: 'model', label: 'Running image analysis', mockNote: 'Simulated' },
];

interface AnalysisProgressProps {
  statuses: Record<AnalysisStepId, StepStatus>;
  isMock: boolean;
  onCancel: () => void;
}

export const AnalysisProgress = forwardRef<HTMLHeadingElement, AnalysisProgressProps>(function AnalysisProgress(
  { statuses, isMock, onCancel },
  headingRef,
) {
  const active = STEPS.find((s) => statuses[s.id] === 'active');
  return (
    <section className="panel progress" aria-labelledby="progress-heading" aria-busy="true">
      <div className="panel__header">
        <h2 id="progress-heading" className="panel__title" ref={headingRef} tabIndex={-1}>
          Checking the image
        </h2>
        <button type="button" className="button button--ghost button--small" onClick={onCancel}>
          Cancel
        </button>
      </div>
      {isMock && (
        <p className="progress__mock-note">
          <Icon name="info" size={16} />
          Prototype mode: no credential check or detection model is running. The result will be illustrative sample data.
        </p>
      )}
      <ol className="steps">
        {STEPS.map((step) => {
          const status = statuses[step.id];
          return (
            <li key={step.id} className={`steps__item steps__item--${status}`}>
              <span className="steps__marker" aria-hidden="true">
                {status === 'done' ? <Icon name="check" size={16} /> : status === 'active' ? <span className="spinner" /> : null}
              </span>
              <span className="steps__label">
                {step.label}
                <span className="visually-hidden">
                  {status === 'done' ? ' — done' : status === 'active' ? ' — in progress' : ' — waiting'}
                </span>
              </span>
              {isMock && <span className="steps__note">{step.mockNote}</span>}
            </li>
          );
        })}
      </ol>
      <p className="visually-hidden" aria-live="polite">
        {active ? `${active.label}…` : ''}
      </p>
    </section>
  );
});
