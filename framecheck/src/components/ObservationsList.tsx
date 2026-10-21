import type { Observation, ObservationSource } from '../types/analysis';
import { Icon, type IconName } from './Icon';

const SOURCE_META: Record<ObservationSource, { label: string; icon: IconName }> = {
  file: { label: 'File', icon: 'file' },
  model: { label: 'Model', icon: 'chart' },
  provenance: { label: 'Credential', icon: 'shield' },
};

export function ObservationsList({ observations }: { observations: Observation[] }) {
  return (
    <section aria-labelledby="observations-heading">
      <h3 id="observations-heading" className="section-title">
        What we noticed
      </h3>
      <p className="section-intro">Supporting context only — none of these settles the question on its own.</p>
      {observations.length === 0 ? (
        <p className="muted">No additional observations for this image.</p>
      ) : (
        <ul className="observations">
          {observations.map((obs) => {
            const meta = SOURCE_META[obs.source];
            return (
              <li key={obs.id} className="observation">
                <span className="observation__source">
                  <Icon name={meta.icon} size={14} />
                  {meta.label}
                </span>
                <p className="observation__title">{obs.title}</p>
                <p className="observation__detail">{obs.detail}</p>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
