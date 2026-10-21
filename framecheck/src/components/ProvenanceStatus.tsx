import type { ProvenanceDetails, ProvenanceStatus as Status } from '../types/analysis';
import { Icon, type IconName } from './Icon';

const STATUS_COPY: Record<Status, { title: string; detail: string; icon: IconName }> = {
  verified: {
    title: 'Verified credential found',
    detail:
      'The file carries a Content Credential whose signature checks out. It describes who signed it and the history they recorded — not whether the scene itself is true.',
    icon: 'shieldCheck',
  },
  unverified: {
    title: 'Credential found but could not be verified',
    detail:
      'The file carries a Content Credential, but its signature or contents didn’t validate. The image may have changed after signing, or the signer may not be recognized.',
    icon: 'shieldOff',
  },
  not_found: {
    title: 'No credential found',
    detail:
      'This file doesn’t carry a Content Credential. That’s common — most cameras, apps and platforms don’t add one, and sharing often strips it. It doesn’t mean the image is AI-generated.',
    icon: 'shield',
  },
};

function formatDate(iso?: string): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? null
    : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

interface ProvenanceStatusProps {
  status: Status;
  details: ProvenanceDetails | null;
}

export function ProvenanceStatus({ status, details }: ProvenanceStatusProps) {
  const copy = STATUS_COPY[status];
  const signedAt = formatDate(details?.signedAt);

  return (
    <section className={`provenance provenance--${status}`} aria-labelledby="provenance-heading">
      <h3 id="provenance-heading" className="section-title">
        Content credentials
      </h3>
      <div className="provenance__status">
        <span className="provenance__icon">
          <Icon name={copy.icon} size={22} />
        </span>
        <div>
          <p className="provenance__title">{copy.title}</p>
          <p className="provenance__detail">{copy.detail}</p>
        </div>
      </div>

      {details && (
        <dl className="facts facts--stacked">
          {details.signer && (
            <div>
              <dt>Signed by</dt>
              <dd>{details.signer}</dd>
            </div>
          )}
          {signedAt && (
            <div>
              <dt>Signed</dt>
              <dd>{signedAt}</dd>
            </div>
          )}
          {details.claimGenerator && (
            <div>
              <dt>Recorded with</dt>
              <dd>{details.claimGenerator}</dd>
            </div>
          )}
          {details.declaresGenerativeAI !== undefined && (
            <div>
              <dt>Generative AI</dt>
              <dd>{details.declaresGenerativeAI ? 'Declared in the credential' : 'Not declared in the credential'}</dd>
            </div>
          )}
          {details.actions && details.actions.length > 0 && (
            <div>
              <dt>Recorded history</dt>
              <dd>
                <ol className="history">
                  {details.actions.map((action, i) => (
                    <li key={i}>
                      {action.label}
                      {action.softwareAgent && <span className="history__agent"> · {action.softwareAgent}</span>}
                    </li>
                  ))}
                </ol>
              </dd>
            </div>
          )}
          {details.validationNotes && details.validationNotes.length > 0 && (
            <div>
              <dt>Validation</dt>
              <dd>
                <ul className="plain-list">
                  {details.validationNotes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </dd>
            </div>
          )}
        </dl>
      )}
    </section>
  );
}
