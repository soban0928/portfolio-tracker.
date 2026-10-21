export const GENERAL_LIMITATIONS = [
  'Detectors produce false positives (flagging ordinary photos) and false negatives (missing AI-generated or edited images).',
  'Results are less reliable for generators the model hasn’t seen, for edited or composited images, and for screenshots.',
  'Compression, resizing, filters and re-sharing can change the result in either direction.',
  'A Content Credential describes what its signer recorded. A missing credential is common and isn’t evidence either way.',
  'Use this as one input alongside context: where the image came from, who shared it, and whether other sources corroborate it.',
];

interface LimitationsNoticeProps {
  /** Caveats specific to this result, shown before the general limits. */
  specific: string[];
  defaultOpen?: boolean;
}

export function LimitationsNotice({ specific, defaultOpen = false }: LimitationsNoticeProps) {
  return (
    <details className="limits" open={defaultOpen}>
      <summary>
        <span className="limits__title">Limits of this result</span>
        <span className="limits__hint">Results can be wrong — read before relying on this</span>
      </summary>
      <div className="limits__body">
        {specific.length > 0 && (
          <>
            <p className="limits__subhead">For this image</p>
            <ul className="plain-list">
              {specific.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
            <p className="limits__subhead">In general</p>
          </>
        )}
        <ul className="plain-list">
          {GENERAL_LIMITATIONS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
    </details>
  );
}
