import { Icon } from './Icon';

interface HeaderProps {
  /** False once real services are connected and images leave the browser. */
  localOnly: boolean;
}

export function Header({ localOnly }: HeaderProps) {
  return (
    <header className="site-header">
      <div className="container site-header__inner">
        <a className="wordmark" href="#top" aria-label="FrameCheck home">
          <svg className="wordmark__mark" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <path
              d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <circle cx="12" cy="12" r="2.5" fill="currentColor" />
          </svg>
          <span>FrameCheck</span>
        </a>
        <nav aria-label="Primary" className="site-header__nav">
          <a href="#how-it-works">How it works</a>
        </nav>
        <p className="privacy-note">
          <Icon name="lock" size={16} />
          {localOnly ? (
            <span>
              <strong>Runs in your browser.</strong> Images aren’t uploaded or stored.
            </span>
          ) : (
            <span>Images are sent to the analysis service to be checked.</span>
          )}
        </p>
      </div>
    </header>
  );
}
