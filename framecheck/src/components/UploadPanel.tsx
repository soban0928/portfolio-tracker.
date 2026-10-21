import { useId, useRef, useState, type DragEvent } from 'react';
import {
  ACCEPT_ATTRIBUTE,
  MAX_FILE_SIZE_BYTES,
  SUPPORTED_FORMAT_LABELS,
  detectFormat,
  formatBytes,
} from '../lib/fileValidation';
import { Icon } from './Icon';

export type UploadState =
  | { phase: 'empty' }
  | { phase: 'reading'; fileName: string; progress: number }
  | { phase: 'ready'; file: File; previewUrl: string; width: number; height: number; isSample: boolean };

export interface UploadErrorMessage {
  title: string;
  detail: string;
}

interface UploadPanelProps {
  state: UploadState;
  error: UploadErrorMessage | null;
  /** Disables changes while an analysis is running. */
  locked: boolean;
  onFileSelected: (file: File) => void;
  onUseSample: () => void;
  onRemove: () => void;
  onAnalyze: () => void;
}

export function UploadPanel({
  state,
  error,
  locked,
  onFileSelected,
  onUseSample,
  onRemove,
  onAnalyze,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const hintId = useId();
  const errorId = useId();

  const openPicker = () => inputRef.current?.click();

  const handleDragOver = (event: DragEvent) => {
    if (locked) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    setDragging(true);
  };

  const handleDragLeave = (event: DragEvent) => {
    // Ignore leave events fired when moving between child elements.
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setDragging(false);
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    if (locked) return;
    const file = event.dataTransfer.files?.[0];
    if (file) onFileSelected(file);
  };

  const describedBy = error ? `${hintId} ${errorId}` : hintId;

  return (
    <section
      className={`upload-panel${dragging ? ' upload-panel--dragging' : ''}`}
      aria-labelledby="upload-heading"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <h2 id="upload-heading" className="visually-hidden">
        Choose an image
      </h2>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        className="visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFileSelected(file);
          // Allow choosing the same file again after removing it.
          event.target.value = '';
        }}
      />

      {state.phase === 'ready' ? (
        <ReadyView state={state} locked={locked} onReplace={openPicker} onRemove={onRemove} onAnalyze={onAnalyze} />
      ) : (
        <div className="dropzone">
          {state.phase === 'reading' ? (
            <ReadingView fileName={state.fileName} progress={state.progress} />
          ) : (
            <>
              <span className="dropzone__icon">
                <Icon name="upload" size={26} />
              </span>
              <p className="dropzone__title">
                {dragging ? 'Drop the image to add it' : 'Drag an image here, or choose one from your device'}
              </p>
              <p id={hintId} className="dropzone__hint">
                {SUPPORTED_FORMAT_LABELS.slice(0, -1).join(', ')}, or {SUPPORTED_FORMAT_LABELS.at(-1)} (where your browser
                supports it) · up to {formatBytes(MAX_FILE_SIZE_BYTES)}
              </p>
              <div className="dropzone__actions">
                <button
                  id="choose-image-button"
                  type="button"
                  className="button button--primary"
                  onClick={openPicker}
                  aria-describedby={describedBy}
                >
                  <Icon name="image" size={18} />
                  Choose an image
                </button>
                <button type="button" className="button button--ghost" onClick={onUseSample}>
                  Use sample image
                </button>
              </div>
              <p className="dropzone__footnote">Tip: the original file gives better results than a screenshot or re-shared copy.</p>
            </>
          )}
        </div>
      )}

      {error && (
        <div id={errorId} className="notice notice--error" role="alert">
          <Icon name="alert" size={20} />
          <div>
            <p className="notice__title">{error.title}</p>
            <p>{error.detail}</p>
          </div>
        </div>
      )}
    </section>
  );
}

function ReadingView({ fileName, progress }: { fileName: string; progress: number }) {
  const percent = Math.round(progress * 100);
  return (
    <div className="reading" aria-live="polite">
      <p className="dropzone__title">Opening “{fileName}”…</p>
      <div
        className="meter"
        role="progressbar"
        aria-label="Opening image on this device"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <div className="meter__fill" style={{ width: `${percent}%` }} />
      </div>
      <p className="dropzone__hint">Reading the file on this device — nothing is uploaded.</p>
    </div>
  );
}

function ReadyView({
  state,
  locked,
  onReplace,
  onRemove,
  onAnalyze,
}: {
  state: Extract<UploadState, { phase: 'ready' }>;
  locked: boolean;
  onReplace: () => void;
  onRemove: () => void;
  onAnalyze: () => void;
}) {
  const { file, previewUrl, width, height, isSample } = state;
  const format = detectFormat(file)?.label ?? 'Image';
  return (
    <div className="ready">
      <figure className="ready__preview">
        <img
          src={previewUrl}
          alt={isSample ? 'Sample image: an illustrated harbor with a pier and a small sailboat' : `Preview of ${file.name}`}
        />
      </figure>
      <div className="ready__details">
        {isSample && <span className="tag">Sample image</span>}
        <p className="ready__name" title={file.name}>
          {file.name}
        </p>
        <dl className="facts">
          <div>
            <dt>Format</dt>
            <dd>{format}</dd>
          </div>
          <div>
            <dt>Dimensions</dt>
            <dd>
              {width} × {height} px
            </dd>
          </div>
          <div>
            <dt>File size</dt>
            <dd>{formatBytes(file.size)}</dd>
          </div>
        </dl>
        <div className="ready__actions">
          <button type="button" className="button button--primary" onClick={onAnalyze} disabled={locked}>
            Check this image
          </button>
          <button type="button" className="button button--secondary" onClick={onReplace} disabled={locked}>
            <Icon name="refresh" size={16} />
            Replace
          </button>
          <button type="button" className="button button--ghost" onClick={onRemove} disabled={locked}>
            <Icon name="close" size={16} />
            Remove
          </button>
        </div>
      </div>
    </div>
  );
}
