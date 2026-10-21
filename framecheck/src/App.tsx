import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnalysisErrorPanel } from './components/AnalysisErrorPanel';
import { AnalysisProgress, type StepStatus } from './components/AnalysisProgress';
import { Header } from './components/Header';
import { HowItWorks } from './components/HowItWorks';
import { PrototypeControls } from './components/PrototypeControls';
import { ResultSummary } from './components/ResultSummary';
import { UploadPanel, type UploadErrorMessage, type UploadState } from './components/UploadPanel';
import { describeValidationError, validateImageFile } from './lib/fileValidation';
import { readImageFile } from './lib/readImage';
import { createSampleImageFile } from './lib/sampleImage';
import { createAnalysisClient } from './services';
import type { MockAnalysisClientOptions, MockScenarioChoice } from './services/mock/mockAnalysisService';
import { AnalysisError, type AnalysisClient, type AnalysisResult, type AnalysisStepId } from './types/analysis';

type AnalysisState =
  | { phase: 'idle' }
  | { phase: 'running'; steps: Record<AnalysisStepId, StepStatus> }
  | { phase: 'success'; result: AnalysisResult }
  | { phase: 'error'; message: string };

const INITIAL_STEPS: Record<AnalysisStepId, StepStatus> = { reading: 'pending', credentials: 'pending', model: 'pending' };

interface AppProps {
  /** Overrides for tests; production uses `createAnalysisClient`. */
  client?: AnalysisClient;
  mockOptions?: Omit<MockAnalysisClientOptions, 'getScenario'>;
}

export default function App({ client: injectedClient, mockOptions }: AppProps) {
  const [upload, setUpload] = useState<UploadState>({ phase: 'empty' });
  const [uploadError, setUploadError] = useState<UploadErrorMessage | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisState>({ phase: 'idle' });
  const [scenario, setScenario] = useState<MockScenarioChoice>('auto');

  const scenarioRef = useRef(scenario);
  scenarioRef.current = scenario;
  const readTokenRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const resultHeadingRef = useRef<HTMLHeadingElement>(null);
  const errorHeadingRef = useRef<HTMLHeadingElement>(null);
  const progressHeadingRef = useRef<HTMLHeadingElement>(null);

  const client = useMemo(
    () => injectedClient ?? createAnalysisClient({ ...mockOptions, getScenario: () => scenarioRef.current }),
    [injectedClient, mockOptions],
  );

  const setPreviewUrl = (url: string | null) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = url;
  };

  useEffect(
    () => () => {
      abortRef.current?.abort();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    },
    [],
  );

  // Move focus to the outcome so keyboard and screen reader users land on it.
  useEffect(() => {
    if (analysis.phase === 'running') {
      const heading = progressHeadingRef.current;
      heading?.focus({ preventScroll: true });
      heading?.closest('section')?.scrollIntoView?.({ behavior: 'smooth', block: 'center' });
    }
    if (analysis.phase === 'success') resultHeadingRef.current?.focus();
    if (analysis.phase === 'error') errorHeadingRef.current?.focus();
  }, [analysis.phase]);

  const handleFile = useCallback(async (file: File, isSample = false) => {
    const validationError = validateImageFile(file);
    if (validationError) {
      setUploadError(describeValidationError(validationError));
      return;
    }

    const token = ++readTokenRef.current;
    setUploadError(null);
    setAnalysis({ phase: 'idle' });
    setUpload({ phase: 'reading', fileName: file.name, progress: 0 });

    try {
      const loaded = await readImageFile(file, (progress) => {
        if (token === readTokenRef.current) setUpload({ phase: 'reading', fileName: file.name, progress });
      });
      if (token !== readTokenRef.current) {
        URL.revokeObjectURL(loaded.previewUrl);
        return;
      }
      setPreviewUrl(loaded.previewUrl);
      setUpload({ phase: 'ready', file, isSample, ...loaded });
    } catch {
      if (token !== readTokenRef.current) return;
      setPreviewUrl(null);
      setUpload({ phase: 'empty' });
      setUploadError({
        title: 'This image couldn’t be opened',
        detail: `“${file.name}” may be damaged, or your browser may not support this format (AVIF support varies). Try a JPEG, PNG or WebP version.`,
      });
    }
  }, []);

  const handleUseSample = useCallback(async () => {
    try {
      await handleFile(await createSampleImageFile(), true);
    } catch {
      setUploadError({
        title: 'The sample image couldn’t be created',
        detail: 'Your browser blocked drawing the sample. You can still choose an image of your own.',
      });
    }
  }, [handleFile]);

  const handleRemove = useCallback(() => {
    readTokenRef.current++;
    abortRef.current?.abort();
    setPreviewUrl(null);
    setUpload({ phase: 'empty' });
    setUploadError(null);
    setAnalysis({ phase: 'idle' });
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (upload.phase !== 'ready') return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setAnalysis({ phase: 'running', steps: INITIAL_STEPS });

    try {
      const result = await client.analyzeImage(upload.file, {
        signal: controller.signal,
        onStep: ({ step, status }) => {
          if (controller.signal.aborted) return;
          setAnalysis((prev) =>
            prev.phase === 'running' ? { phase: 'running', steps: { ...prev.steps, [step]: status } } : prev,
          );
        },
      });
      if (!controller.signal.aborted) setAnalysis({ phase: 'success', result });
    } catch (err) {
      if (controller.signal.aborted) return;
      const message =
        err instanceof AnalysisError ? err.message : 'Something went wrong while checking this image. Please try again.';
      setAnalysis({ phase: 'error', message });
    }
  }, [client, upload]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setAnalysis({ phase: 'idle' });
  }, []);

  const handleCheckAnother = useCallback(() => {
    handleRemove();
    // Focusing the picker button also scrolls the upload panel into view.
    requestAnimationFrame(() => document.getElementById('choose-image-button')?.focus());
  }, [handleRemove]);

  const running = analysis.phase === 'running';
  const showUpload = analysis.phase !== 'success';

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Header localOnly={client.isMock} />
      <main id="main" tabIndex={-1}>
        <div className="container hero">
          <p className="eyebrow">Image provenance &amp; analysis{client.isMock ? ' · Prototype' : ''}</p>
          <h1>Check the signals behind an image.</h1>
          <p className="hero__lede">
            We look for available provenance information and model-based patterns. No single check can prove who made an
            image.
          </p>
        </div>

        <div className="container workspace">
          {showUpload && (
            <>
              <UploadPanel
                state={upload}
                error={uploadError}
                locked={running}
                onFileSelected={(file) => void handleFile(file)}
                onUseSample={() => void handleUseSample()}
                onRemove={handleRemove}
                onAnalyze={() => void handleAnalyze()}
              />
              {client.isMock && <PrototypeControls scenario={scenario} onChange={setScenario} disabled={running} />}
            </>
          )}

          {analysis.phase === 'running' && (
            <AnalysisProgress ref={progressHeadingRef} statuses={analysis.steps} isMock={client.isMock} onCancel={handleCancel} />
          )}

          {analysis.phase === 'error' && (
            <AnalysisErrorPanel
              ref={errorHeadingRef}
              message={analysis.message}
              onRetry={() => void handleAnalyze()}
              onChooseAnother={handleCheckAnother}
            />
          )}

          {analysis.phase === 'success' && upload.phase === 'ready' && (
            <ResultSummary
              ref={resultHeadingRef}
              result={analysis.result}
              previewUrl={upload.previewUrl}
              previewAlt={upload.isSample ? 'The sample harbor image that was checked' : `The image that was checked: ${upload.file.name}`}
              onCheckAnother={handleCheckAnother}
            />
          )}
        </div>

        <HowItWorks />
      </main>
      <footer className="site-footer">
        <div className="container">
          <p>
            FrameCheck prototype. {client.isMock ? 'All results are illustrative sample data. ' : ''}Results are
            estimates and can be wrong.
          </p>
        </div>
      </footer>
    </>
  );
}
