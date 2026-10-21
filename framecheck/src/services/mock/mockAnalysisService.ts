/**
 * Mock implementation of `AnalysisClient` for the prototype.
 *
 * It reads basic file details locally, waits through each visible step, and
 * returns an illustrative fixture. No detection model or credential check runs
 * and the image never leaves the browser.
 */
import {
  AnalysisError,
  type AnalysisClient,
  type AnalysisResult,
  type AnalysisStepId,
  type AnalyzeOptions,
} from '../../types/analysis';
import { SAMPLE_FILE_NAME } from '../../lib/sampleImage';
import { readImageDimensions, type ImageDimensions } from '../imageDimensions';
import { MOCK_FIXTURES, SAMPLE_IMAGE_SCENARIO, type MockScenarioId } from './mockAnalysisFixtures';

/** `auto` picks a fixture from the file; `analysis_error` simulates a failure. */
export type MockScenarioChoice = 'auto' | MockScenarioId | 'analysis_error';

export interface MockAnalysisClientOptions {
  /** Read on each run so the UI can switch scenarios without recreating the client. */
  getScenario?: () => MockScenarioChoice;
  /** Simulated duration of each step. */
  stepDelayMs?: number;
  /** Injectable for tests (jsdom can't decode images). */
  readDimensions?: (file: File) => Promise<ImageDimensions>;
}

const AUTO_ROTATION: MockScenarioId[] = [
  'ai_like_no_credential',
  'mixed_unverified_credential',
  'few_ai_like_verified_credential',
  'ai_like_verified_genai_credential',
];

/** Very small images get the inconclusive fixture in `auto` mode. */
const MIN_USEFUL_EDGE_PX = 256;

/** Stable, cheap string hash so the same file always gets the same fixture. */
function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) hash = (hash * 31 + value.charCodeAt(i)) | 0;
  return Math.abs(hash);
}

export function pickAutoScenario(file: Pick<File, 'name' | 'size'>, dims: ImageDimensions): MockScenarioId {
  if (file.name === SAMPLE_FILE_NAME) return SAMPLE_IMAGE_SCENARIO;
  if (Math.min(dims.width, dims.height) < MIN_USEFUL_EDGE_PX) return 'inconclusive_no_credential';
  return AUTO_ROTATION[hashString(`${file.name}:${file.size}`) % AUTO_ROTATION.length];
}

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}

export function createMockAnalysisClient(options: MockAnalysisClientOptions = {}): AnalysisClient {
  const { getScenario = () => 'auto', stepDelayMs = 900, readDimensions = readImageDimensions } = options;

  return {
    isMock: true,
    async analyzeImage(file: File, { signal, onStep }: AnalyzeOptions = {}): Promise<AnalysisResult> {
      const runStep = async (step: AnalysisStepId, work?: () => Promise<void>) => {
        onStep?.({ step, status: 'active' });
        await Promise.all([wait(stepDelayMs, signal), work?.()]);
        onStep?.({ step, status: 'done' });
      };

      let dims: ImageDimensions = { width: 0, height: 0 };
      await runStep('reading', async () => {
        try {
          dims = await readDimensions(file);
        } catch {
          throw new AnalysisError('This image couldn’t be opened for analysis.', 'unreadable_image');
        }
      });

      const choice = getScenario();
      await runStep('credentials');

      if (choice === 'analysis_error') {
        onStep?.({ step: 'model', status: 'active' });
        await wait(stepDelayMs, signal);
        throw new AnalysisError(
          'The image analysis service didn’t respond. This is a simulated error in the prototype.',
          'service_unavailable',
        );
      }

      await runStep('model');

      const scenario = choice === 'auto' ? pickAutoScenario(file, dims) : choice;
      const fixture = structuredClone(MOCK_FIXTURES[scenario]);
      return {
        ...fixture,
        imageInfo: {
          fileName: file.name,
          mimeType: file.type,
          sizeBytes: file.size,
          width: dims.width,
          height: dims.height,
        },
        analyzedAt: new Date().toISOString(),
        isIllustrative: true,
      };
    },
  };
}
