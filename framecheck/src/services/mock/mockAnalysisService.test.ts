import { AnalysisError, type AnalysisStepUpdate } from '../../types/analysis';
import { SAMPLE_FILE_NAME } from '../../lib/sampleImage';
import { SAMPLE_IMAGE_SCENARIO } from './mockAnalysisFixtures';
import { createMockAnalysisClient, pickAutoScenario, type MockScenarioChoice } from './mockAnalysisService';

const dims = { width: 1600, height: 1200 };
const makeFile = (name = 'photo.jpg') => new File(['abc'], name, { type: 'image/jpeg' });
const makeClient = (scenario: MockScenarioChoice = 'auto') =>
  createMockAnalysisClient({ getScenario: () => scenario, stepDelayMs: 0, readDimensions: async () => dims });

describe('mock analysis client', () => {
  it('reports each step in order and returns an illustrative result', async () => {
    const updates: AnalysisStepUpdate[] = [];
    const result = await makeClient().analyzeImage(makeFile(), { onStep: (u) => updates.push(u) });

    expect(updates.map((u) => `${u.step}:${u.status}`)).toEqual([
      'reading:active',
      'reading:done',
      'credentials:active',
      'credentials:done',
      'model:active',
      'model:done',
    ]);
    expect(result.isIllustrative).toBe(true);
    expect(result.imageInfo).toMatchObject({ fileName: 'photo.jpg', width: 1600, height: 1200 });
  });

  it('uses the fixed scenario for the sample image', () => {
    expect(pickAutoScenario({ name: SAMPLE_FILE_NAME, size: 10 }, dims)).toBe(SAMPLE_IMAGE_SCENARIO);
  });

  it('is deterministic per file and inconclusive for tiny images', () => {
    const f = { name: 'a.jpg', size: 1234 };
    expect(pickAutoScenario(f, dims)).toBe(pickAutoScenario(f, dims));
    expect(pickAutoScenario(f, { width: 100, height: 80 })).toBe('inconclusive_no_credential');
  });

  it('honours an explicit scenario', async () => {
    const result = await makeClient('few_ai_like_verified_credential').analyzeImage(makeFile());
    expect(result.provenanceStatus).toBe('verified');
    expect(result.modelLabel).toBe('few_ai_like_patterns');
  });

  it('can simulate an analysis error', async () => {
    await expect(makeClient('analysis_error').analyzeImage(makeFile())).rejects.toBeInstanceOf(AnalysisError);
  });

  it('stops when aborted', async () => {
    const controller = new AbortController();
    const client = createMockAnalysisClient({ stepDelayMs: 1000, readDimensions: async () => dims });
    const pending = client.analyzeImage(makeFile(), { signal: controller.signal });
    controller.abort(new DOMException('cancelled', 'AbortError'));
    await expect(pending).rejects.toThrow('cancelled');
  });
});
