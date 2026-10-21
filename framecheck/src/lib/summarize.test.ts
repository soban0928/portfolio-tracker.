import { MOCK_FIXTURES } from '../services/mock/mockAnalysisFixtures';
import type { AnalysisResult } from '../types/analysis';
import { scoreToPercent, summarizeResult } from './summarize';

const build = (key: keyof typeof MOCK_FIXTURES): AnalysisResult => ({
  ...MOCK_FIXTURES[key],
  imageInfo: { fileName: 'x.png', mimeType: 'image/png', sizeBytes: 1, width: 1, height: 1 },
  analyzedAt: '2026-01-01T00:00:00Z',
  isIllustrative: true,
});

describe('summarizeResult', () => {
  it('uses cautious headlines driven by the model label', () => {
    expect(summarizeResult(build('ai_like_no_credential')).headline).toBe('Some signals suggest AI generation');
    expect(summarizeResult(build('mixed_unverified_credential')).headline).toBe('Signals are mixed');
  });

  it('never treats a missing credential as evidence of AI generation', () => {
    const { body } = summarizeResult(build('ai_like_no_credential'));
    expect(body).toContain('No Content Credential was found');
    expect(body).toContain('Many genuine images don’t carry one');
  });

  it('avoids absolute wording in every fixture', () => {
    for (const key of Object.keys(MOCK_FIXTURES) as (keyof typeof MOCK_FIXTURES)[]) {
      const { headline, body } = summarizeResult(build(key));
      expect(`${headline} ${body}`).not.toMatch(/definitely|100%|\bproven\b|is fake|is real|authentic/i);
    }
  });
});

describe('scoreToPercent', () => {
  it('rounds and clamps', () => {
    expect(scoreToPercent(0.724)).toBe(72);
    expect(scoreToPercent(1.4)).toBe(100);
    expect(scoreToPercent(-1)).toBe(0);
  });
});
