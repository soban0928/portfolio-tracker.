import type { AnalysisResult, ModelLabel } from '../types/analysis';

export interface ResultSummary {
  headline: string;
  body: string;
}

const MODEL_HEADLINES: Record<ModelLabel, string> = {
  ai_like_patterns: 'Some signals suggest AI generation',
  mixed: 'Signals are mixed',
  few_ai_like_patterns: 'Few signals suggest AI generation',
  inconclusive: 'The analysis couldn’t reach an estimate',
};

const MODEL_SENTENCES: Record<ModelLabel, string> = {
  ai_like_patterns:
    'The image analysis model found patterns it often associates with AI-generated images.',
  mixed: 'The image analysis model found some patterns it associates with AI-generated images, and some it doesn’t.',
  few_ai_like_patterns:
    'The image analysis model found few patterns it associates with AI-generated images. That doesn’t establish where the image came from.',
  inconclusive:
    'The image analysis model couldn’t produce a usable estimate for this file, so the result relies on any provenance information alone.',
};

/**
 * Builds the plain-language headline. The headline is driven by the model
 * label; provenance adds context but never flips it into a verdict.
 */
export function summarizeResult(result: AnalysisResult): ResultSummary {
  const parts = [MODEL_SENTENCES[result.modelLabel]];

  switch (result.provenanceStatus) {
    case 'verified':
      parts.push(
        result.provenanceDetails?.declaresGenerativeAI
          ? 'A verified Content Credential is attached and it records the use of generative AI.'
          : 'A verified Content Credential is attached, which describes a signed origin and edit history.',
      );
      break;
    case 'unverified':
      parts.push('A Content Credential is attached, but it couldn’t be verified, so treat its claims with care.');
      break;
    case 'not_found':
      parts.push('No Content Credential was found. Many genuine images don’t carry one.');
      break;
  }

  return { headline: MODEL_HEADLINES[result.modelLabel], body: parts.join(' ') };
}

/** 0.723 → 72 */
export function scoreToPercent(score: number): number {
  return Math.round(Math.min(1, Math.max(0, score)) * 100);
}
