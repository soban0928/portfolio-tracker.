/**
 * ILLUSTRATIVE MOCK DATA — NOT REAL ANALYSIS.
 *
 * These fixtures stand in for responses from a future detection API and C2PA
 * verification service. Scores, signers, observations and versions are
 * invented for the prototype and say nothing about any real image.
 */
import type { AnalysisResult } from '../../types/analysis';

/** Fields the mock service fills in at runtime from the selected file. */
type FixtureResult = Omit<AnalysisResult, 'imageInfo' | 'analyzedAt' | 'isIllustrative'>;

export type MockScenarioId =
  | 'ai_like_no_credential'
  | 'mixed_unverified_credential'
  | 'few_ai_like_verified_credential'
  | 'ai_like_verified_genai_credential'
  | 'inconclusive_no_credential';

export const MOCK_MODEL_VERSION = 'mock-detector 0.1 (illustrative)';

const SAMPLE_SIGNER = 'Sample Signer (illustrative)';
const SAMPLE_GENERATOR = 'Sample Editing App 1.0 (illustrative)';

export const MOCK_FIXTURES: Record<MockScenarioId, FixtureResult> = {
  ai_like_no_credential: {
    modelLabel: 'ai_like_patterns',
    modelScore: 0.72,
    modelVersion: MOCK_MODEL_VERSION,
    provenanceStatus: 'not_found',
    provenanceDetails: null,
    observations: [
      {
        id: 'texture-statistics',
        source: 'model',
        title: 'Image-wide patterns resemble AI-generated examples',
        detail:
          'The model’s estimate comes from statistical patterns spread across the whole image, not from any single feature you could point to.',
      },
      {
        id: 'no-camera-metadata',
        source: 'file',
        title: 'No camera metadata present',
        detail:
          'Many apps and platforms remove metadata when sharing, so its absence is common and not meaningful on its own.',
      },
      {
        id: 'single-save',
        source: 'file',
        title: 'Little sign of repeated compression',
        detail: 'The file doesn’t appear to have been re-saved many times, which helps the model read it.',
      },
    ],
    limitations: [
      'With no credential, this result rests on the model estimate alone.',
    ],
  },

  mixed_unverified_credential: {
    modelLabel: 'mixed',
    modelScore: 0.48,
    modelVersion: MOCK_MODEL_VERSION,
    provenanceStatus: 'unverified',
    provenanceDetails: {
      signer: SAMPLE_SIGNER,
      claimGenerator: SAMPLE_GENERATOR,
      signedAt: '2026-03-14T09:21:00Z',
      actions: [{ label: 'Opened' }, { label: 'Color adjustments' }, { label: 'Resized' }],
      validationNotes: [
        'The image data no longer matches what the credential signed. It may have been edited after signing.',
      ],
    },
    observations: [
      {
        id: 'uneven-regions',
        source: 'model',
        title: 'Estimate differs across regions of the image',
        detail:
          'Some areas look more AI-like to the model than others. This can happen with composites, heavy retouching, or local edits.',
      },
      {
        id: 'recompressed',
        source: 'file',
        title: 'Signs of repeated compression',
        detail: 'The file appears to have been saved more than once, which can blur the patterns the model looks for.',
      },
      {
        id: 'credential-mismatch',
        source: 'provenance',
        title: 'Credential doesn’t match the current pixels',
        detail: 'A credential is attached, but the image appears to have changed after it was signed.',
      },
    ],
    limitations: [
      'Repeated compression can push the estimate in either direction.',
      'An unverified credential’s edit history can’t be relied on.',
    ],
  },

  few_ai_like_verified_credential: {
    modelLabel: 'few_ai_like_patterns',
    modelScore: 0.14,
    modelVersion: MOCK_MODEL_VERSION,
    provenanceStatus: 'verified',
    provenanceDetails: {
      signer: SAMPLE_SIGNER,
      claimGenerator: SAMPLE_GENERATOR,
      signedAt: '2026-05-02T16:04:00Z',
      declaresGenerativeAI: false,
      actions: [
        { label: 'Captured', when: '2026-05-02T15:58:00Z' },
        { label: 'Cropped', softwareAgent: SAMPLE_GENERATOR },
        { label: 'Exposure adjusted', softwareAgent: SAMPLE_GENERATOR },
      ],
    },
    observations: [
      {
        id: 'credential-history',
        source: 'provenance',
        title: 'Credential records a capture, a crop and an exposure change',
        detail: 'The signed history lists these steps. It only covers what the signing tools recorded.',
      },
      {
        id: 'consistent-estimate',
        source: 'model',
        title: 'Estimate is fairly consistent across the image',
        detail: 'The model didn’t find areas that stood out strongly from the rest.',
      },
    ],
    limitations: [
      'A verified credential confirms who signed it and what was recorded, not that the scene is accurate.',
    ],
  },

  ai_like_verified_genai_credential: {
    modelLabel: 'ai_like_patterns',
    modelScore: 0.86,
    modelVersion: MOCK_MODEL_VERSION,
    provenanceStatus: 'verified',
    provenanceDetails: {
      signer: SAMPLE_SIGNER,
      claimGenerator: SAMPLE_GENERATOR,
      signedAt: '2026-08-19T11:37:00Z',
      declaresGenerativeAI: true,
      actions: [
        { label: 'Created with a generative AI tool', softwareAgent: SAMPLE_GENERATOR },
        { label: 'Upscaled', softwareAgent: SAMPLE_GENERATOR },
      ],
    },
    observations: [
      {
        id: 'credential-genai',
        source: 'provenance',
        title: 'Credential declares generative AI was used',
        detail: 'The signed history states the image was created with a generative tool.',
      },
      {
        id: 'image-wide-patterns',
        source: 'model',
        title: 'Image-wide patterns resemble AI-generated examples',
        detail: 'The model estimate and the credential point the same way, but they are separate pieces of evidence.',
      },
    ],
    limitations: [],
  },

  inconclusive_no_credential: {
    modelLabel: 'inconclusive',
    modelScore: null,
    modelVersion: MOCK_MODEL_VERSION,
    provenanceStatus: 'not_found',
    provenanceDetails: null,
    observations: [
      {
        id: 'low-detail',
        source: 'file',
        title: 'Very little detail to work with',
        detail:
          'The image is small or heavily compressed, so the patterns the model relies on are mostly lost.',
      },
      {
        id: 'possible-screenshot',
        source: 'file',
        title: 'File looks like a screenshot or re-capture',
        detail: 'Screenshots replace the original file, removing credentials and much of the original detail.',
      },
    ],
    limitations: ['Try the original file if you can find it — screenshots and re-shares lose information.'],
  },
};

export const MOCK_SCENARIO_LABELS: Record<MockScenarioId, string> = {
  ai_like_no_credential: 'AI-like patterns, no credential',
  mixed_unverified_credential: 'Mixed signals, unverifiable credential',
  few_ai_like_verified_credential: 'Few AI-like patterns, verified credential',
  ai_like_verified_genai_credential: 'AI-like patterns, credential declares AI',
  inconclusive_no_credential: 'Inconclusive, no credential',
};

/** The scenario always used for the built-in sample image. */
export const SAMPLE_IMAGE_SCENARIO: MockScenarioId = 'ai_like_no_credential';
