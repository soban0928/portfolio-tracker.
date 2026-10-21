/**
 * Shared contract between the FrameCheck UI and whatever produces analysis
 * results. Today that is the mock service in `services/mock/`; later it can be
 * a real detection API plus a C2PA Content Credentials verification service
 * (see `services/http/`). UI components only ever depend on these types.
 */

/**
 * Coarse label from the detection model. It describes the *patterns* the model
 * saw, never a verdict about who or what made the image.
 */
export type ModelLabel =
  /** Patterns the model associates with AI-generated images dominate. */
  | 'ai_like_patterns'
  /** Patterns point in different directions. */
  | 'mixed'
  /** Few patterns the model associates with AI generation. */
  | 'few_ai_like_patterns'
  /** The model could not produce a usable estimate (too small, too compressed, etc.). */
  | 'inconclusive';

/**
 * Outcome of looking for C2PA Content Credentials. Kept separate from the model
 * result: a missing credential is not evidence of AI generation.
 */
export type ProvenanceStatus =
  /** A credential was found and its signature validated. */
  | 'verified'
  /** A credential was found but could not be validated (broken chain, untrusted signer, tampering...). */
  | 'unverified'
  /** No credential was found in the file. */
  | 'not_found';

/** A single entry from a credential's edit history (a C2PA "action"). */
export interface ProvenanceAction {
  /** Plain-language description, e.g. "Cropped" or "Created". */
  label: string;
  /** Software agent recorded for this action, if any. */
  softwareAgent?: string;
  /** ISO 8601 timestamp, if recorded. */
  when?: string;
}

export interface ProvenanceDetails {
  /** Name of the entity that signed the credential, as stated in the certificate. */
  signer?: string;
  /** ISO 8601 time the credential was signed. */
  signedAt?: string;
  /** Tool that produced the credential (C2PA "claim generator"). */
  claimGenerator?: string;
  /** Recorded edit history, oldest first. */
  actions?: ProvenanceAction[];
  /** Whether the credential declares generative AI involvement (C2PA digitalSourceType). */
  declaresGenerativeAI?: boolean;
  /** Validation messages, especially for `unverified` credentials. */
  validationNotes?: string[];
}

export type ObservationSource =
  /** File-level facts: format, dimensions, metadata presence, compression. */
  | 'file'
  /** Explanations returned by the detection model. Supporting context only. */
  | 'model'
  /** Facts read from a Content Credential. */
  | 'provenance';

export interface Observation {
  id: string;
  source: ObservationSource;
  /** Short, neutral statement. */
  title: string;
  /** One or two sentences of context. */
  detail: string;
}

/** Basic facts about the selected file, read locally in the browser. */
export interface ImageInfo {
  fileName: string;
  mimeType: string;
  sizeBytes: number;
  width: number;
  height: number;
}

export interface AnalysisResult {
  /** Detection model label. */
  modelLabel: ModelLabel;
  /**
   * Model output in the range 0–1: how strongly the image matches patterns the
   * model associates with AI generation. NOT a probability that the image is
   * fake. `null` when the model could not produce an estimate.
   */
  modelScore: number | null;
  /** Identifier of the model build that produced the score. */
  modelVersion: string;
  provenanceStatus: ProvenanceStatus;
  /** Present when a credential was found (verified or not). */
  provenanceDetails: ProvenanceDetails | null;
  /** Supporting observations. None of these is a rule on its own. */
  observations: Observation[];
  /** Caveats specific to this result (in addition to the general limits). */
  limitations: string[];
  /** Facts read locally from the file. */
  imageInfo: ImageInfo;
  /** ISO 8601 time the analysis finished. */
  analyzedAt: string;
  /**
   * True when the result comes from mock fixtures rather than real services.
   * The UI must clearly label these as illustrative.
   */
  isIllustrative: boolean;
}

/** The visible steps of an analysis run, in order. */
export type AnalysisStepId = 'reading' | 'credentials' | 'model';

export interface AnalysisStepUpdate {
  step: AnalysisStepId;
  status: 'active' | 'done';
}

export interface AnalyzeOptions {
  signal?: AbortSignal;
  onStep?: (update: AnalysisStepUpdate) => void;
}

/** Anything that can turn an image into an `AnalysisResult`. */
export interface AnalysisClient {
  /** True when this client produces illustrative (mock) results. */
  readonly isMock: boolean;
  analyzeImage(file: File, options?: AnalyzeOptions): Promise<AnalysisResult>;
}

/** Thrown by clients when analysis fails in a way the user should hear about. */
export class AnalysisError extends Error {
  constructor(
    message: string,
    readonly code: 'service_unavailable' | 'unreadable_image' | 'timeout' | 'unknown' = 'unknown',
  ) {
    super(message);
    this.name = 'AnalysisError';
  }
}
