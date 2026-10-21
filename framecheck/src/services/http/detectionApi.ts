/**
 * Placeholder client for a future image detection API.
 *
 * Not used by the prototype. The request and response shapes below are
 * assumptions to be replaced once a real service and its contract exist.
 */
import { AnalysisError, type ModelLabel, type Observation } from '../../types/analysis';

/** Assumed response shape — adjust to the real API contract. */
export interface DetectionApiResponse {
  label: ModelLabel;
  score: number | null;
  model_version: string;
  explanations?: { id: string; title: string; detail: string }[];
  caveats?: string[];
}

export interface DetectionOutcome {
  modelLabel: ModelLabel;
  modelScore: number | null;
  modelVersion: string;
  observations: Observation[];
  limitations: string[];
}

export async function requestDetection(
  endpoint: string,
  file: File,
  signal?: AbortSignal,
): Promise<DetectionOutcome> {
  const body = new FormData();
  body.append('image', file);

  let response: Response;
  try {
    response = await fetch(endpoint, { method: 'POST', body, signal });
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err;
    throw new AnalysisError('The image analysis service couldn’t be reached.', 'service_unavailable');
  }
  if (!response.ok) {
    throw new AnalysisError(`The image analysis service returned an error (${response.status}).`, 'service_unavailable');
  }

  const data = (await response.json()) as DetectionApiResponse;
  return {
    modelLabel: data.label,
    modelScore: data.score,
    modelVersion: data.model_version,
    // Model explanations are shown only as supporting observations.
    observations: (data.explanations ?? []).map((e) => ({ ...e, source: 'model' as const })),
    limitations: data.caveats ?? [],
  };
}
