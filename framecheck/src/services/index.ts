/**
 * Chooses the analysis backend.
 *
 * The prototype always uses the mock client. To connect real services later,
 * set `VITE_ANALYSIS_MODE=live` with `VITE_DETECTION_API_URL` and
 * `VITE_PROVENANCE_API_URL`, and update the privacy copy in the UI, since
 * images would then be sent to those services.
 */
import type { AnalysisClient } from '../types/analysis';
import { createHttpAnalysisClient } from './http/httpAnalysisClient';
import { createMockAnalysisClient, type MockAnalysisClientOptions } from './mock/mockAnalysisService';

export function createAnalysisClient(mockOptions?: MockAnalysisClientOptions): AnalysisClient {
  const env = import.meta.env;
  if (env.VITE_ANALYSIS_MODE === 'live' && env.VITE_DETECTION_API_URL && env.VITE_PROVENANCE_API_URL) {
    return createHttpAnalysisClient({
      detectionEndpoint: env.VITE_DETECTION_API_URL,
      provenanceEndpoint: env.VITE_PROVENANCE_API_URL,
    });
  }
  return createMockAnalysisClient(mockOptions);
}
