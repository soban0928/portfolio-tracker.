/**
 * `AnalysisClient` that combines the detection API and the provenance service.
 * Not active in the prototype; see `services/index.ts`.
 */
import type { AnalysisClient, AnalysisResult, AnalyzeOptions } from '../../types/analysis';
import { AnalysisError } from '../../types/analysis';
import { readImageDimensions } from '../imageDimensions';
import { requestDetection } from './detectionApi';
import { requestProvenance } from './provenanceService';

export interface HttpAnalysisClientConfig {
  detectionEndpoint: string;
  provenanceEndpoint: string;
}

export function createHttpAnalysisClient(config: HttpAnalysisClientConfig): AnalysisClient {
  return {
    isMock: false,
    async analyzeImage(file: File, { signal, onStep }: AnalyzeOptions = {}): Promise<AnalysisResult> {
      onStep?.({ step: 'reading', status: 'active' });
      const dims = await readImageDimensions(file).catch(() => {
        throw new AnalysisError('This image couldn’t be opened for analysis.', 'unreadable_image');
      });
      onStep?.({ step: 'reading', status: 'done' });

      onStep?.({ step: 'credentials', status: 'active' });
      const provenance = await requestProvenance(config.provenanceEndpoint, file, signal);
      onStep?.({ step: 'credentials', status: 'done' });

      onStep?.({ step: 'model', status: 'active' });
      const detection = await requestDetection(config.detectionEndpoint, file, signal);
      onStep?.({ step: 'model', status: 'done' });

      return {
        ...detection,
        ...provenance,
        imageInfo: { fileName: file.name, mimeType: file.type, sizeBytes: file.size, ...dims },
        analyzedAt: new Date().toISOString(),
        isIllustrative: false,
      };
    },
  };
}
