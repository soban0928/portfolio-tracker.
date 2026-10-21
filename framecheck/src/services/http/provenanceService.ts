/**
 * Placeholder client for a future C2PA Content Credentials verification service.
 *
 * Not used by the prototype. A real implementation might call a server-side
 * verifier or run a C2PA library in the browser; either way it should map its
 * output onto `ProvenanceStatus` and `ProvenanceDetails` so the UI stays the
 * same. A missing manifest must map to `not_found`, never to an AI verdict.
 */
import { AnalysisError, type ProvenanceDetails, type ProvenanceStatus } from '../../types/analysis';

/** Assumed response shape — adjust to the real verifier's contract. */
export interface ProvenanceApiResponse {
  manifest_present: boolean;
  signature_valid?: boolean;
  signer?: string;
  signed_at?: string;
  claim_generator?: string;
  actions?: { label: string; software_agent?: string; when?: string }[];
  declares_generative_ai?: boolean;
  validation_messages?: string[];
}

export interface ProvenanceOutcome {
  provenanceStatus: ProvenanceStatus;
  provenanceDetails: ProvenanceDetails | null;
}

export function mapProvenanceResponse(data: ProvenanceApiResponse): ProvenanceOutcome {
  if (!data.manifest_present) return { provenanceStatus: 'not_found', provenanceDetails: null };
  return {
    provenanceStatus: data.signature_valid ? 'verified' : 'unverified',
    provenanceDetails: {
      signer: data.signer,
      signedAt: data.signed_at,
      claimGenerator: data.claim_generator,
      declaresGenerativeAI: data.declares_generative_ai,
      actions: data.actions?.map((a) => ({ label: a.label, softwareAgent: a.software_agent, when: a.when })),
      validationNotes: data.validation_messages,
    },
  };
}

export async function requestProvenance(
  endpoint: string,
  file: File,
  signal?: AbortSignal,
): Promise<ProvenanceOutcome> {
  const body = new FormData();
  body.append('image', file);

  let response: Response;
  try {
    response = await fetch(endpoint, { method: 'POST', body, signal });
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err;
    throw new AnalysisError('The credential verification service couldn’t be reached.', 'service_unavailable');
  }
  if (!response.ok) {
    throw new AnalysisError(`The credential verification service returned an error (${response.status}).`, 'service_unavailable');
  }
  return mapProvenanceResponse((await response.json()) as ProvenanceApiResponse);
}
