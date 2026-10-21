# FrameCheck (front-end prototype)

A responsive front-end for an image-authenticity checker. People upload an image
and get a careful summary of the available evidence: a model-based estimate and,
kept separate, the status of any C2PA Content Credential.

> **This is a prototype.** It doesn’t run a detection model or verify
> credentials. Every result is **illustrative mock data**, and the UI labels it
> that way. Images are read only in the browser (for a preview and their
> dimensions) and are never uploaded or stored.

## Run it

```bash
cd framecheck
npm install
npm run dev        # http://localhost:5173
npm test           # unit + component tests (Vitest, Testing Library)
npm run build      # type-check and build to dist/
```

Under **Prototype settings** (below the upload panel) you can pick which sample
result to show, including a simulated analysis error.

## Product principles in the UI

- Results are framed as a **model estimate** (“72% AI-like patterns”), never as
  “AI” or “real”, and the copy says the number isn’t a probability that the image is fake.
- **Provenance is separate.** The three states are *Verified credential found*,
  *Credential found but could not be verified*, and *No credential found*. A
  missing credential is never treated as evidence of AI generation.
- Observations are labeled as **supporting context only**, tagged by source
  (file, model, credential). There are no visual rules about eyes, hands, text or lighting.
- Every result includes a **Limits of this result** disclosure.

## Structure

```
src/
  types/analysis.ts                AnalysisResult and AnalysisClient contracts
  services/
    index.ts                       picks the mock client (default) or the live one
    mock/mockAnalysisFixtures.ts   ILLUSTRATIVE fixture data
    mock/mockAnalysisService.ts    mock client: real local file read, simulated steps
    http/detectionApi.ts           placeholder client for a detection API
    http/provenanceService.ts      placeholder client for a C2PA verification service
    http/httpAnalysisClient.ts     combines the two behind AnalysisClient
  lib/                             file validation, local image reading, sample image, summary copy
  components/                      UploadPanel, AnalysisProgress, ResultSummary, ConfidenceDisplay,
                                   ProvenanceStatus, ObservationsList, LimitationsNotice, HowItWorks…
```

### Connecting real services later

1. Implement the real request/response mapping in `services/http/detectionApi.ts`
   and `services/http/provenanceService.ts`. The response shapes there are
   placeholders, not a real contract.
2. Set `VITE_ANALYSIS_MODE=live`, `VITE_DETECTION_API_URL` and
   `VITE_PROVENANCE_API_URL`. `createAnalysisClient()` then returns the HTTP
   client, and the header’s privacy note switches to say images are sent for analysis.
3. Review the privacy copy in `HowItWorks.tsx`, which describes the
   browser-only prototype.

The UI depends only on `AnalysisResult`:

| Field               | Meaning |
| ------------------- | ------- |
| `modelLabel`        | `ai_like_patterns` · `mixed` · `few_ai_like_patterns` · `inconclusive` |
| `modelScore`        | 0–1 strength of AI-like patterns (not a probability of being fake), or `null` |
| `modelVersion`      | model build identifier |
| `provenanceStatus`  | `verified` · `unverified` · `not_found` |
| `provenanceDetails` | signer, signing time, claim generator, recorded actions, generative-AI declaration, validation notes |
| `observations`      | supporting observations tagged `file` / `model` / `provenance` |
| `limitations`       | caveats specific to this result |
