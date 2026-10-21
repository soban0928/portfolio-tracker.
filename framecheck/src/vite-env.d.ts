/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ANALYSIS_MODE?: 'mock' | 'live';
  readonly VITE_DETECTION_API_URL?: string;
  readonly VITE_PROVENANCE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
