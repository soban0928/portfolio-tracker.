export const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;

interface SupportedFormat {
  mime: string;
  label: string;
  extensions: string[];
}

export const SUPPORTED_FORMATS: SupportedFormat[] = [
  { mime: 'image/jpeg', label: 'JPEG', extensions: ['jpg', 'jpeg'] },
  { mime: 'image/png', label: 'PNG', extensions: ['png'] },
  { mime: 'image/webp', label: 'WebP', extensions: ['webp'] },
  // Accepted, but whether it opens depends on the browser; decoding failures
  // are reported separately as an unreadable file.
  { mime: 'image/avif', label: 'AVIF', extensions: ['avif'] },
];

/** Value for the file input's `accept` attribute. */
export const ACCEPT_ATTRIBUTE = SUPPORTED_FORMATS.flatMap((f) => [
  f.mime,
  ...f.extensions.map((ext) => `.${ext}`),
]).join(',');

export const SUPPORTED_FORMAT_LABELS = SUPPORTED_FORMATS.map((f) => f.label);

export type FileValidationError =
  | { kind: 'unsupported'; fileName: string }
  | { kind: 'oversized'; fileName: string; sizeBytes: number }
  | { kind: 'empty'; fileName: string };

function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot === -1 ? '' : name.slice(dot + 1).toLowerCase();
}

/** Resolves the format from MIME type, falling back to the file extension. */
export function detectFormat(file: Pick<File, 'name' | 'type'>): SupportedFormat | null {
  if (file.type) {
    return SUPPORTED_FORMATS.find((f) => f.mime === file.type) ?? null;
  }
  const ext = extensionOf(file.name);
  return SUPPORTED_FORMATS.find((f) => f.extensions.includes(ext)) ?? null;
}

export function validateImageFile(
  file: Pick<File, 'name' | 'type' | 'size'>,
): FileValidationError | null {
  if (!detectFormat(file)) return { kind: 'unsupported', fileName: file.name };
  if (file.size === 0) return { kind: 'empty', fileName: file.name };
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return { kind: 'oversized', fileName: file.name, sizeBytes: file.size };
  }
  return null;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function describeValidationError(error: FileValidationError): {
  title: string;
  detail: string;
} {
  switch (error.kind) {
    case 'unsupported':
      return {
        title: 'This file type isn’t supported',
        detail: `“${error.fileName}” isn’t a ${SUPPORTED_FORMAT_LABELS.slice(0, -1).join(', ')} or ${SUPPORTED_FORMAT_LABELS.at(-1)} image. Try exporting it in one of those formats.`,
      };
    case 'oversized':
      return {
        title: 'This image is too large',
        detail: `“${error.fileName}” is ${formatBytes(error.sizeBytes)}. Images up to ${formatBytes(MAX_FILE_SIZE_BYTES)} are supported.`,
      };
    case 'empty':
      return {
        title: 'This file is empty',
        detail: `“${error.fileName}” has no data. Try selecting the image again.`,
      };
  }
}
