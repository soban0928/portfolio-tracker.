export interface LoadedImage {
  /** Object URL for previews. Revoke with `URL.revokeObjectURL` when done. */
  previewUrl: string;
  width: number;
  height: number;
}

/**
 * Reads a file locally, reporting progress, then confirms the browser can
 * decode it. Nothing leaves the device.
 */
export function readImageFile(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<LoadedImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) onProgress?.(event.loaded / event.total);
    };
    reader.onerror = () => reject(new Error('read_failed'));
    reader.onload = () => {
      onProgress?.(1);
      const blob = new Blob([reader.result as ArrayBuffer], { type: file.type });
      const previewUrl = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => resolve({ previewUrl, width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => {
        URL.revokeObjectURL(previewUrl);
        reject(new Error('decode_failed'));
      };
      img.src = previewUrl;
    };
    reader.readAsArrayBuffer(file);
  });
}
