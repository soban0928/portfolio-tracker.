export interface ImageDimensions {
  width: number;
  height: number;
}

/** Decodes an image locally to read its pixel dimensions. */
export async function readImageDimensions(file: Blob): Promise<ImageDimensions> {
  if (typeof createImageBitmap === 'function') {
    const bitmap = await createImageBitmap(file);
    const dims = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return dims;
  }
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => reject(new Error('decode_failed'));
      img.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}
