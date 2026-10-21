export const SAMPLE_FILE_NAME = 'framecheck-sample-harbor.png';

/**
 * Draws the prototype's sample image on a canvas, so the demo needs no bundled
 * or remote photo. It is a simple illustrated harbor scene.
 */
export async function createSampleImageFile(): Promise<File> {
  const width = 1200;
  const height = 800;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas is not available in this browser.');

  const sky = ctx.createLinearGradient(0, 0, 0, height * 0.62);
  sky.addColorStop(0, '#9fb3c8');
  sky.addColorStop(1, '#e9d9c1');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#f4e3c3';
  ctx.beginPath();
  ctx.arc(width * 0.72, height * 0.36, 70, 0, Math.PI * 2);
  ctx.fill();

  // Distant hills
  ctx.fillStyle = '#7d8ea3';
  ctx.beginPath();
  ctx.moveTo(0, height * 0.55);
  ctx.bezierCurveTo(width * 0.2, height * 0.42, width * 0.4, height * 0.5, width * 0.55, height * 0.47);
  ctx.bezierCurveTo(width * 0.75, height * 0.43, width * 0.9, height * 0.52, width, height * 0.5);
  ctx.lineTo(width, height * 0.62);
  ctx.lineTo(0, height * 0.62);
  ctx.closePath();
  ctx.fill();

  // Water
  const water = ctx.createLinearGradient(0, height * 0.62, 0, height);
  water.addColorStop(0, '#5f7891');
  water.addColorStop(1, '#2e3f55');
  ctx.fillStyle = water;
  ctx.fillRect(0, height * 0.62, width, height * 0.38);

  // Ripples
  ctx.strokeStyle = 'rgba(244, 227, 195, 0.35)';
  ctx.lineWidth = 3;
  for (let i = 0; i < 14; i++) {
    const y = height * 0.66 + i * 22;
    const x = width * 0.6 + Math.sin(i * 1.7) * 60;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 80 - i * 3, y);
    ctx.stroke();
  }

  // Pier
  ctx.fillStyle = '#3b3128';
  ctx.fillRect(0, height * 0.7, width * 0.38, 18);
  for (let i = 0; i < 6; i++) ctx.fillRect(30 + i * 75, height * 0.7, 10, 90);

  // Boat
  ctx.fillStyle = '#e8e2d6';
  ctx.beginPath();
  ctx.moveTo(width * 0.42, height * 0.74);
  ctx.lineTo(width * 0.58, height * 0.74);
  ctx.lineTo(width * 0.55, height * 0.79);
  ctx.lineTo(width * 0.45, height * 0.79);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#3b3128';
  ctx.fillRect(width * 0.495, height * 0.58, 5, height * 0.16);
  ctx.fillStyle = '#d7c7ab';
  ctx.beginPath();
  ctx.moveTo(width * 0.5 + 6, height * 0.6);
  ctx.lineTo(width * 0.56, height * 0.72);
  ctx.lineTo(width * 0.5 + 6, height * 0.72);
  ctx.closePath();
  ctx.fill();

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('Could not create the sample image.');
  return new File([blob], SAMPLE_FILE_NAME, { type: 'image/png' });
}
