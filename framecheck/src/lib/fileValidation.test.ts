import { MAX_FILE_SIZE_BYTES, describeValidationError, validateImageFile } from './fileValidation';

const file = (name: string, type: string, size = 1000) => ({ name, type, size });

describe('validateImageFile', () => {
  it.each([
    ['photo.jpg', 'image/jpeg'],
    ['photo.png', 'image/png'],
    ['photo.webp', 'image/webp'],
    ['photo.avif', 'image/avif'],
    ['photo.JPEG', ''],
  ])('accepts %s (%s)', (name, type) => {
    expect(validateImageFile(file(name, type))).toBeNull();
  });

  it('rejects unsupported formats', () => {
    expect(validateImageFile(file('anim.gif', 'image/gif'))).toEqual({ kind: 'unsupported', fileName: 'anim.gif' });
    expect(validateImageFile(file('notes.pdf', ''))?.kind).toBe('unsupported');
  });

  it('rejects files over the size limit', () => {
    expect(validateImageFile(file('big.png', 'image/png', MAX_FILE_SIZE_BYTES + 1))?.kind).toBe('oversized');
    expect(validateImageFile(file('ok.png', 'image/png', MAX_FILE_SIZE_BYTES))).toBeNull();
  });

  it('rejects empty files', () => {
    expect(validateImageFile(file('empty.png', 'image/png', 0))?.kind).toBe('empty');
  });

  it('describes errors in plain language', () => {
    const msg = describeValidationError({ kind: 'oversized', fileName: 'big.png', sizeBytes: 25 * 1024 * 1024 });
    expect(msg.title).toBe('This image is too large');
    expect(msg.detail).toContain('25.0 MB');
    expect(msg.detail).toContain('20.0 MB');
  });
});
