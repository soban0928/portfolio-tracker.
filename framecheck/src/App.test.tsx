import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { MAX_FILE_SIZE_BYTES } from './lib/fileValidation';

// jsdom can't decode images, so stub the local file reader.
vi.mock('./lib/readImage', () => ({
  readImageFile: vi.fn(async (_file: File, onProgress?: (p: number) => void) => {
    onProgress?.(1);
    return { previewUrl: 'blob:preview', width: 1600, height: 1200 };
  }),
}));

beforeAll(() => {
  URL.revokeObjectURL = vi.fn();
});

const mockOptions = { stepDelayMs: 0, readDimensions: async () => ({ width: 1600, height: 1200 }) };

function fileInput(container: HTMLElement) {
  return container.querySelector('input[type="file"]') as HTMLInputElement;
}

function selectFile(container: HTMLElement, file: File) {
  fireEvent.change(fileInput(container), { target: { files: [file] } });
}

describe('FrameCheck app', () => {
  it('renders the heading, privacy note and how-it-works section', () => {
    render(<App mockOptions={mockOptions} />);
    expect(screen.getByRole('heading', { level: 1, name: 'Check the signals behind an image.' })).toBeInTheDocument();
    expect(screen.getByText(/Images aren’t uploaded or stored/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'How it works' })).toHaveAttribute('href', '#how-it-works');
    expect(screen.getByRole('heading', { name: 'How this works' })).toBeInTheDocument();
  });

  it('shows an error for unsupported files', () => {
    const { container } = render(<App mockOptions={mockOptions} />);
    selectFile(container, new File(['x'], 'clip.gif', { type: 'image/gif' }));
    expect(screen.getByRole('alert')).toHaveTextContent('This file type isn’t supported');
  });

  it('shows an error for oversized files', () => {
    const { container } = render(<App mockOptions={mockOptions} />);
    const big = new File(['x'], 'huge.png', { type: 'image/png' });
    Object.defineProperty(big, 'size', { value: MAX_FILE_SIZE_BYTES + 1 });
    selectFile(container, big);
    expect(screen.getByRole('alert')).toHaveTextContent('This image is too large');
  });

  it('runs a mock analysis and shows a clearly labeled, cautious result', async () => {
    const user = userEvent.setup();
    const { container } = render(<App mockOptions={mockOptions} />);
    selectFile(container, new File(['x'], 'photo.jpg', { type: 'image/jpeg' }));

    expect(await screen.findByAltText('Preview of photo.jpg')).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Sample result to show'), 'ai_like_no_credential');
    await user.click(screen.getByRole('button', { name: 'Check this image' }));

    const heading = await screen.findByRole('heading', { name: 'Some signals suggest AI generation' });
    expect(heading).toHaveFocus();
    expect(screen.getByText('Illustrative sample result')).toBeInTheDocument();
    expect(screen.getByRole('meter')).toHaveAttribute('aria-valuenow', '72');
    expect(screen.getByText(/not the probability that the image is fake/)).toBeInTheDocument();

    const credentials = screen.getByRole('region', { name: 'Content credentials' });
    expect(within(credentials).getByText('No credential found')).toBeInTheDocument();
    expect(within(credentials).getByText(/doesn’t mean the image is AI-generated/)).toBeInTheDocument();

    expect(screen.getByRole('heading', { name: 'What we noticed' })).toBeInTheDocument();
    expect(screen.getByText('Limits of this result')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Check another image' }));
    expect(screen.getByRole('button', { name: 'Choose an image' })).toBeInTheDocument();
  });

  it('shows the analysis error state and lets the user retry', async () => {
    const user = userEvent.setup();
    const { container } = render(<App mockOptions={mockOptions} />);
    selectFile(container, new File(['x'], 'photo.png', { type: 'image/png' }));
    await screen.findByAltText('Preview of photo.png');

    await user.selectOptions(screen.getByLabelText('Sample result to show'), 'analysis_error');
    await user.click(screen.getByRole('button', { name: 'Check this image' }));
    expect(await screen.findByRole('heading', { name: 'The check couldn’t be completed' })).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Sample result to show'), 'mixed_unverified_credential');
    await user.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('heading', { name: 'Signals are mixed' })).toBeInTheDocument();
    expect(screen.getByText('Credential found but could not be verified')).toBeInTheDocument();
  });
});
