import numpy as np
import soundfile as sf
import librosa

def wavfile_to_examples(wav_file, return_tensor=True):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.
    torch: Return data as a Pytorch tensor ready for VGGish

  Returns:
    See waveform_to_examples.
  """
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    print(type(samples), samples.dtype)
    print(samples)

wavfile_to_examples('../test.wav')

au = librosa.load('../a.mp3', sr=None)
print(type(au))
