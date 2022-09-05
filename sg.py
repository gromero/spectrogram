import librosa
import tensorflow as tf
import numpy as np

ts, ts_size = librosa.load("./ffb86d3c_nohash_0.wav", sr=16000)
ts_tensor = tf.convert_to_tensor(ts)
print(ts_tensor)
wav_decoder = tf.cast(ts_tensor, tf.float32)
print(wav_decoder)
# wav_decoder = wav_decoder/tf.constant(2**15,dtype=tf.float32)
wav_decoder = wav_decoder/tf.reduce_max(wav_decoder)
print(tf.shape(wav_decoder)[-1])

time_series_samples = wav_decoder # 16000 samples, i.e. 1 s or 1000 ms

window_size_samples = 480   # 30 ms
window_stride_samples = 320 # 20 ms

stfts = tf.signal.stft(time_series_samples, frame_length=window_size_samples, frame_step=window_stride_samples, fft_length=512, window_fn=tf.signal.hann_window)

spectrograms = tf.abs(stfts) # compute magnitudes

print(stfts)
tf.print(spectrograms, summarize=-1)
# stfts.shape = 512 (fft_length) // 2 + 1 = 257
num_spectrogram_bins = stfts.shape[-1]
print("num_spectrogram_bins = ", num_spectrogram_bins)

sample_rate = 16000
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                    sample_rate,
                                                                    lower_edge_hertz, upper_edge_hertz)

print(linear_to_mel_weight_matrix)
tf.print(linear_to_mel_weight_matrix, summarize=-1)

mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
tf.print(mel_spectrograms, summarize=-1)
print(mel_spectrograms)

# I don't see why that step is necessary
mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

tf.print(mel_spectrograms, summarize=-1)
print(mel_spectrograms)

# Compute a stabilized log to get log-magnitude mel-scale spectrograms.

x = mel_spectrograms + 1e-6
print("x = ")
print(x)

log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
tf.print(log_mel_spectrograms, summarize=-1)
print(log_mel_spectrograms)

dct_coefficient_count = 10 # How many MFCC or log filterbank energy features
spectrogram_length = 49 # ((total_size_samples - window_size_samples) / stride_size_sample) + 1

# input shape is 1 x 49 x 10 x 1

mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :dct_coefficient_count]
print(mfccs)
mfccs = tf.reshape(mfccs,[1, spectrogram_length, dct_coefficient_count, 1])
print("mfccs =", mfccs)

input_scale = 0.5847029089927673
input_zero_point = 83
dat_q = np.array(mfccs/input_scale + input_zero_point, dtype=np.int8)
print("dat_q = ", dat_q)


print("Saving to .npz...")
np.savez("./inputs.npz", input_1=dat_q)
