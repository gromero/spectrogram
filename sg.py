import librosa
import tensorflow as tf
import numpy as np
import argparse as ap
import sys
import os
import struct

# TODO(gromero): refactor
def load_raw(filename):
    # Raw file must have exactly 16000 PCM samples
    fsize = os.stat(filename).st_size
    # PCM 24-bit => 3 bytes
    chunk_size = 3
    chunks = fsize / chunk_size

    fd = open(filename, "rb")

    ff = []
    d = fd.read(chunk_size)
    print(d)
    while d is not None and len(d) == 3:
        # print(".")

        da = bytearray(d)
        da.insert(0, 0x00)
        i = struct.unpack("<i", da)
        f = float(i[0])
        f = f / 0x80000000 # normalization factor from int -> float
        ff.append(f)

        d = fd.read(chunk_size)

    ff = np.float32(ff)
    return ff

def main(args):
    filename = args.input

    _, ext = os.path.splitext(filename)
    if ext == ".wav":
        try:
            ts, ts_size = librosa.load(filename, sr=16000)

        except FileNotFoundError:
            print(f"Could not find input wave file {wav_file}")
            sys.exit(1)

    elif ext == ".raw":
        ts = load_raw(filename)
        ts_size = 16000

    else:
        print("Could not find file extension .wav or .raw, exiting...")
        sys.exit(1)

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

    print("stfts = ", stfts)
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

    filename = 0
    output = os.path.splitext(args.output)[filename]

    print("Saving to .npz...")
    # savez() will append extension .npz
    np.savez(output, input_1=dat_q)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Obtain a 1x49x10x1 spectrogram from a .wav or raw PCM 24-bit file")
    parser.add_argument("input", help="A .wav or .raw file, 1 seconds @ 16 kHz, i.e. 16000 samples")
    parser.add_argument("--output", help="Filename for the .npz output file", default="inputs.npz")

    args = parser.parse_args()

    main(args)
