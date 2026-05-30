#!/usr/bin/env python3
"""
Convert .wav files into IQ .npy samples (for Starlink beacon samples)
"""

import wave
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path("/home/ubuntu/rf_analysis-main/dwingeloo/samples/starlink_samples")

WAV_FILE = None
SAMPLE_TIME = 5.0


def read_wav_iq(path):
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if ch not in (1, 2):
        raise ValueError("WAV must be mono or stereo")

    # Map bit depths to types and normalizers
    typ_map = {
        1: (np.uint8, 128.0),
        2: (np.int16, 32768.0),
        4: (np.int32, 2147483648.0),
    }

    if width not in typ_map:
        raise ValueError(f"Unsupported sample width: {width}")

    dtype, divisor = typ_map[width]

    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)

    if width == 1:
        data = (data - divisor) / divisor
    else:
        data /= divisor

    data = data.reshape(-1, ch)

    i = data[:, 0]

    if ch == 2:
        q = data[:, 1]
    else:
        q = np.zeros_like(i)

    return i + 1j * q, sr


def convert_wav(wav_path, output_dir, sample_time):
    print(f"Processing {wav_path.name}")

    x, sr = read_wav_iq(wav_path)

    chunk_len = int(sr * sample_time)
    num_chunks = len(x) // chunk_len

    if num_chunks == 0:
        print(f"File {wav_path.name} too short for configured sample time.")
        return

    # Truncate trailing fractional chunk and reshape cleanly
    x = x[: num_chunks * chunk_len]

    chunks = x.reshape(num_chunks, chunk_len)

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_chunks):
        c = chunks[idx]

        # Format layout as shape (2, N)
        iq_sample = np.stack(
            (
                np.real(c),
                np.imag(c),
            ),
            axis=0
        ).astype(np.float32)

        outfile = output_dir / f"{wav_path.stem}_{idx:04d}.npy"

        np.save(outfile, iq_sample)

    print(f"  Wrote {num_chunks} samples")


def main():
    out_dir = Path(DEFAULT_OUT)

    if WAV_FILE is not None:
        files = [Path(WAV_FILE)]
    else:
        files = sorted(Path(".").glob("*.wav"))

    for f in files:
        try:
            convert_wav(
                wav_path=f,
                output_dir=out_dir,
                sample_time=SAMPLE_TIME,
            )

        except Exception as e:
            print(f"Error processing {f.name}: {e}")


if __name__ == "__main__":
    main()
