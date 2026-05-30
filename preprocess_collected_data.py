#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np

# Configuration
ROOT = Path("/home/ubuntu/classifier/classifier")
OUT_BASE = Path("/home/ubuntu/rf_analysis-main/rf_analysis-main/dwingeloo/samples")

CONFIG_FILE = Path("/home/ubuntu/classifier/config.json")
SAMPLE_RATE = 48000.0

TARGET_MB = 2.0
STRIDE_MB = 0.5
MAX_SAMPLES = 1000


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}

    for sat, cfg in raw.items():
        if "bandwidth" in cfg and "min_signal_level" in cfg:
            out[str(sat)] = {"bandwidth": float(cfg["bandwidth"]), "min_signal_level": float(cfg["min_signal_level"])}

    return out


def bytes_to_iq_2ch(iq_bytes):
    arr = np.frombuffer(iq_bytes, dtype="<i2")

    if arr.size % 2:
        arr = arr[:-1]

    return np.stack([arr[0::2], arr[1::2]], axis=0)


def bandlimit_fft(x, sample_rate, bandwidth):
    n = x.size
    X = np.fft.fftshift(np.fft.fft(x))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))

    X[np.abs(freqs) > (bandwidth / 2.0)] = 0

    return np.fft.ifft(np.fft.ifftshift(X)).astype(np.complex64)


def atomic_write(path, data, is_npy=False):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    if is_npy:
        with open(tmp, "wb") as f:
            np.save(f, data)
    else:
        tmp.write_bytes(data)

    os.replace(tmp, path)


def process_file(iq_path, sat_name, sat_cfg, sample_rate, do_bandlimit):
    file_size = iq_path.stat().st_size

    # 4 bytes per complex sample (int16 I + int16 Q)
    target_bytes = int(TARGET_MB * 1024 * 1024)
    chunk_bytes = (target_bytes // 4) * 4

    stride_bytes = int(STRIDE_MB * 1024 * 1024)
    stride_bytes = (stride_bytes // 4) * 4

    if file_size < chunk_bytes:
        print(f"[SKIP] {iq_path.name}: File smaller than one chunk.")
        return

    # Calculate sliding window offsets
    offsets = []

    off = 0
    max_start = file_size - chunk_bytes

    while off <= max_start:
        offsets.append(off)

        if len(offsets) >= MAX_SAMPLES:
            break

        off += stride_bytes

    bw = sat_cfg["bandwidth"]
    thr = sat_cfg["min_signal_level"]

    # Format folder paths preserving directory casing
    clean_name = iq_path.name.replace(os.sep, "_").replace(" ", "_")
    sat_tag = sat_name.replace(" ", "_").upper()

    sat_base = OUT_BASE / f"{sat_name}_samples"

    iq_out_dir = sat_base / "iq_samples"
    npy_out_dir = sat_base / "numpy_samples"

    iq_out_dir.mkdir(parents=True, exist_ok=True)
    npy_out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0

    with open(iq_path, "rb") as f:
        for idx, off in enumerate(offsets, start=1):
            f.seek(off, os.SEEK_SET)

            raw = f.read(chunk_bytes)

            if len(raw) != chunk_bytes:
                continue

            iq2 = bytes_to_iq_2ch(raw)

            # Unpack into complex float array for signal DSP calculations
            x = (iq2[0].astype(np.float32) + 1j * iq2[1].astype(np.float32)).astype(np.complex64)

            if do_bandlimit:
                x = bandlimit_fft(x, sample_rate, bw)

            max_amp = float(np.max(np.abs(x)))

            if max_amp <= thr:
                continue

            out_stem = f"{sat_tag}_{clean_name}_{idx}"

            kept += 1

            if kept <= 5 or kept % 50 == 0:
                print(f"KEEP - {sat_name} {iq_path.name} #{idx}: off={off} max_amp={max_amp:.2f}")

            # Re-quantize to int16 data limits
            i_clipped = np.clip(np.real(x), -32768, 32767).astype("<i2")
            q_clipped = np.clip(np.imag(x), -32768, 32767).astype("<i2")

            # Format raw interleaved bytes
            if do_bandlimit:
                interleaved = np.empty((i_clipped.size * 2,), dtype="<i2")

                interleaved[0::2] = i_clipped
                interleaved[1::2] = q_clipped

                iq_bytes = interleaved.tobytes()
                iq2_out = np.stack([i_clipped, q_clipped], axis=0)

            else:
                iq_bytes = raw
                iq2_out = iq2

            atomic_write(iq_out_dir / f"{out_stem}.iq", iq_bytes)
            atomic_write(npy_out_dir / f"{out_stem}.npy", iq2_out, is_npy=True)

    print(f"DONE - {iq_path.name} ({sat_name}) -> saved {kept}/{len(offsets)} samples.")


def main():
    cfg = load_config(CONFIG_FILE)

    do_bandlimit = SAMPLE_RATE is not None

    if not ROOT.exists():
        print(f"Error: Target root directory {ROOT} does not exist.")
        return 1

    files_processed = 0

    for sat_dir in sorted(ROOT.iterdir()):
        if not sat_dir.is_dir() or sat_dir.name not in cfg:
            continue

        sat_cfg = cfg[sat_dir.name]

        iq_files = []

        for p in sat_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".iq":
                iq_files.append(p)

        for iq_path in sorted(iq_files):
            files_processed += 1

            process_file(
                iq_path=iq_path,
                sat_name=sat_dir.name,
                sat_cfg=sat_cfg,
                sample_rate=SAMPLE_RATE,
                do_bandlimit=do_bandlimit,
            )

    if files_processed == 0:
        print("Completed. No tracks or matching data variants were loaded.")

    return 0


if __name__ == "__main__":
    exit(main())
