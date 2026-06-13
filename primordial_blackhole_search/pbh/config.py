"""Project-wide constants. One place, no magic numbers elsewhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
NOISE_DIR = DATA_DIR / "noise"
SHARD_DIR = DATA_DIR / "shards"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# --- data
SAMPLE_RATE = 4096  # Hz
SEGMENT_LEN = 4096  # seconds per cached noise segment
N_SEGMENTS_H1 = 16  # ~18 hours of single-detector data
# O3a scan window for quiet segments (starts 2019-05-21, well into O3a)
SCAN_START = 1242000000
SCAN_DAYS = 45

# train/val/test split over segments (by index after sorting by GPS)
N_TRAIN_SEG = 8
N_VAL_SEG = 2
N_TEST_SEG = 6

# --- signal band / windowing
F_LOWER = 50.0  # Hz analysis low cutoff (O3 noise wall below; keeps signals <~6 min)
F_HIGH = 1024.0  # Hz upper edge for spectrograms
WINDOW_SEC = 256  # v1 detection window the models see
WINDOW_SEC_SHORT = 64  # v2 rung 2: shorter window for independent track views
PSD_SEG_SEC = 16  # Welch segment length for PSD estimation
WHITEN_CROP_SEC = 8  # corrupted edges to drop after whitening

# --- spectrogram
STFT_SEC = 1.0  # FFT length (1 Hz resolution - thin tracks linger at low f)
STFT_HOP_SEC = 0.5
N_FREQ_BINS = 128  # log-spaced 50 -> 1024 Hz
N_TIME_BINS = 256  # 256 s / 0.5 s hop = 512, max-pooled x2 in time

# --- injection population (the subsolar target)
M_MIN, M_MAX = 0.2, 1.0  # component masses, solar masses, log-uniform
TRAIN_SNR_RANGE = (5.0, 30.0)  # optimal network SNR for training injections
EVAL_SNR_RANGE = (4.0, 24.0)  # injection SNR range for sensitivity campaigns
APPROXIMANT = "TaylorF2"

# --- training
BATCH_SIZE = 64
SEED = 20260610
