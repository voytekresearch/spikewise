# Directories
PARENT_DIR = "/labs/bvoyteklab/Bender/Tovar_data"
THINGS_DIR = f"{PARENT_DIR}/THINGS"
THINGS_PLUS_DIR = f"{PARENT_DIR}/THINGSplus"
THINGS_OOO_DIR = f"{PARENT_DIR}/THINGS-odd-one-out"
IMAGES_DIR = f"{THINGS_DIR}/Images"
MEG_DIR = f"{PARENT_DIR}/THINGS-data/THINGS-MEG/ds004212/derivatives"
PREPROCESSED_DIR = f"{MEG_DIR}/preprocessed"
SPARAM_DIR = f"{PARENT_DIR}/sparam"
OUTPUT_DIR = f"{PARENT_DIR}/output"
LAYER_ACTIVATION_DIR = f"{OUTPUT_DIR}/layer_activations"

# Files
EMBEDDINGS49_FNAME = (
    f"{THINGS_OOO_DIR}/data/embedding49/spose_embedding_49d_sorted.txt"
)
EMBEDDINGS49_LABELS_FNAME = (
    f"{THINGS_OOO_DIR}/data/embedding49/labels_short49.mat"
)
EMBEDDINGS66_FNAME = f"{THINGS_OOO_DIR}/data/spose_embedding_66d_sorted.txt"
EMBEDDINGS66_LABELS_FNAME = f"{THINGS_OOO_DIR}/variables/labels.txt"
OBJECT_LABELS_FNAME = f"{THINGS_OOO_DIR}/variables/unique_id.txt"
TRAINED_PAIRWISE_DECODING_RDM_FNAME = (
    f"{MEG_DIR}/output/validation-pairwise_decoding_RDM1854.mat"
)

# Single image embeddings model
SOURCE = "custom"
MODEL = "clip"
MODEL_VARIANT = "ViT-L/14@336px"
MODULE = "visual.ln_post"

# Spectral estimation parameters
FMIN = 1  # Hz
FMAX = 50  # Hz
N_FREQS = 128
TIME_WINDOW_LEN = 1.0  # s
DECIM_FACTOR = 1  # decimation/downsampling factor
N_PEAKS = 4
PEAK_WIDTH_LIMS = (2, 8)
FREQ_BANDS = {"theta": [4, 8], "alpha": [8, 12], "beta": [15, 30]}
