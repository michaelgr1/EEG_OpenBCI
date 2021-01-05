from brainflow.board_shim import BoardIds, BoardShim

APP_STYLE = "Fusion"

BOARD_ID = BoardIds.SYNTHETIC_BOARD.value
# BOARD_ID = BoardIds.CYTON_BOARD.value

SAMPLING_RATE = BoardShim.get_sampling_rate(BOARD_ID)

IMAGES_SSD_DRIVER_LETTER = "F"

EEG_DATA_FILE_NAME = "eeg_data.csv"

SLICE_INDEX_FILE_NAME = "slice_index.txt"

RESONANCE_REFERENCE_FILE_NAME = "reference.csv"

RESONANCE_DATA_FILE_NAME = "eeg_data.csv"

if BOARD_ID == BoardIds.SYNTHETIC_BOARD.value:
    print("*" * 40)
    print("Warning: Using Synthetic Board...!")
    print("*" * 40)
