# config.py

CSV_PATH_INITIAL = "hoteldata_raw_first.csv"
CSV_PATH_INCREMENTAL = "hoteldata_raw_new.csv"

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"

CHROMA_DB_DIR = "./chroma_hotels"
CHROMA_COLLECTION_NAME = "hotels_embeddings"

SIM_THRESHOLD = 0.72
SIM_THRESHOLD_CHROMA = 0.55
N_NEIGHBORS = 10
