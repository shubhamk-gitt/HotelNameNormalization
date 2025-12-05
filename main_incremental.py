# main_incremental.py

from config import (
    CSV_PATH_INCREMENTAL,
    MODEL_NAME,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    SIM_THRESHOLD,
    N_NEIGHBORS,
)
from preprocessing import load_and_clean_csv, build_hotel_text, add_block_key
from hotel_canonicalizer import HotelCanonicalizer


def main() -> None:
    new_df = load_and_clean_csv(CSV_PATH_INCREMENTAL)
    new_df = build_hotel_text(new_df)
    new_df = add_block_key(new_df)

    canonicalizer = HotelCanonicalizer(
        model_name=MODEL_NAME,
        chroma_path=CHROMA_DB_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        sim_threshold=SIM_THRESHOLD,
        n_neighbors=N_NEIGHBORS,
    )

    new_df_with_ids = canonicalizer.incremental_match_and_upsert(new_df)
    new_df_with_ids.to_csv("hotels_with_canonical_ids_incremental.csv", index=False)
    print("Incremental matching complete and stored in Chroma.")


if __name__ == "__main__":
    main()
