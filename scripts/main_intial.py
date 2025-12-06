# main_initial.py
from config import (
    CSV_PATH_INITIAL,
    MODEL_NAME,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    SIM_THRESHOLD,
    N_NEIGHBORS,
)
from preprocessing import load_and_clean_csv, build_hotel_text, add_block_key
from hotel_canonicalizer import HotelCanonicalizer


def main() -> None:
    df = load_and_clean_csv(CSV_PATH_INITIAL)
    df = build_hotel_text(df)
    df = add_block_key(df)

    canonicalizer = HotelCanonicalizer(
        model_name=MODEL_NAME,
        chroma_path=CHROMA_DB_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        sim_threshold=SIM_THRESHOLD,
        n_neighbors=N_NEIGHBORS,
    )

    df_with_ids = canonicalizer.initial_cluster_and_upsert(df)
    df_with_ids.to_csv("hotels_with_canonical_ids_initial.csv", index=False)
    print("Initial clustering complete and stored in Chroma.")


if __name__ == "__main__":
    main()
