# pip install pandas sentence-transformers chromadb

import pandas as pd
import chromadb
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ===== CONFIG =====
CSV_PATH = "hoteldata_raw.csv"
LOCAL_MODEL_PATH = "sentence-transformers/all-distilroberta-v1"   # folder where model is stored
CHROMA_DB_DIR = "./chroma_hotels"
COLLECTION_NAME = "hotels_embeddings"

# ===== LOAD CSV =====
df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

# ensure id exists
if "record_id" not in df.columns:
    raise ValueError("Missing required column 'record_id'")

# ===== BUILD A SINGLE TEXT FIELD FOR EMBEDDING (Vectorized Pandas) =====
df["hotel_text"] = df[
    ["hotel_name", "address_line1", "address_line2", "city", "state", "country", "postal_code", "phone_number"]
].agg(" | ".join, axis=1).str.strip()

# Optional: remove repeated separators like " |  | "
df["hotel_text"] = df["hotel_text"].str.replace(r"\s+\|\s+", " | ", regex=True)

texts = df["hotel_text"].tolist()
ids = df["record_id"].astype(str).tolist()

# ===== LOAD LOCAL MODEL =====
model = SentenceTransformer(LOCAL_MODEL_PATH)

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# ===== INIT & STORE IN CHROMA =====
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None  # we are providing embeddings ourselves
)

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=df.to_dict(orient="records"),
)

print(f"Successfully stored {len(df)} hotel embeddings into ChromaDB collection '{COLLECTION_NAME}'")

# Make sure df index aligns with embeddings row order
df = df.reset_index(drop=True)

# create
for col in ["postal_code"]:
    if col not in df.columns:
        df[col] = ""

df["postal_norm"] = df["postal_code"].fillna("").astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]", "", regex=True)



df["block_key"] =  df["postal_norm"]

# ids aligned with df rows and embeddings
ids = df["record_id"].astype(str).tolist()
N = len(ids)


SIM_THRESHOLD = 0.72     # tune this
N_NEIGHBORS = 10         # max neighbors per point

edges = set()            # store edges as (id_i, id_j) with id_i < id_j, to avoid duplicates

# Map block_key -> list of row indices
block_to_indices = df.groupby("block_key").indices  # dict: key -> np.array of indices

for block_key, idx_array in block_to_indices.items():
    # If only one hotel in this block, no need to run NN here
    if len(idx_array) <= 1:
        continue

    # Subset embeddings and ids for this block
    block_embeddings = embeddings[idx_array, :]      # (B, D)
    block_ids = [ids[i] for i in idx_array]         # same order as block_embeddings

    # Fit nearest neighbors on this block
    k = min(N_NEIGHBORS, len(block_ids))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(block_embeddings)

    distances, neighbors = nn.kneighbors(block_embeddings)

    # For each point in this block, add edges to similar neighbors
    for i_local, (nbr_ids_local, dists_local) in enumerate(zip(neighbors, distances)):
        id_i = block_ids[i_local]
        for j_local, dist in zip(nbr_ids_local, dists_local):
            if i_local == j_local:
                continue

            sim = 1.0 - dist
            if sim >= SIM_THRESHOLD:
                id_j = block_ids[j_local]
                edge = tuple(sorted((id_i, id_j)))
                edges.add(edge)

print(f"Total edges after blocking: {len(edges)}")

# Build graph and compute connected components
G = nx.Graph()
G.add_nodes_from(ids)
G.add_edges_from(edges)

components = list(nx.connected_components(G))
print(f"Found {len(components)} clusters (connected components).")

record_to_cluster = {}

for cluster_idx, comp in enumerate(components, start=1):
    canonical_id = f"HID_{cluster_idx:06d}"
    for rec_id in comp:
        record_to_cluster[rec_id] = canonical_id




df["canonical_hotel_id"] = df["record_id"].astype(str).map(record_to_cluster)

# Optional: sanity check
print(df[["record_id", "hotel_name", "city", "country", "postal_code", "canonical_hotel_id"]].head())

# Save enriched data
df.to_csv("hotels_with_canonical_ids_blocked.csv", index=False)
print("Wrote clustered hotels to hotels_with_canonical_ids_blocked.csv")
