# hotel_canonicalizer.py

import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import chromadb
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


class HotelCanonicalizer:
    def __init__(
        self,
        model_name: str,
        chroma_path: str,
        collection_name: str,
        sim_threshold: float = 0.72,
        sim_threshold_chroma: float = 0.55,
        n_neighbors: int = 10,
    ) -> None:
        print(f"[INIT] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        print(f"[INIT] Connecting to Chroma DB at: {chroma_path}")
        self.client = chromadb.PersistentClient(path=chroma_path)

        print(f"[INIT] Using Chroma collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

        self.sim_threshold = sim_threshold
        self.sim_threshold_chroma = sim_threshold_chroma
        self.n_neighbors = n_neighbors
        self._next_canonical_index: Optional[int] = None

    # ---------- Helpers ----------

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        print(f"[EMBED] Encoding {len(texts)} records...")
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _init_next_canonical_index_from_chroma(self) -> None:
        print("[INIT] Scanning existing canonical IDs from Chroma...")

        res = self.collection.get(include=["metadatas"], limit=1000000)
        metadatas_raw = res.get("metadatas", [])

        # Chroma `get` can return:
        # - a flat list of dicts: [md1, md2, ...]
        # - or a list of lists: [[md1, md2, ...]]
        if not metadatas_raw:
            flat_metadatas = []
        else:
            first = metadatas_raw[0]
            if isinstance(first, dict) or first is None:
                # already a flat list of dicts / None
                flat_metadatas = metadatas_raw
            else:
                # assume list-of-lists, flatten
                flat_metadatas = []
                for sub in metadatas_raw:
                    if sub is None:
                        continue
                    for md in sub:
                        flat_metadatas.append(md)

        max_num = 0
        for md in flat_metadatas:
            if not md:
                continue
            cid = md.get("canonical_hotel_id")
            if cid:
                m = re.search(r"HID_(\d+)", cid)
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num

        self._next_canonical_index = max_num + 1
        print(f"[INIT] Next canonical index starts from: {self._next_canonical_index}")


    def _generate_canonical_id(self) -> str:
        if self._next_canonical_index is None:
            self._init_next_canonical_index_from_chroma()
        cid = f"HID_{self._next_canonical_index:06d}"
        print(f"[CREATE] Generating new Canonical ID => {cid}")
        self._next_canonical_index += 1
        return cid

    # ---------- Global clustering (first-time) ----------

    def initial_cluster_and_upsert(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[GLOBAL] Starting global clustering for {len(df)} hotels...")

        df = df.reset_index(drop=True)
        ids = df["record_id"].astype(str).tolist()
        texts = df["hotel_text"].tolist()

        # compute embeddings
        embeddings = self._embed_texts(texts)

        edges = set()
        block_to_indices = df.groupby("block_key").indices

        print(f"[GLOBAL] Number of blocks = {len(block_to_indices)}")

        # find matches inside each block
        for block_key, idx_array in block_to_indices.items():
            print(f"[GLOBAL] Processing block: {block_key} | size: {len(idx_array)}")
            if len(idx_array) <= 1:
                print("   [SKIP] Only one hotel in this block")
                continue

            block_embeddings = embeddings[idx_array, :]
            block_ids = [ids[i] for i in idx_array]

            k = min(self.n_neighbors, len(block_embeddings))
            nn = NearestNeighbors(n_neighbors=k, metric="cosine")
            nn.fit(block_embeddings)
            distances, neighbors = nn.kneighbors(block_embeddings)

            block_edge_count = 0

            for i_local in range(len(block_embeddings)):
                id_i = block_ids[i_local]
                for j_local, dist in zip(neighbors[i_local], distances[i_local]):
                    if j_local == i_local:
                        continue
                    sim = 1.0 - dist
                    if sim >= self.sim_threshold:
                        id_j = block_ids[j_local]
                        edges.add(tuple(sorted((id_i, id_j))))
                        block_edge_count += 1

            print(f"   [BLOCK] Added edges: {block_edge_count}")

        print(f"[GLOBAL] Total edges created across all blocks: {len(edges)}")

        G = nx.Graph()
        G.add_nodes_from(ids)
        G.add_edges_from(edges)

        components = list(nx.connected_components(G))
        print(f"[GLOBAL] Found {len(components)} clusters")

        # assign canonical ids
        record_to_cluster: Dict[str, str] = {}
        next_idx = 1

        for comp in components:
            cid = f"HID_{next_idx:06d}"
            print(f"[GLOBAL] Creating cluster {cid} with {len(comp)} records")
            next_idx += 1
            for rid in comp:
                record_to_cluster[rid] = cid

        df["canonical_hotel_id"] = df["record_id"].astype(str).map(record_to_cluster)

        # upsert to chroma
        meta_cols = [
            "record_id", "hotel_name", "city", "country", "postal_code",
            "block_key", "canonical_hotel_id"
        ]
        meta_cols = [c for c in meta_cols if c in df.columns]
        metadatas = df[meta_cols].to_dict(orient="records")

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )

        print("[GLOBAL] Upsert complete.")
        print("[GLOBAL] Initial clustering & canonical assignment complete.")

        self._next_canonical_index = next_idx
        return df

    # ---------- Incremental matching (daily/weekly) ----------

    def incremental_match_and_upsert(self, new_df: pd.DataFrame) -> pd.DataFrame:
        print(f"[INCR] Starting incremental matching for {len(new_df)} hotels...")
        new_df = new_df.reset_index(drop=True)

        if self._next_canonical_index is None:
            self._init_next_canonical_index_from_chroma()

        if "block_key" not in new_df.columns:
            raise ValueError("new_df must contain 'block_key' column.")
        if "hotel_text" not in new_df.columns:
            raise ValueError("new_df must contain 'hotel_text' column.")

        new_ids = new_df["record_id"].astype(str).tolist()
        new_texts = new_df["hotel_text"].tolist()
        new_embeddings = self._embed_texts(new_texts)

        canonical_results = []

        for i in range(len(new_df)):
            rec_id = new_ids[i]
            blk = new_df.at[i, "block_key"]
            emb = new_embeddings[i].tolist()

            print(f"\n[INCR] Processing new record: {rec_id} | block={blk}")

            where_filter = {"block_key": blk} if blk else None

            # NOTE: no "ids" in include – Chroma always returns ids
            result = self.collection.query(
                query_embeddings=[emb],
                n_results=self.n_neighbors,
                where=where_filter,
                include=["metadatas", "distances"],  # <- fixed
            )

            best_sim = -1.0
            best_canonical = None

            ids_raw = result.get("ids", [])
            dists_raw = result.get("distances", [])
            mds_raw = result.get("metadatas", [])

            if not ids_raw:
                print("[INCR] No candidates returned from Chroma.")
            else:
                # Handle possible shapes: [ [..] ] or [..]
                nbr_ids = ids_raw[0] if isinstance(ids_raw[0], list) else ids_raw
                nbr_dists = dists_raw[0] if dists_raw and isinstance(dists_raw[0], list) else dists_raw
                nbr_mds = mds_raw[0] if mds_raw and isinstance(mds_raw[0], list) else mds_raw

                print(f"[INCR] {len(nbr_ids)} candidates found in block")

                for nbr_id, dist, md in zip(nbr_ids, nbr_dists, nbr_mds):
                    sim = 1.0 - dist
                    cid = md.get("canonical_hotel_id") if md else None
                    print(f"      -> Neighbor: {nbr_id} | sim={sim:.4f} | cid={cid}")

                    if cid and sim > best_sim:
                        best_sim = sim
                        best_canonical = cid

            if best_canonical is not None and best_sim >= self.sim_threshold_chroma:
                print(f"[MATCH] Reusing CID={best_canonical} | sim={best_sim:.4f}")
                cid = best_canonical
            else:
                cid = self._generate_canonical_id()
                print(f"[MATCH] No strong match, new CID={cid} (best_sim={best_sim:.4f})")

            canonical_results.append(cid)

        new_df["canonical_hotel_id"] = canonical_results

        meta_cols = [
            "record_id", "hotel_name", "city", "country", "postal_code",
            "block_key", "canonical_hotel_id"
        ]
        meta_cols = [c for c in meta_cols if c in new_df.columns]
        new_metadatas = new_df[meta_cols].to_dict(orient="records")
        new_docs = new_df["hotel_text"].tolist()

        self.collection.upsert(
            ids=new_ids,
            embeddings=new_embeddings.tolist(),
            metadatas=new_metadatas,
            documents=new_docs,
        )

        print(f"[INCR] Incremental matching complete. Upserted {len(new_df)} records to Chroma.")
        return new_df

