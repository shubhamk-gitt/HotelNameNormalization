# HotelNameNormalization

Deduplication & Canonicalization of hotels data from multiple suppliers here we are using NLP, snetence transformer embeddings & incremental clustering with ChromaDB

The goal is to normalize, cluster, and assign a stable canonical_hotel_id so that downstream systems always reference a single hotel record.

## Key features 
Embedding-based similarity : Sentence-transformers (all-distilroberta-v1 model) used to convert hotel text to vectors

Blocks based matching: Use Postal code to reduces false cross country or city matches and speeds up the nearest neighbor search.

Graph based clustering : Cosine similarity edges and connected components used for initial grouping of the hotels

Chroma Vector DB : Stores embeddings, metadata & canonical_hotel_id and does  scalable incremental matching with vector search.

Incremental matching pipeline : Scheduled pipeline that run for new hotels matches to existing clusters without run again the global clustering.

Airflow orchestration : pipelines for intial and incremental jobs 

## How it works

Intital Global clustering (Can be ran once or montly)

- Embed all hotel records
- group records by block_key in our case it is normalized and cleaned postal code , can be also combination of fields if required for better results.
- Apply KNN within each block , draw edage between two if similarity >= threshold value
- Give one unique canonical_hotel_id to each Connected component
- Store embeddings + metadata + canonical ID in Chroma DB and (In relational Database or csv file)

Incremental Updates (Can be a nigltly scheduled job):
- Only process new records and embed the new hotels data
- Query the block_key with ChromaDB block_key meta_data
- If match in block_key found query the nearest neighbors with same block key
- If high similarity then reuse same canonical ID else create a new canonical ID
- insert back in to ChromaDB

## Airflow dags
We can use the airflow dags to run the Initial clustering job and the incremental job

These two jobs can be triggered with airflow scheduler or airflow ui

commands:
airflow dags trigger hotel_initial_canonicalization

airflow dags trigger hotel_incremental_canonicalization


we can place the dag files in the airlfow/dags directory

and initalize airflow and run in local host.