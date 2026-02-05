import os
import glob
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from src.config import ITEM_CF_NPY, MODELS_DIR
from src.data_preprocessing import load_and_concat_reviews, load_products
from tqdm import tqdm

def build_item_cf_matrix():
    reviews = load_and_concat_reviews()
    for col in ["author_id", "product_id", "rating"]:
        if col not in reviews.columns:
            raise ValueError(f"Expected column {col} in reviews")
    df = reviews[["author_id", "product_id", "rating"]].dropna()
    df["author_id"] = df["author_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

    users = df["author_id"].unique()
    items = df["product_id"].unique()
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {p:i for i,p in enumerate(items)}

    rows = df["author_id"].map(user2idx)
    cols = df["product_id"].map(item2idx)
    vals = df["rating"].astype(float)

    from scipy.sparse import coo_matrix
    ui = coo_matrix((vals, (rows, cols)), shape=(len(users), len(items))).tocsr()

    item_user = ui.T.tocsr() 

    item_user_norm = normalize(item_user, axis=1)

   
    k = 50
    n_items = item_user_norm.shape[0]
    from sklearn.metrics.pairwise import cosine_similarity
    neighbors = {}
    print("Computing top-k item neighbors (collaborative)...")
    batch_size = 500
    for start in tqdm(range(0, n_items, batch_size)):
        stop = min(n_items, start+batch_size)
        block = item_user_norm[start:stop]
        sims = cosine_similarity(block, item_user_norm) 
        for i_local, row in enumerate(sims):
            i = start + i_local
            row[i] = -1.0
            idx = np.argpartition(-row, k)[:k]
            top_idx = idx[np.argsort(-row[idx])]
            top_scores = row[top_idx]
            neighbors[str(i)] = (top_idx.tolist(), top_scores.tolist())

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        "user2idx": user2idx,
        "item2idx": item2idx,
        "neighbors": neighbors,
        "items_list": list(items)
    }, os.path.join(MODELS_DIR, "item_cf_data.pkl"))
    print("Saved collaborative neighbors to models/item_cf_data.pkl")
    return True

if __name__ == "__main__":
    build_item_cf_matrix()
