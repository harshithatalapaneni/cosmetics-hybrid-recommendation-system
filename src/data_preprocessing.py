import os
import glob
import pandas as pd
import numpy as np
from src.config import PRODUCTS_FILE, REVIEWS_GLOB, MODELS_DIR
from tqdm import tqdm
import joblib

def load_products(products_file=PRODUCTS_FILE):
    print(f"Loading products from {products_file}")
    df = pd.read_csv(products_file, dtype=str)
    for col in ["price_usd", "sale_price_usd", "rating", "loves_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_and_concat_reviews(glob_pattern=REVIEWS_GLOB):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No review files found with pattern {glob_pattern}")
    dfs = []
    for f in tqdm(files, desc="Loading review files"):
        dfs.append(pd.read_csv(f, dtype=str, low_memory=False))
    reviews = pd.concat(dfs, ignore_index=True)
    if "rating" in reviews.columns:
        reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    return reviews

def basic_clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.strip()

def prepare_and_merge(products_file=PRODUCTS_FILE, reviews_glob=REVIEWS_GLOB):
    products = load_products(products_file)
    reviews = load_and_concat_reviews(reviews_glob)

    text_cols = []
    if "highlights" in products.columns:
        text_cols.append("highlights")
    if "ingredients" in products.columns:
        text_cols.append("ingredients")
    if "product_name" in products.columns:
        text_cols.append("product_name")

    for c in text_cols:
        products[c] = products[c].apply(basic_clean_text)

    if "review_text" in reviews.columns:
        reviews["review_text"] = reviews["review_text"].apply(basic_clean_text)
    else:
        reviews["review_text"] = ""

    products["product_id"] = products["product_id"].astype(str)
    reviews["product_id"] = reviews["product_id"].astype(str)

    os.makedirs(MODELS_DIR, exist_ok=True)
    meta = products[["product_id", "product_name", "brand_name", "price_usd", "primary_category"]].copy()
    joblib.dump(meta, os.path.join(MODELS_DIR, "product_meta.pkl"))
    print("Saved product meta.")

    return products, reviews

if __name__ == "__main__":
    print("Running data_preprocessing.py as script")
    prepare_and_merge()
