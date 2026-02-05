import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "archive")
MODELS_DIR = os.path.join(BASE_DIR, "models")

PRODUCTS_FILE = os.path.join(DATA_DIR, "product_info.csv")
REVIEWS_GLOB = os.path.join(DATA_DIR, "reviews_*.csv")

TFIDF_PKL = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
PRODUCT_TFIDF_NPY = os.path.join(MODELS_DIR, "product_tfidf_matrix.npz")
ITEM_CF_NPY = os.path.join(MODELS_DIR, "item_cf_matrix.npz")
PRODUCT_META_PKL = os.path.join(MODELS_DIR, "product_meta.pkl")
