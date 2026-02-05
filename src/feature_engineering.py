import os
import ast
import joblib
import numpy as np
import pandas as pd
from src.config import TFIDF_PKL, PRODUCT_TFIDF_NPY, MODELS_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from src.data_preprocessing import prepare_and_merge
from tqdm import tqdm
import re
import json
import spacy
from collections import Counter

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

COSMETIC_INGREDIENTS = {
    'hyaluronic acid', 'salicylic acid', 'glycolic acid', 'lactic acid', 'niacinamide',
    'retinol', 'vitamin c', 'vitamin e', 'peptides', 'ceramides', 'aha', 'bha',
    'benzoyl peroxide', 'tea tree', 'witch hazel', 'aloe vera', 'green tea',
    'jojoba oil', 'argan oil', 'coconut oil', 'shea butter', 'glycerin',
    'squalane', 'collagen', 'spf', 'uv protection', 'zinc oxide', 'titanium dioxide',
    'caffeine', 'licorice root', 'kojic acid', 'tranexamic acid', 'azelaic acid',
    'bakuchiol', 'ferulic acid', 'mandelic acid', 'panthenol', 'allantoin',
    'centella asiatica', 'snail mucin', 'probiotics', 'prebiotics',
    'parfum', 'fragrance', 'perfume', 'eau de parfum', 'eau de toilette',
    'alcohol denat', 'essential oil', 'bergamot', 'sandalwood', 'vanilla',
    'rose', 'jasmine', 'lavender', 'citrus', 'woody', 'floral', 'fresh',
    'spicy', 'musky', 'amber', 'patchouli', 'oud', 'vetiver', 'neroli'
}

SKIN_CONCERNS = {
    'acne', 'pimples', 'breakouts', 'blackheads', 'whiteheads',
    'wrinkles', 'fine lines', 'aging', 'anti-aging',
    'dark spots', 'hyperpigmentation', 'discoloration', 'dark circles',
    'redness', 'rosacea', 'sensitivity', 'irritation', 'inflammation',
    'dryness', 'dehydration', 'flakiness', 'roughness',
    'oiliness', 'shininess', 'large pores', 'clogged pores',
    'dullness', 'uneven texture', 'uneven tone', 'scarring'
}

CHEMICAL_COMPOUNDS = {
    'alcohol denat', 'ethylhexyl methoxycinnamate', 'butyl methoxydibenzoylmethane',
    'ethylhexyl salicylate', 'benzyl salicylate', 'd-limonene', 'linalool',
    'benzyl benzoate', 'citral', 'geraniol', 'eugenol', 'benzyl alcohol',
    'farnesol', 'citronellol', 'isoeugenol', 'coumarin', 'alpha-isomethyl ionone',
    'benzyl cinnamate', 'cinnamal', 'hexyl cinnamal', 'octoxynol'
}

def extract_entities_with_spacy(text):
    if not text or pd.isna(text):
        return [], []
    
    text_sample = str(text)[:5000]
    
    try:
        doc = nlp(text_sample)
        
        ingredients = set()
        concerns = set()
        
        for ent in doc.ents:
            entity_text = ent.text.lower().strip()
            
            for ingredient in COSMETIC_INGREDIENTS:
                if ingredient in entity_text or entity_text in ingredient:
                    ingredients.add(ingredient)
            
            for concern in SKIN_CONCERNS:
                if concern in entity_text or entity_text in concern:
                    concerns.add(concern)
            
            for compound in CHEMICAL_COMPOUNDS:
                if compound in entity_text or entity_text in compound:
                    ingredients.add(compound)
        
        return list(ingredients)[:10], list(concerns)[:5]
    
    except Exception as e:
        print(f"Error in spaCy NER processing: {e}")
        return [], []

def extract_ingredients_and_concerns(text):
    if not text or pd.isna(text):
        return [], []
    
    text_lower = str(text).lower()
    
    ingredients = set()
    concerns = set()
    
    ner_ingredients, ner_concerns = extract_entities_with_spacy(text)
    ingredients.update(ner_ingredients)
    concerns.update(ner_concerns)
    
    for ingredient in COSMETIC_INGREDIENTS:
        if ingredient in text_lower:
            ingredients.add(ingredient)
    
    for compound in CHEMICAL_COMPOUNDS:
        if compound in text_lower:
            ingredients.add(compound)
    
    for concern in SKIN_CONCERNS:
        if concern in text_lower:
            concerns.add(concern)
    
    ingredient_patterns = [
        r'\b(?:vitamin\s+[a-c-e])\b',
        r'\b(?:hyaluronic|salicylic|glycolic|lactic|azelaic|mandelic|tranexamic|ferulic)\s+acid\b',
        r'\b(?:jojoba|argan|coconut|rosehip|marula|bergamot|sandalwood)\s+(?:oil|extract)\b',
        r'\b(?:alcohol\s+denat)\b',
        r'\b(?:eau\s+de\s+parfum|eau\s+de\s+toilette)\b',
        r'\b(?:benzyl|ethylhexyl|butyl)\s+\w+\b',
    ]
    
    for pattern in ingredient_patterns:
        try:
            matches = re.findall(pattern, text_lower)
            ingredients.update(matches)
        except:
            pass
    
    return list(ingredients)[:15], list(concerns)[:5]

def parse_ingredients_list(ingredients_str):
    if not ingredients_str or pd.isna(ingredients_str):
        return []
    
    try:
        if isinstance(ingredients_str, str) and ingredients_str.startswith('['):
            ingredients_list = ast.literal_eval(ingredients_str)
            if isinstance(ingredients_list, list):
                return ingredients_list
        elif isinstance(ingredients_str, str):
            return [ingredients_str]
    except:
        pass
    
    return []

def extract_ingredients_from_structured(ingredients_list):
    meaningful_ingredients = set()
    
    for ingredient in ingredients_list:
        ingredient_lower = str(ingredient).lower()
        
        for known_ingredient in COSMETIC_INGREDIENTS.union(CHEMICAL_COMPOUNDS):
            if known_ingredient in ingredient_lower:
                meaningful_ingredients.add(known_ingredient)
        
        chemical_patterns = [
            r'\b(?:retinol|niacinamide|peptide|ceramide|spf|parfum|fragrance|alcohol denat)\b',
            r'\b(?:hyaluronic|salicylic|glycolic|lactic)\s+acid\b',
            r'\bvitamin\s+[a-c-e]\b',
            r'\b(?:ethylhexyl|butyl|benzyl)\s+\w+\b',
            r'\b(?:limonene|linalool|geraniol|eugenol|coumarin)\b',
        ]
        
        for pattern in chemical_patterns:
            try:
                matches = re.findall(pattern, ingredient_lower)
                meaningful_ingredients.update(matches)
            except:
                pass
    
    return list(meaningful_ingredients)

def detect_skin_type_from_reviews(review_text, skin_type_column=None):
    detected_types = []
    
    if skin_type_column and pd.notna(skin_type_column) and skin_type_column.strip():
        detected_types.append(skin_type_column.strip().lower())
    
    if review_text and pd.notna(review_text):
        text_lower = str(review_text).lower()
        skin_type_keywords = {
            'oily': ['oily', 'greasy', 'shiny', 'oil control', 'sebum', 'shine'],
            'dry': ['dry', 'flaky', 'dehydrated', 'moisture', 'hydration', 'parched'],
            'combination': ['combination', 't-zone', 'oily t-zone', 'combo', 'mixed'],
            'normal': ['normal', 'balanced', 'neither oily nor dry', 'well-balanced'],
            'sensitive': ['sensitive', 'irritated', 'redness', 'reactive', 'stinging']
        }
        
        for skin_type, keywords in skin_type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if skin_type not in detected_types:
                    detected_types.append(skin_type)
    
    return list(set(detected_types))[:2]

def join_product_text(row):
    parts = []
    for field in ["product_name", "highlights", "ingredients", "primary_category", "secondary_category"]:
        if field in row and pd.notna(row[field]):
            v = row[field]
            if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, (list, tuple)):
                        parts.append(" ".join([str(x) for x in parsed]))
                    else:
                        parts.append(str(parsed))
                except Exception:
                    parts.append(v)
            else:
                parts.append(str(v))
    return " ".join(parts)

def create_product_corpus(products_df, reviews_df):
    print("Aggregating reviews per product...")
    reviews_group = reviews_df.groupby("product_id")["review_text"].apply(lambda texts: " ".join(texts.dropna().astype(str).values[:200]))
    
    print("Detecting skin types from reviews...")
    
    def detect_skin_types_group(texts_skin):
        all_detected = []
        for idx, (review_text, skin_type) in enumerate(texts_skin):
            if pd.notna(review_text) or pd.notna(skin_type):
                detected = detect_skin_type_from_reviews(review_text, skin_type)
                all_detected.extend(detected)
        if all_detected:
            counter = Counter(all_detected)
            return [skin_type for skin_type, count in counter.most_common(3)]
        return []
    
    reviews_with_skin = reviews_df.groupby("product_id").apply(
        lambda group: list(zip(group["review_text"].fillna(""), group["skin_type"].fillna("")))
    )
    skin_type_group = reviews_with_skin.apply(detect_skin_types_group)
    
    products_df = products_df.set_index("product_id")
    products_df["product_text"] = products_df.apply(join_product_text, axis=1)
    products_df["review_agg"] = reviews_group
    products_df["review_agg"] = products_df["review_agg"].fillna("")
    
    products_df["detected_skin_types"] = skin_type_group
    products_df["detected_skin_types"] = products_df["detected_skin_types"].apply(lambda x: x if isinstance(x, list) else [])
    
    print("Extracting ingredients from structured data...")
    products_df["structured_ingredients"] = products_df["ingredients"].apply(parse_ingredients_list)
    products_df["extracted_ingredients_structured"] = products_df["structured_ingredients"].apply(extract_ingredients_from_structured)
    
    print("Extracting ingredients and skin concerns using NER and enhanced matching...")
    tqdm.pandas(desc="NER + Keyword Extraction")
    ingredients_concerns = products_df["product_text"].progress_apply(
        lambda x: extract_ingredients_and_concerns(x)
    )
    products_df["extracted_ingredients_ner"] = ingredients_concerns.apply(lambda x: x[0])
    products_df["extracted_concerns"] = ingredients_concerns.apply(lambda x: x[1])
    
    products_df["extracted_ingredients"] = products_df.apply(
        lambda row: list(set(row["extracted_ingredients_structured"] + row["extracted_ingredients_ner"])),
        axis=1
    )
    
    products_df["full_text"] = (
        products_df["product_text"].fillna("") + " " + 
        products_df["review_agg"].fillna("") + " " +
        products_df["extracted_ingredients"].apply(lambda x: " ".join(x) if x else "") + " " +
        products_df["extracted_concerns"].apply(lambda x: " ".join(x) if x else "") + " " +
        products_df["detected_skin_types"].apply(lambda x: " ".join(x) if x else "")
    ).str.replace(r"\s+", " ", regex=True)
    
    products_df = products_df.reset_index()
    return products_df

def build_tfidf(products_df, max_features=50000):
    print("Training TF-IDF...")
    corpus = products_df["full_text"].astype(str).values
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), min_df=2, stop_words="english")
    X = tfidf.fit_transform(corpus)
    Xn = normalize(X, norm="l2", axis=1)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tfidf, TFIDF_PKL)
    from scipy import sparse
    sparse.save_npz(PRODUCT_TFIDF_NPY, Xn)
    print(f"Saved TF-IDF to {TFIDF_PKL} and product matrix to {PRODUCT_TFIDF_NPY}")
    return tfidf, Xn

def run_feature_pipeline():
    products, reviews = prepare_and_merge()
    products_df = create_product_corpus(products, reviews)
    tfidf, product_matrix = build_tfidf(products_df)
    
    enhanced_products = products_df[[
        "product_id", "product_name", "brand_name", "price_usd", "primary_category",
        "secondary_category", "detected_skin_types", "extracted_ingredients", "extracted_concerns", "structured_ingredients"
    ]].copy()
    
    joblib.dump(enhanced_products, os.path.join(MODELS_DIR, "product_meta.pkl"))
    joblib.dump(products_df, os.path.join(MODELS_DIR, "products_full.pkl"))
    
    print("\n=== Enhanced NER Extraction Results ===")
    
    skincare_products = products_df[~products_df["primary_category"].str.contains("fragrance", case=False, na=False)].head(3)
    fragrance_products = products_df[products_df["primary_category"].str.contains("fragrance", case=False, na=False)].head(2)
    
    print("\n--- SKINCARE PRODUCTS ---")
    for idx, product in skincare_products.iterrows():
        print(f"\nProduct: {product['product_name']}")
        print(f"Category: {product['primary_category']} -> {product['secondary_category']}")
        print(f"Structured Ingredients sample: {str(product['structured_ingredients'])[:100]}...")
        print(f"Extracted Ingredients: {product['extracted_ingredients'][:8]}")
        print(f"Extracted Concerns: {product['extracted_concerns'][:3]}")
        print(f"Detected Skin Types: {product['detected_skin_types']}")
    
    print("\n--- FRAGRANCE PRODUCTS ---")
    for idx, product in fragrance_products.iterrows():
        print(f"\nProduct: {product['product_name']}")
        print(f"Category: {product['primary_category']} -> {product['secondary_category']}")
        print(f"Extracted Ingredients: {product['extracted_ingredients'][:8]}")
        print(f"Detected Skin Types: {product['detected_skin_types']}")
    
    total_ner_ingredients = products_df['extracted_ingredients_ner'].apply(len).sum()
    total_keyword_ingredients = products_df['extracted_ingredients_structured'].apply(len).sum()
    
    print(f"\n=== NER EXTRACTION STATISTICS ===")
    print(f"Total products processed: {len(products_df)}")
    print(f"Products with extracted ingredients: {len(products_df[products_df['extracted_ingredients'].str.len() > 0])}")
    print(f"Products with detected skin types: {len(products_df[products_df['detected_skin_types'].str.len() > 0])}")
    print(f"Ingredients found via NER: {total_ner_ingredients}")
    print(f"Ingredients found via structured data: {total_keyword_ingredients}")
    print(f"Average ingredients per product: {products_df['extracted_ingredients'].apply(len).mean():.2f}")

if __name__ == "__main__":
    run_feature_pipeline()