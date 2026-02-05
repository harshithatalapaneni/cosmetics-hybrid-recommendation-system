import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from src.config import TFIDF_PKL, PRODUCT_TFIDF_NPY, MODELS_DIR
from src.weather_utils import get_weather_from_coords

SUBCATEGORY_MAP = {
    "moisturizers": ["moisturizer", "moisturizing", "hydration", "hydrating"],
    "treatments": ["treatment", "mask treatment", "spot treatment", "serum", "face treatment"],
    "eye care": ["eye cream", "eye serum", "eye gel", "eye treatment", "eye care"],
    "lip balms & treatments": ["lip balm", "lip treatment", "lip mask", "lip care"],
    "sunscreen": ["sunscreen", "spf lotion", "sun block", "sun protection", "spf"],
    "cleansers": ["cleanser", "face wash", "gel cleanser", "facial soap", "cleansing gel", "cleansing"],
    "value & gift sets": ["gift set", "value set", "kit", "bundle"],
    "masks": ["mask", "face mask", "sheet mask", "mask treatment"],
    "mini size": ["mini", "travel size", "sample size", "trial size"],
    "wellness": ["wellness", "supplement", "vitamin", "health"],
    "high tech tools": ["tool", "device", "high-tech tool", "beauty device"],
    "self tanners": ["self tanner", "tanning", "bronzer", "gradual tan"],
    "shop by concern": ["acne", "wrinkles", "dryness", "dark spots", "sensitive skin"],
    "lip": ["lipstick", "lip gloss", "lip liner", "lip balm", "lip color"],
    "face": ["foundation", "primer", "powder", "concealer", "bb cream", "cc cream"],
    "eye": ["eyeshadow", "eyeliner", "mascara", "eyebrow pencil", "brow gel"],
    "cheek": ["blush", "bronzer", "highlighter", "contour", "face palette"],
    "brushes & applicators": ["brush", "applicator", "sponge", "blender"],
    "makeup palettes": ["palette", "eye palette", "face palette", "cheek palette"],
    "accessories": ["mirror", "makeup bag", "cosmetic case", "tool accessory"],
    "nail": ["nail polish", "nail care", "nail treatment", "top coat", "base coat"]
}

def load_artifacts():
    tfidf = joblib.load(TFIDF_PKL)
    product_matrix = sparse.load_npz(PRODUCT_TFIDF_NPY).toarray()
    products_full = joblib.load(os.path.join(MODELS_DIR, "products_full.pkl"))
    item_cf = joblib.load(os.path.join(MODELS_DIR, "item_cf_data.pkl"))
    return tfidf, product_matrix, products_full, item_cf

def get_product_idx_map(products_full):
    ids = products_full["product_id"].astype(str).tolist()
    return {pid: idx for idx, pid in enumerate(ids)}

def recommend(user_input, top_k=10, alpha=0.6):
    tfidf, product_matrix, products_full, item_cf = load_artifacts()
    pid2idx = get_product_idx_map(products_full)

    print(f"RECOMMENDATION DEBUG START")
    print(f"User Input: skin_type={user_input.get('skin_type')}, sub_category={user_input.get('sub_category')}")
    print(f"Liked products received: {user_input.get('liked_product_ids', [])}")

    products = products_full.copy()
    for col in ["primary_category", "secondary_category", "tertiary_category", "brand_name", "product_name"]:
        products[col] = products[col].fillna("").astype(str).str.strip().str.lower()
    products["price_usd"] = pd.to_numeric(products["price_usd"], errors="coerce")
    products["review_agg"] = products.get("review_agg", "")
    products["full_text"] = products.get("full_text", "")
    
    products["detected_skin_types"] = products.get("detected_skin_types", products.apply(lambda x: [], axis=1))
    products["extracted_ingredients"] = products.get("extracted_ingredients", products.apply(lambda x: [], axis=1))
    products["extracted_concerns"] = products.get("extracted_concerns", products.apply(lambda x: [], axis=1))

    pref = user_input.get("preferences", {})
    sub_category = user_input.get("sub_category", None)
    brand = pref.get("brand_name", None)
    main_category = user_input.get("main_category", None)
    context_enabled = user_input.get("context_enabled", False)

    subcat_lc = sub_category.strip().lower() if sub_category else ""
    brand_lc = brand.strip().lower() if brand else ""
    main_cat_lc = main_category.strip().lower() if main_category else ""

    subcats_to_match = SUBCATEGORY_MAP.get(subcat_lc, [subcat_lc])

    if brand_lc:
        brand_mask = (products["brand_name"] == brand_lc)
    else:
        brand_mask = pd.Series([True] * len(products))

    if main_cat_lc:
        main_cat_mask = (products["primary_category"] == main_cat_lc)
    else:
        main_cat_mask = pd.Series([True] * len(products))

    if subcat_lc:
        subcat_keywords = SUBCATEGORY_MAP.get(subcat_lc, [subcat_lc])
        
        subcat_mask = pd.Series([False] * len(products))
        for keyword in subcat_keywords:
            mask_secondary = products["secondary_category"].str.contains(keyword, case=False, na=False)
            mask_tertiary = products["tertiary_category"].str.contains(keyword, case=False, na=False)
            mask_primary = products["primary_category"].str.contains(keyword, case=False, na=False)
            mask_product_name = products["product_name"].str.contains(keyword, case=False, na=False)
            subcat_mask = subcat_mask | mask_secondary | mask_tertiary | mask_primary | mask_product_name
    else:
        subcat_mask = pd.Series([True] * len(products))

    candidate_mask = brand_mask & main_cat_mask & subcat_mask
    candidate_indices = candidate_mask[candidate_mask].index.tolist()

    if not candidate_indices:
        print(f"Fallback 1: No candidates after brand+main+subcategory filters")
        candidate_mask = brand_mask & main_cat_mask
        candidate_indices = candidate_mask[candidate_mask].index.tolist()
        print(f"Fallback 1 candidates: {len(candidate_indices)}")

    if not candidate_indices:
        print(f"Fallback 2: No candidates after brand+main category filters")
        candidate_mask = brand_mask & subcat_mask
        candidate_indices = candidate_mask[candidate_mask].index.tolist()
        print(f"Fallback 2 candidates: {len(candidate_indices)}")

    if not candidate_indices:
        print(f"Fallback 3: No candidates after brand+subcategory filters")
        candidate_mask = brand_mask
        candidate_indices = candidate_mask[candidate_mask].index.tolist()
        print(f"Fallback 3 candidates: {len(candidate_indices)}")

    if not candidate_indices:
        print(f"Final fallback: Using all products")
        candidate_mask = pd.Series([True] * len(products))
        candidate_indices = candidate_mask[candidate_mask].index.tolist()

    print(f"Final candidate count: {len(candidate_indices)}")

    query_vec = None
    
    liked_pids = user_input.get("liked_product_ids", [])

    print(f"LIKED PRODUCTS ANALYSIS:")
    print(f"   Raw liked products received: {liked_pids}")
    print(f"   Number of liked products: {len(liked_pids)}")

    liked_pids_clean = [str(pid).strip() for pid in liked_pids if pid and str(pid).strip()]
    print(f"   Valid liked products after cleaning: {len(liked_pids_clean)}: {liked_pids_clean}")

    liked_existing = [pid for pid in liked_pids_clean if pid in pid2idx]
    liked_missing = [pid for pid in liked_pids_clean if pid not in pid2idx]

    print(f"COLLABORATIVE FILTERING ANALYSIS:")
    print(f"   Liked products received: {len(liked_pids_clean)}")
    print(f"   Liked products found in dataset: {len(liked_existing)}")
    print(f"   Liked products NOT in dataset: {len(liked_missing)}")

    if liked_missing:
        print(f"   Missing product IDs: {liked_missing}")

    if liked_existing:
        idxs = [pid2idx[pid] for pid in liked_existing if pid2idx[pid] in candidate_indices]
        
        print(f"   Liked products in candidate set: {len(idxs)}/{len(liked_existing)}")
        
        if idxs:
            query_vec = product_matrix[idxs].mean(axis=0)
            print(f"   Using {len(idxs)} liked products for query vector")
            
            print(f"   Liked products details:")
            for pid in liked_existing:
                if pid in pid2idx:
                    in_candidates = pid2idx[pid] in candidate_indices
                    product_row = products.iloc[pid2idx[pid]]
                    status = "IN CANDIDATES" if in_candidates else "NOT IN CANDIDATES"
                    print(f"      - {product_row['product_name']} | {product_row['brand_name']} | ID: {pid} | {status}")
        else:
            print(f"   Liked products found but none in candidate set")
            idxs = [pid2idx[pid] for pid in liked_existing]
            if idxs:
                query_vec = product_matrix[idxs].mean(axis=0)
                print(f"   Fallback: Using {len(idxs)} liked products (ignoring candidate filter)")
    else:
        print(f"   No liked products provided or none found in dataset")

    if user_input.get("text_query"):
        qvec = tfidf.transform([user_input["text_query"]]).toarray()
        query_vec = qvec if query_vec is None else (query_vec + qvec) / 2

    if query_vec is None:
        query_vec = product_matrix[candidate_indices].mean(axis=0)
        print(f"   Using candidate products average for query vector")

    content_scores = np.full(len(products), -np.inf)
    candidate_matrix = product_matrix[candidate_indices]
    content_scores_candidate = cosine_similarity(query_vec.reshape(1, -1), candidate_matrix).flatten()
    content_scores[candidate_indices] = content_scores_candidate

    print(f"CONTENT-BASED SCORES:")
    print(f"   Content scores range: {content_scores_candidate.min():.3f} to {content_scores_candidate.max():.3f}")
    print(f"   Mean content score: {content_scores_candidate.mean():.3f}")

    collab_scores = np.zeros_like(content_scores)
    boosts_applied = []

    cf_contributions = []
    
    if liked_existing:
        items_list = item_cf["items_list"]
        items_map = {pid: idx for idx, pid in enumerate(items_list)}
        
        print(f"COLLABORATIVE FILTERING DETAILS:")
        total_cf_matches = 0
        
        for pid in liked_existing:
            if pid in items_map:
                nbr_data = item_cf["neighbors"].get(str(items_map[pid]), None)
                if nbr_data:
                    nbr_idxs, nbr_scores = nbr_data
                    print(f"   Product {pid} has {len(nbr_idxs)} neighbors in CF model")
                    
                    product_matches = 0
                    for nbr_idx, score in zip(nbr_idxs, nbr_scores):
                        nbr_pid = items_list[nbr_idx]
                        if nbr_pid in pid2idx and pid2idx[nbr_pid] in candidate_indices:
                            collab_scores[pid2idx[nbr_pid]] += score
                            product_matches += 1
                            total_cf_matches += 1
                            
                            cf_contributions.append({
                                'liked_product': pid,
                                'recommended_product': nbr_pid,
                                'score': score,
                                'product_name': products.iloc[pid2idx[nbr_pid]]['product_name']
                            })
                    
                    print(f"      {product_matches} neighbors in candidate set")
            else:
                print(f"   Product {pid} not found in CF items map")
        
        print(f"   Total CF matches in candidate set: {total_cf_matches}")
        
        if collab_scores.sum() > 0:
            max_collab = np.max(collab_scores)
            collab_scores /= (max_collab + 1e-9)
            print(f"   Collaborative scores range: {collab_scores.min():.3f} to {collab_scores.max():.3f}")
            print(f"   Max collaborative score before normalization: {max_collab:.3f}")
            
            if cf_contributions:
                print(f"   Top CF contributions:")
                cf_contributions.sort(key=lambda x: x['score'], reverse=True)
                for contrib in cf_contributions[:5]:
                    print(f"      +{contrib['score']:.3f} for '{contrib['product_name']}' (from product {contrib['liked_product']})")
        else:
            print(f"   No collaborative filtering scores generated")
    else:
        print(f"   No liked products for collaborative filtering")

    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores

    print(f"HYBRID SCORING:")
    print(f"   Alpha (content weight): {alpha}, Collaborative weight: {1-alpha}")
    print(f"   Hybrid scores range: {hybrid_scores.min():.3f} to {hybrid_scores.max():.3f}")
    
    top_indices = np.argsort(-hybrid_scores)[:5]
    print(f"   Top 5 score breakdown:")
    for i, idx in enumerate(top_indices):
        if hybrid_scores[idx] > -np.inf:
            content_part = content_scores[idx] * alpha
            collab_part = collab_scores[idx] * (1 - alpha)
            product_name = products.iloc[idx]['product_name'][:50] + "..." if len(products.iloc[idx]['product_name']) > 50 else products.iloc[idx]['product_name']
            print(f"      {i+1}. {product_name}")
            print(f"          Total: {hybrid_scores[idx]:.3f} = Content: {content_part:.3f} + Collaborative: {collab_part:.3f}")

    if "budget_min" in pref and "budget_max" in pref:
        bmin, bmax = pref["budget_min"], pref["budget_max"]
        hybrid_scores[products["price_usd"] < bmin] = -np.inf
        hybrid_scores[products["price_usd"] > bmax] = -np.inf
        boosts_applied.append(f"Budget: {bmin}-{bmax}")

    user_skin_type = user_input.get("skin_type")
    if user_skin_type:
        skin = user_skin_type.lower()
        
        skin_mask_text = products["review_agg"].fillna("").str.lower().str.contains(skin) & candidate_mask
        
        skin_mask_detected = products["detected_skin_types"].apply(
            lambda types: skin in [t.lower() for t in types] if types else False
        ) & candidate_mask
        
        skin_mask_combined = skin_mask_text | skin_mask_detected
        
        hybrid_scores[skin_mask_combined] += 0.2
        boosts_applied.append(f"Skin type: {user_skin_type}")

    if context_enabled and user_input.get("lat") and user_input.get("lon"):
        try:
            weather = get_weather_from_coords(user_input["lat"], user_input["lon"])
            if weather and weather.get("weather"):
                w = weather["weather"].lower()
                weather_mask_candidate = candidate_mask
                if w in ["rain", "fog", "drizzle", "overcast"]:
                    hybrid_scores += (products["full_text"].str.contains("hydrating|moisturizing", case=False).astype(float) * 0.15) * weather_mask_candidate.astype(float)
                    boosts_applied.append(f"Weather: {weather['weather']} (hydrating boost)")
                elif weather.get("temp", 0) > 30:
                    hybrid_scores += (products["full_text"].str.contains("lightweight|oil-free|non-greasy", case=False).astype(float) * 0.15) * weather_mask_candidate.astype(float)
                    boosts_applied.append(f"Weather: {weather['weather']} (lightweight boost)")
        except Exception as e:
            print(f"Weather API error: {e}")

    if subcat_lc:
        subcat_relevance = pd.Series([0.0] * len(products))
        subcat_keywords = SUBCATEGORY_MAP.get(subcat_lc, [subcat_lc])
        
        for keyword in subcat_keywords:
            cat_match = (
                products["secondary_category"].str.contains(keyword, case=False, na=False) |
                products["tertiary_category"].str.contains(keyword, case=False, na=False) |
                products["primary_category"].str.contains(keyword, case=False, na=False)
            )
            name_match = products["product_name"].str.contains(keyword, case=False, na=False)
            text_match = products["full_text"].str.contains(keyword, case=False, na=False)
            
            subcat_relevance[cat_match] += 0.3
            subcat_relevance[name_match] += 0.2
            subcat_relevance[text_match] += 0.1
        
        hybrid_scores += subcat_relevance.values * candidate_mask.astype(float)
        boosts_applied.append(f"Subcategory: {sub_category}")

    final_order = np.argsort(-hybrid_scores)[:top_k]
    top = []
    for idx in final_order:
        if hybrid_scores[idx] == -np.inf:
            continue
        row = products.iloc[idx]
        top.append({
            "product_id": row["product_id"],
            "product_name": row.get("product_name", ""),
            "brand_name": row.get("brand_name", ""),
            "price_usd": row.get("price_usd", None),
            "score": float(hybrid_scores[idx]),
            "boosts": boosts_applied,
            "primary_category": row.get("primary_category", ""),
            "secondary_category": row.get("secondary_category", ""),
            "detected_skin_types": row.get("detected_skin_types", []),
            "extracted_ingredients": row.get("extracted_ingredients", [])[:3],
            "extracted_concerns": row.get("extracted_concerns", [])[:2]
        })

    print(f"FINAL RECOMMENDATIONS:")
    print(f"   Boosts applied: {boosts_applied}")
    for i, rec in enumerate(top[:5]):
        print(f"   {i+1}. {rec['product_name'][:60]}...")
        print(f"       Score: {rec['score']:.3f} | Brand: {rec['brand_name']} | Price: ${rec['price_usd']}")
    
    print(f"RECOMMENDATION DEBUG END")

    return top