from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.hybrid_recommender import recommend
from src.explainability import explain
from src.config import PRODUCTS_FILE
import csv
from datetime import datetime
import os
from src.weather_utils import get_weather_from_coords

app = Flask(__name__)
USER_INPUTS_FILE = "user_inputs.csv"

products_df = pd.read_csv(PRODUCTS_FILE)

products_df["primary_category"] = products_df["primary_category"].fillna("").astype(str).str.lower()
products_df["secondary_category"] = products_df["secondary_category"].fillna("").astype(str).str.lower()
products_df["tertiary_category"] = products_df["tertiary_category"].fillna("").astype(str).str.lower()
products_df["brand_name"] = products_df["brand_name"].fillna("").astype(str).str.strip()

skincare_keywords = ["skin", "moisturizer", "cleanser", "hydrating", "serum", "face care", "skincare"]
makeup_keywords = ["makeup", "foundation", "blush", "concealer", "mascara", "lipstick", "eye makeup", "make up"]

def category_matches_keywords(category, keywords):
    category = category.lower()
    return any(keyword in category for keyword in keywords)

main_categories = []

unique_primary_categories = products_df["primary_category"].unique()

for cat in unique_primary_categories:
    if category_matches_keywords(cat, skincare_keywords):
        main_categories.append(cat)
    elif category_matches_keywords(cat, makeup_keywords):
        main_categories.append(cat)

main_categories = list(set(main_categories))

subcategories_map = {}
for main_cat in main_categories:
    subs = products_df.loc[products_df["primary_category"] == main_cat, "secondary_category"].unique().tolist()
    subs = [s for s in subs if s]
    subcategories_map[main_cat] = subs

brands_all = products_df["brand_name"].unique().tolist()


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/skin_type", methods=["GET", "POST"])
def skin_type():
    return render_template("skin_type.html", skin_types=["combination", "oily", "dry", "normal"])


@app.route("/skin_test")
def skin_test():
    return render_template("skin_test.html")


@app.route("/budget", methods=["POST"])
def budget():
    skin_type = request.form.get("skin_type")
    return render_template("budget.html", skin_type=skin_type, budgets=["0-10", "10-30", "30-50", "50-100", "100+"])


@app.route("/product_type", methods=["POST"])
def product_type():
    skin_type = request.form.get("skin_type")
    budget = request.form.get("budget")
    return render_template("product_type.html",
                           skin_type=skin_type,
                           budget=budget,
                           categories=main_categories)


@app.route("/sub_category", methods=["POST"])
def sub_category():
    skin_type = request.form.get("skin_type")
    budget = request.form.get("budget")
    main_category = request.form.get("main_category")
    subcategories = subcategories_map.get(main_category, [])
    return render_template("sub_category.html",
                           skin_type=skin_type,
                           budget=budget,
                           main_category=main_category,
                           subcategories=subcategories)


@app.route("/liked_products", methods=["POST"])
def liked_products():
    skin_type = request.form.get("skin_type")
    budget = request.form.get("budget")
    main_category = request.form.get("main_category")
    sub_category = request.form.get("sub_category")

    print(f"DEBUG - Liked products route:")
    print(f"   main_category: {main_category}")
    print(f"   sub_category: {sub_category}")

    if sub_category and sub_category.strip():
        filtered_products = products_df[
            (products_df["primary_category"] == sub_category.lower()) |
            (products_df["secondary_category"] == sub_category.lower()) |
            (products_df["tertiary_category"] == sub_category.lower()) |
            (products_df["primary_category"].str.contains(sub_category.lower(), case=False, na=False)) |
            (products_df["secondary_category"].str.contains(sub_category.lower(), case=False, na=False))
        ]
    else:
        filtered_products = products_df[products_df["primary_category"] == main_category.lower()]

    print(f"   Found {len(filtered_products)} products after filtering")

    liked_products_options = filtered_products[["product_id", "product_name", "brand_name"]].drop_duplicates().head(50)
    
    products_list = []
    for _, row in liked_products_options.iterrows():
        products_list.append({
            "product_id": str(row["product_id"]),
            "product_name": f"{row['product_name']} - {row['brand_name']}"
        })

    print(f"   Sending {len(products_list)} products to template")

    return render_template("liked_products.html",
                           skin_type=skin_type,
                           budget=budget,
                           main_category=main_category,
                           sub_category=sub_category,
                           products=products_list)


@app.route("/brand", methods=["POST"])
def brand():
    skin_type = request.form.get("skin_type")
    budget = request.form.get("budget")
    main_category = request.form.get("main_category")
    sub_category = request.form.get("sub_category")
    
    liked_products = request.form.getlist("liked_products")
    
    print(f"DEBUG - Brand route:")
    print(f"   Received {len(liked_products)} liked products: {liked_products}")
    
    if not liked_products:
        liked_products = [v for k, v in request.form.items() if k.startswith('liked_products')]
    
    print(f"DEBUG - Brand route received:")
    print(f"   skin_type: {skin_type}")
    print(f"   budget: {budget}")
    print(f"   main_category: {main_category}")
    print(f"   sub_category: {sub_category}")
    print(f"   liked_products: {liked_products}")
    print(f"   Number of liked products: {len(liked_products)}")
    
    if liked_products:
        print(f"   First liked product ID: {liked_products[0]}")
        existing_products = products_df[products_df['product_id'].astype(str).isin(liked_products)]
        print(f"   Found {len(existing_products)} of {len(liked_products)} liked products in dataset")

    if sub_category:
        df_filtered = products_df[
            (products_df["primary_category"] == main_category) &
            ((products_df["secondary_category"] == sub_category) | 
             (products_df["tertiary_category"] == sub_category))
        ]
    else:
        df_filtered = products_df[products_df["primary_category"] == main_category]

    brands = df_filtered["brand_name"].dropna().unique().tolist()

    if not brands:
        brands = brands_all

    return render_template("brand.html",
                           skin_type=skin_type,
                           budget=budget,
                           main_category=main_category,
                           sub_category=sub_category,
                           brands=brands,
                           liked_products=liked_products)

@app.route("/context", methods=["POST"])
def context():
    skin_type = request.form.get("skin_type")
    budget = request.form.get("budget")
    main_category = request.form.get("main_category")
    sub_category = request.form.get("sub_category")
    brand = request.form.get("brand")
    
    liked_products = request.form.getlist("liked_products")
    
    print(f"DEBUG - Context route:")
    print(f"   Received {len(liked_products)} liked products: {liked_products}")
    
    if not liked_products:
        liked_products = [v for k, v in request.form.items() if k.startswith('liked_products')]
    
    print(f"DEBUG - Context route received liked_products: {liked_products}")
    
    return render_template("context.html",
                           skin_type=skin_type,
                           budget=budget,
                           main_category=main_category,
                           sub_category=sub_category,
                           brand=brand,
                           liked_products=liked_products)


@app.route('/get-weather-data', methods=['POST'])
def get_weather_data():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        
        print(f"Getting weather data for coordinates: {lat}, {lon}")
        
        if lat and lon:
            weather_data = get_weather_from_coords(lat, lon)
            print(f"Weather data received: {weather_data}")
            
            return jsonify({
                'weather': weather_data.get('weather'),
                'temperature': weather_data.get('temperature'),
                'success': True
            })
        else:
            return jsonify({'error': 'Missing coordinates', 'success': False}), 400
            
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return jsonify({'error': 'Could not fetch weather data', 'success': False}), 500


@app.route("/recommendation", methods=["POST"])
def recommendation():
    skin_type = request.form.get("skin_type")
    budget = request.form.get("budget")
    main_category = request.form.get("main_category")
    sub_category = request.form.get("sub_category")
    brand = request.form.get("brand")
    context_enabled = request.form.get("context") == "yes"
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    
    liked_products = request.form.getlist("liked_products")
    
    print(f"DEBUG - Recommendation route FORM DATA:")
    all_form_keys = list(request.form.keys())
    print(f"   All form keys: {all_form_keys}")
    
    liked_products_params = []
    for key, value in request.form.items():
        if 'liked' in key.lower():
            print(f"   Found liked product param: '{key}' = '{value}'")
            if key == 'liked_products':
                liked_products_params.append(value)
    
    print(f"   liked_products from getlist: {liked_products} (count: {len(liked_products)})")
    print(f"   liked_products from manual scan: {liked_products_params} (count: {len(liked_products_params)})")
    
    if len(liked_products_params) > len(liked_products):
        liked_products = liked_products_params
        print(f"   Using manual scan results: {len(liked_products)} products")
    
    
    if not liked_products:
        liked_products = [v for k, v in request.form.items() if k.startswith('liked_products')]

    print(f"DEBUG - Recommendation route:")
    print(f"   liked_products received: {liked_products}")
    print(f"   Number of liked products: {len(liked_products)}")


    pref = {}
    if budget:
        try:
            if budget == "100+":
                bmin, bmax = 100.0, 10000.0
            else:
                bmin, bmax = map(float, budget.split("-"))
            pref["budget_min"] = bmin
            pref["budget_max"] = bmax
        except Exception as e:
            print(f"Budget parsing error: {e}")
            pref["budget_min"] = 0.0
            pref["budget_max"] = 1000.0

    if brand:
        pref["brand_name"] = brand
    
    

    query = {
        "skin_type": skin_type,
        "preferences": pref,
        "liked_product_ids": liked_products,
        "text_query": "",
        "lat": lat,
        "lon": lon,
        "main_category": main_category,
        "sub_category": sub_category,
        "context_enabled": context_enabled
    }

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skin_type": skin_type,
        "budget": budget,
        "main_category": main_category,
        "sub_category": sub_category,
        "brand": brand,
        "context_enabled": context_enabled,
        "lat": lat,
        "lon": lon,
        "liked_products": ", ".join(liked_products) if liked_products else "None"
    }

    file_exists = os.path.isfile(USER_INPUTS_FILE)
    with open(USER_INPUTS_FILE, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"FINAL USER QUERY:")
    print(f"   skin_type={skin_type}, budget={budget}, main_category={main_category}")
    print(f"   sub_category={sub_category}, brand={brand}, context_enabled={context_enabled}")
    print(f"   liked_products={len(liked_products)} products: {liked_products}")

    context_input = None
    if context_enabled and lat and lon:
        try:
            print(f"Fetching weather data for lat={lat}, lon={lon}")
            weather_data = get_weather_from_coords(float(lat), float(lon))
            print(f"Raw weather data: {weather_data}")
            
            if weather_data.get("weather") and weather_data.get("temperature"):
                context_input = {
                    "weather": weather_data.get("weather"),
                    "temperature": str(weather_data.get("temperature"))
                }
                print(f"Context input created: {context_input}")
            else:
                print(f"Incomplete weather data: {weather_data}")
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            context_input = None
    else:
        print(f"Context not enabled or no location: context_enabled={context_enabled}, lat={lat}, lon={lon}")

    recs = recommend(query, top_k=10)
    
    explanations = {}
    user_input_for_explain = {
        "skin_type": skin_type,
        "sub_category": sub_category,
        "brand": brand,
        "budget": budget
    }
    
    for product in recs:
        product_id = product.get("product_id")
        if product_id:
            explanations[product_id] = explain(
                product_id=product_id,
                user_input=user_input_for_explain,
                context_input=context_input
            )
            print(f"Explanation for {product_id}: {len(explanations[product_id].get('boosts', []))} boosts")

    return render_template("recommendation.html", 
                         recommendations=recs,
                         explanations=explanations,
                         skin_type=skin_type,
                         context_enabled=context_enabled,
                         lat=lat,
                         lon=lon,
                         sub_category=sub_category,
                         brand=brand,
                         budget=budget,
                         liked_products_count=len(liked_products))


@app.route("/api/explain", methods=["POST"])
def api_explain():
    payload = request.json
    product_id = payload.get("product_id")
    
    user_input = {
        "skin_type": payload.get("skin_type", ""),
        "context_enabled": payload.get("context_enabled", False),
        "lat": payload.get("lat", ""),
        "lon": payload.get("lon", ""),
        "sub_category": payload.get("sub_category", ""),
        "brand": payload.get("brand", ""),
        "budget": payload.get("budget", "")
    }
    
    res = explain(product_id, user_input)
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True, port=5001)