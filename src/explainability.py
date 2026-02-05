
import pandas as pd
import numpy as np
from src.hybrid_recommender import load_artifacts, get_product_idx_map
from src.weather_utils import get_weather_from_coords


def explain(product_id, user_input=None, context_input=None):
    print(f"Starting explanation for product_id: {product_id}")
    
    try:
        tfidf, product_matrix, products_full, item_cf = load_artifacts()
        pid2idx = get_product_idx_map(products_full)
        print(f"Artifacts loaded successfully")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return {
            "explanation": "We couldn't load the recommendation system data.",
            "keywords": [],
            "sample_positive_review": "",
            "boosts": [],
            "categories": [],
            "ingredients": [],
            "skin_concerns": [],
            "detected_skin_types": []
        }

    try:
        product = products_full[products_full["product_id"] == product_id].iloc[0]
        print(f"Found product: {product.get('product_name', 'Unknown')}")
    except (IndexError, KeyError) as e:
        print(f"Product not found: {e}")
        return {
            "explanation": "We couldn't find detailed information about this product.",
            "keywords": [],
            "sample_positive_review": "",
            "boosts": [],
            "categories": [],
            "ingredients": [],
            "skin_concerns": [],
            "detected_skin_types": []
        }

    try:
        product_name = product.get("product_name", "This product")
        product_brand = product.get("brand_name", "")
        price_usd = product.get("price_usd")
        primary_category = product.get("primary_category", "").lower()
        secondary_category = product.get("secondary_category", "").lower()
        review_text = product.get("review_agg", "")
        full_text = product.get("full_text", "")
        
        print(f"Basic fields extracted")
    except Exception as e:
        print(f"Error extracting basic fields: {e}")
        return {
            "explanation": "Error reading product information.",
            "keywords": [],
            "sample_positive_review": "",
            "boosts": [],
            "categories": [],
            "ingredients": [],
            "skin_concerns": [],
            "detected_skin_types": []
        }

    keywords = []

    sample_positive_review = ""
    try:
        if review_text and isinstance(review_text, str) and len(review_text.strip()) > 0:
            positive_keywords = [
                'love', 'great', 'amazing', 'excellent', 'awesome', 'fantastic', 'wonderful',
                'perfect', 'best', 'good', 'nice', 'recommend', 'happy', 'satisfied',
                'works well', 'effective', 'results', 'improved', 'better', 'soft',
                'smooth', 'hydrated', 'glowing', 'radiant', 'clear', 'beautiful',
                'worth it', 'repurchase', 'favorite', 'holy grail', 'game changer',
                'works', 'help', 'difference', 'improvement', 'quality', 'pleased', 'impressed',
                'results', 'improvement', 'effective', 'wonderful', 'amazing results'
            ]
            
            negative_keywords = [
                'not', "don't", "doesn't", "didn't", 'waste', 'disappointed',
                'terrible', 'awful', 'horrible', 'bad', 'worst', 'return',
                'break out', 'irritated', 'allergic', 'rash', 'dry', 'oily',
                'sticky', 'greasy', 'smell', 'chemical', 'expensive', 'pricey',
                'shame', 'suffering', 'cystic', 'acne', 'no improvement', 'never seen results',
                'throwing away', 'waste of money'
            ]
            
            sentences = review_text.split('.')
            positive_sentences = []
            
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:
                    sentence_lower = clean_sentence.lower()
                    
                    has_positive = any(keyword in sentence_lower for keyword in positive_keywords)
                    has_negative = any(keyword in sentence_lower for keyword in negative_keywords)
                    
                    if has_positive and not has_negative:
                        positive_sentences.append(clean_sentence)
            
            if positive_sentences:
                positive_sentences.sort(key=len, reverse=True)
                sample_positive_review = positive_sentences[0] + "."
                print(f"Positive review extracted: {sample_positive_review[:60]}...")
            else:
                for sentence in sentences:
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 20 and not clean_sentence.lower().startswith(("this product", "i received", "i bought")):
                        sentence_lower = clean_sentence.lower()
                        has_negative = any(keyword in sentence_lower for keyword in negative_keywords)
                        if not has_negative:
                            sample_positive_review = clean_sentence + "."
                            break
                
                if not sample_positive_review and sentences:
                    first_sentence = sentences[0].strip()
                    if len(first_sentence) > 10:
                        first_lower = first_sentence.lower()
                        has_negative = any(keyword in first_lower for keyword in negative_keywords)
                        if not has_negative:
                            sample_positive_review = first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence + "."
                            print(f"Using first sentence as review: {sample_positive_review[:60]}...")
                        else:
                            print("First sentence contains negative content, skipping")
                    else:
                        print("No suitable review text found")
        else:
            print("No review text available for this product")
            
    except Exception as e:
        print(f"Error extracting review: {e}")
        sample_positive_review = ""

    try:
        extracted_ingredients = product.get("extracted_ingredients", [])
        if not isinstance(extracted_ingredients, list):
            extracted_ingredients = []
        
        extracted_concerns = product.get("extracted_concerns", [])
        if not isinstance(extracted_concerns, list):
            extracted_concerns = []
        
        detected_skin_types = product.get("detected_skin_types", [])
        if not isinstance(detected_skin_types, list):
            detected_skin_types = []
        
        print(f"NER features - Ingredients: {len(extracted_ingredients)}, Concerns: {len(extracted_concerns)}, Skin Types: {len(detected_skin_types)}")
    except Exception as e:
        print(f"Error extracting NER features: {e}")
        extracted_ingredients = []
        extracted_concerns = []
        detected_skin_types = []

    boosts = []
    context_boosts = []
    
    try:
        if detected_skin_types:
            skin_types_str = ", ".join([st.title() for st in detected_skin_types])
            boosts.append(f"Recommended for: {skin_types_str} skin types")
        
        ingredient_benefits = {
            'retinol': 'anti-aging and texture improvement',
            'vitamin c': 'brightening and antioxidant protection',
            'hyaluronic acid': 'intense hydration and skin plumping',
            'niacinamide': 'oil control and redness reduction',
            'salicylic acid': 'pore cleansing and acne fighting',
            'glycolic acid': 'exfoliation and radiance boosting',
            'peptides': 'firmness and wrinkle reduction',
            'ceramides': 'barrier repair and moisture retention',
            'spf': 'essential sun protection',
            'squalane': 'lightweight hydration and barrier support',
            'lactic acid': 'gentle exfoliation and hydration',
            'azelaic acid': 'acne and rosacea treatment',
            'bakuchiol': 'natural retinol alternative',
            'tea tree': 'antibacterial and acne control',
            'witch hazel': 'pore tightening and oil control',
            'aloe vera': 'soothing and calming inflammation',
            'green tea': 'antioxidant and anti-inflammatory',
            'jojoba oil': 'moisture balancing',
            'argan oil': 'nourishing and antioxidant rich',
            'shea butter': 'rich emollient and skin softening',
            'glycerin': 'humectant for moisture retention',
            'collagen': 'skin firming and elasticity',
            'caffeine': 'de-puffing and circulation',
            'licorice root': 'brightening and anti-inflammatory',
        }
        
        if extracted_ingredients:
            ingredient_explanations_added = 0
            for ingredient in extracted_ingredients[:5]:
                if ingredient in ingredient_benefits:
                    boosts.append(f"Contains {ingredient} for {ingredient_benefits[ingredient]}")
                    ingredient_explanations_added += 1
                    if ingredient_explanations_added >= 2:
                        break
        
        user_skin_type = user_input.get("skin_type", "").lower() if user_input else ""
        if user_skin_type:
            user_skin_lower = user_skin_type.lower()
            detected_lower = [st.lower() for st in detected_skin_types]
            
            if user_skin_lower in detected_lower:
                boosts.append(f"Specifically recommended for {user_skin_type} skin based on user reviews")
        
        if extracted_concerns:
            top_concerns = extracted_concerns[:2]
            concern_descriptions = {
                'acne': 'acne and breakouts',
                'wrinkles': 'fine lines and wrinkles', 
                'aging': 'signs of aging',
                'dark spots': 'dark spots and hyperpigmentation',
                'redness': 'redness and irritation',
                'dryness': 'dryness and dehydration',
                'oiliness': 'excess oil and shine',
            }
            
            for concern in top_concerns:
                description = concern_descriptions.get(concern, concern)
                boosts.append(f"Targets {description} based on product formulation")
        
        user_subcategory = user_input.get("sub_category", "").lower() if user_input else ""
        if user_subcategory:
            product_categories = [
                primary_category,
                secondary_category,
                str(product.get("tertiary_category", "")).lower()
            ]
            if any(user_subcategory in cat for cat in product_categories if cat):
                boosts.append(f"Matches your selected '{user_subcategory.title()}' category")
        
        user_brand = user_input.get("brand", "").lower() if user_input and user_input.get("brand") else ""
        if product_brand and user_brand and product_brand.lower() == user_brand:
            boosts.append(f"Matches your preferred brand '{product_brand.title()}'")
        
        if user_input and "budget" in user_input and price_usd is not None and not pd.isna(price_usd):
            try:
                price = float(price_usd)
                user_budget = user_input["budget"]
                if user_budget == "100+":
                    if price >= 100:
                        boosts.append(f"Fits your ${user_budget} budget at ${price:.2f}")
                else:
                    budget_parts = user_budget.split("-")
                    if len(budget_parts) == 2:
                        min_budget, max_budget = float(budget_parts[0]), float(budget_parts[1])
                        if min_budget <= price <= max_budget:
                            boosts.append(f"Fits your ${user_budget} budget at ${price:.2f}")
            except (ValueError, TypeError, AttributeError):
                pass

        if context_input:
            print(f"Context input detected: {context_input}")
            
            print(f"Weather context: {context_input.get('weather', 'Not provided')}")
            print(f"Temperature context: {context_input.get('temperature', 'Not provided')}")
            
            product_ingredients_lower = [ing.lower() for ing in extracted_ingredients]
            
            weather_code_mapping = {
                0: "clear",
                1: "clear", 
                2: "partly cloudy",
                3: "overcast",
                45: "fog",
                48: "fog",
                51: "drizzle",
                53: "drizzle",
                55: "drizzle",
                61: "rain",
                63: "rain",
                65: "rain",
                71: "snow",
                73: "snow",
                75: "snow",
                95: "thunderstorm"
            }
            
            weather_code = context_input.get("weather")
            weather_text = ""
            if weather_code is not None:
                try:
                    weather_code_int = int(weather_code)
                    weather_text = weather_code_mapping.get(weather_code_int, "clear")
                    print(f"Converted weather code {weather_code} to: {weather_text}")
                except (ValueError, TypeError):
                    weather_text = str(weather_code).lower()
                    print(f"Using raw weather value: {weather_text}")
            
            if weather_text and product_ingredients_lower:
                print(f"Processing weather: {weather_text}")
                
                weather_ingredient_mapping = {
                    'clear': {
                        'spf': 'UV protection is essential in sunny conditions',
                        'vitamin c': 'antioxidants protect against sun-induced free radicals',
                        'niacinamide': 'helps prevent sun-induced hyperpigmentation',
                        'green tea': 'provides extra antioxidant defense against UV damage',
                        'licorice root': 'helps brighten skin affected by sun exposure'
                    },
                    'partly cloudy': {
                        'hyaluronic acid': 'maintains hydration during changing light conditions',
                        'niacinamide': 'balances oil production as weather fluctuates',
                        'vitamin c': 'protects against UV rays that penetrate clouds',
                        'peptides': 'supports skin barrier during weather transitions'
                    },
                    'overcast': {
                        'glycolic acid': 'gentle exfoliation revitalizes dull complexion',
                        'vitamin c': 'brightens skin lacking natural sunlight',
                        'peptides': 'promotes skin renewal in stable, cloudy conditions',
                        'retinol': 'works effectively without sun sensitivity concerns'
                    },
                    'fog': {
                        'salicylic acid': 'prevents pore congestion in humid, foggy air',
                        'niacinamide': 'controls excess oil in moisture-rich conditions',
                        'tea tree': 'fights bacteria that thrive in damp environments'
                    },
                    'rain': {
                        'salicylic acid': 'prevents breakouts caused by humid rainy weather',
                        'hyaluronic acid': 'provides lightweight hydration that absorbs quickly',
                        'ceramides': 'strengthens skin barrier against humidity changes'
                    },
                    'snow': {
                        'ceramides': 'repairs skin barrier damaged by cold, dry air',
                        'shea butter': 'creates protective layer against freezing temperatures',
                        'squalane': 'provides deep moisture without feeling heavy under layers',
                        'peptides': 'helps maintain skin elasticity in freezing conditions'
                    },
                    'thunderstorm': {
                        'niacinamide': 'calms skin irritated by sudden weather changes',
                        'aloe vera': 'soothes skin affected by atmospheric pressure drops',
                        'green tea': 'provides antioxidant protection during unstable weather'
                    }
                }
                
                weather_matched = False
                
                for weather_key, ingredient_map in weather_ingredient_mapping.items():
                    if weather_key in weather_text:
                        matching_ingredients = []
                        for ingredient, benefit in ingredient_map.items():
                            if any(ingredient in prod_ing for prod_ing in product_ingredients_lower):
                                matching_ingredients.append((ingredient, benefit))
                        
                        if matching_ingredients:
                            for ingredient, benefit in matching_ingredients[:2]:  
                                context_boosts.append(f"{benefit} during {weather_text} conditions")
                                weather_matched = True
                                print(f"Added weather-ingredient boost: {ingredient} for {weather_text}")
                            break
                
                if not weather_matched:
                    print(f"No ingredient matches found for {weather_text}, skipping generic explanation")
            
            temperature_value = context_input.get("temperature")
            if temperature_value and product_ingredients_lower:
                try:
                    temp = float(temperature_value)
                    print(f"Processing temperature: {temp}°C")
                    
                    if temp >= 30: 
                        hot_ingredients = {
                            'niacinamide': 'controls excess oil production in hot weather',
                            'salicylic acid': 'prevents heat-induced breakouts and congestion', 
                            'hyaluronic acid': 'provides lightweight hydration without heaviness',
                            'tea tree': 'keeps skin fresh and clear in sweaty conditions',
                            'witch hazel': 'refreshes and tightens pores in heat'
                        }
                        matching_found = False
                        for ingredient, benefit in hot_ingredients.items():
                            if any(ingredient in prod_ing for prod_ing in product_ingredients_lower):
                                context_boosts.append(f"{benefit} in hot weather ({temp}°C)")
                                matching_found = True
                                break
                        if not matching_found:
                            print(f"No hot-weather ingredient matches found for {temp}°C")
                                
                    elif temp <= 10:  
                        cold_ingredients = {
                            'ceramides': 'strengthens skin barrier against cold, dry air',
                            'shea butter': 'provides rich protection from harsh winter conditions',
                            'squalane': 'delivers deep moisture without feeling heavy',
                            'peptides': 'supports skin resilience in cold temperatures'
                        }
                        matching_found = False
                        for ingredient, benefit in cold_ingredients.items():
                            if any(ingredient in prod_ing for prod_ing in product_ingredients_lower):
                                context_boosts.append(f"{benefit} in cold weather ({temp}°C)")
                                matching_found = True
                                break
                        if not matching_found:
                            print(f"No cold-weather ingredient matches found for {temp}°C")
                    
                    print(f"Temperature analysis completed for {temp}°C")
                except (ValueError, TypeError):
                    print(f"Could not parse temperature: {temperature_value}")
            
            print(f"Context boosts generated: {len(context_boosts)} context boosts")
        
        print(f"Regular boosts generated: {len(boosts)} regular boosts")
        
    except Exception as e:
        print(f"Error generating boosts: {e}")
        boosts = ["• it matches your general preferences"]

    try:
        explanation_parts = []
        explanation_parts.append(f"<strong>{product_name}</strong> was recommended because:")
        
        all_boosts = []
        
        if context_boosts:
            print(f"Adding ALL {len(context_boosts)} context boosts to explanation")
            all_boosts.extend(context_boosts)
        
        if boosts:
            remaining_slots = 6 - len(all_boosts)
            print(f"{remaining_slots} slots remaining for regular boosts")
            if remaining_slots > 0:
                all_boosts.extend(boosts[:remaining_slots])
        
        if not all_boosts:
            all_boosts.append("• it matches your general preferences")
        
        print(f"Final boosts to include: {all_boosts}")
        
        for boost in all_boosts:
            explanation_parts.append(f"• {boost}")

        explanation = "\n".join(explanation_parts)
        structured_boosts = all_boosts.copy()
        print(f"Explanation built successfully with {len(all_boosts)} boosts ({len(context_boosts)} context-based)")
        
    except Exception as e:
        print(f"Error building explanation: {e}")
        explanation = f"<strong>{product_name}</strong> was recommended because it matches your preferences."

    return {
        "explanation": explanation,
        "structured_boosts": structured_boosts,
        "keywords": keywords,
        "sample_positive_review": sample_positive_review,
        "boosts": boosts + context_boosts,
        "categories": [
            f"Primary: {product.get('primary_category', 'N/A')}",
            f"Secondary: {product.get('secondary_category', 'N/A')}",
            f"Tertiary: {product.get('tertiary_category', 'N/A')}"
        ],
        "ingredients": extracted_ingredients[:5],
        "skin_concerns": extracted_concerns[:3],
        "detected_skin_types": detected_skin_types
    }