import json

def calculate_daily_calories(profile_path):
    """
    Reads a user profile JSON and calculates the Total Daily Energy Expenditure (TDEE)
    based on the Mifflin-St Jeor Equation and specific goals.
    """
    with open(profile_path, 'r') as f:
        user = json.load(f)

    # Extract Biometrics
    w = user['biometrics']['weight_kg']
    h = user['biometrics']['height_cm']
    a = user['biometrics']['age']
    gender = user['biometrics']['gender'].lower()
    activity_multiplier = user['biometrics']['activity_level']
    
    # 1. Calculate Basal Metabolic Rate (BMR)
    # Formula: (10 * weight) + (6.25 * height) - (5 * age) + s
    if gender == 'male':
        bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
    else:
        bmr = (10 * w) + (6.25 * h) - (5 * a) - 161

    # 2. Calculate TDEE (Maintenance Calories)
    tdee = bmr * activity_multiplier

    # 3. Adjust based on Goal
    target = user['goals']['target']
    rate = user['goals'].get('weekly_rate_kg', 0)
    
    # Roughly 7700 calories per 1kg of body fat
    # Daily adjustment = (7700 * weekly_rate) / 7 days
    daily_adjustment = (7700 * rate) / 7

    if target == "lose_weight":
        final_target = tdee - daily_adjustment
    elif target == "gain_weight":
        final_target = tdee + daily_adjustment
    else:
        final_target = tdee

    return {
        "user_id": user['user_id'],
        "bmr": round(bmr),
        "maintenance_tdee": round(tdee),
        "daily_calorie_target": round(final_target),
        "dietary_framework": user['preferences']['dietary_framework']
    }

# --- Example Usage for your MVP ---
# result = calculate_daily_calories('user_maintain.json')
# print(f"Target for {result['user_id']}: {result['daily_calorie_target']} kcal")