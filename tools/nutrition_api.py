"""
tools/nutrition_api.py — USDA FoodData Central API Client
==========================================================
Provides two clean functions for the NutritionistAgent:

    search_food(query)          →  list of {fdc_id, description}
    get_food_nutrients(fdc_id)  →  flat nutrient dict per 100 g

Requires:
    pip install requests python-dotenv

Environment variable (set in .env):
    USDA_API_KEY=your_key_here

Get a free key at: https://fdc.nal.usda.gov/api-guide.html
"""

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.getenv("USDA_API_KEY")
_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# USDA nutrient IDs we care about (stable across datasets)
# Full list: https://fdc.nal.usda.gov/fdc-app.html#/nutrient-list
_NUTRIENT_MAP: dict[int, str] = {
    1008: "calories",      # Energy (kcal)
    1003: "protein_g",     # Protein
    1005: "carbs_g",       # Carbohydrate, by difference
    1004: "fat_g",         # Total lipid (fat)
    1079: "fiber_g",       # Fiber, total dietary
    2000: "sugar_g",       # Sugars, total
    1093: "sodium_mg",     # Sodium
}

# How many search results to return by default
_DEFAULT_PAGE_SIZE = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_food(
    query: str,
    page_size: int = _DEFAULT_PAGE_SIZE,
    data_type: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Search the USDA FoodData Central database by food name.

    Parameters
    ----------
    query : str
        Human-readable food name, e.g. "Gala Apple" or "chicken breast".
    page_size : int
        Maximum number of results to return (default 5, max 50).
    data_type : list[str] | None
        Restrict results to specific USDA datasets.
        Options: "Foundation", "SR Legacy", "Survey (FNDDS)", "Branded".
        Defaults to Foundation + SR Legacy for the cleanest nutrient data.

    Returns
    -------
    list[dict]  — each item has keys: fdc_id (int), description (str),
                  data_type (str), brand_owner (str | None)

    On error returns: [{"error": "<message>"}]
    """
    if not _API_KEY:
        return [{"error": "USDA_API_KEY is not set. Add it to your .env file."}]

    if data_type is None:
        data_type = ["Foundation", "SR Legacy"]

    params: dict[str, Any] = {
        "api_key":   _API_KEY,
        "query":     query.strip(),
        "pageSize":  min(page_size, 50),
        "dataType":  ",".join(data_type),
    }

    try:
        response = requests.get(
            f"{_BASE_URL}/foods/search",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return [{"error": "Network error — could not reach the USDA API. Check your internet connection."}]
    except requests.exceptions.Timeout:
        return [{"error": "Request timed out. The USDA API did not respond within 10 seconds."}]
    except requests.exceptions.HTTPError as exc:
        return [{"error": f"USDA API returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"}]

    data = response.json()
    foods = data.get("foods", [])

    if not foods:
        return [{"error": f"No results found for '{query}'. Try a different search term."}]

    return [
        {
            "fdc_id":      item["fdcId"],
            "description": item.get("description", "Unknown"),
            "data_type":   item.get("dataType", "Unknown"),
            "brand_owner": item.get("brandOwner"),        # None for whole foods
        }
        for item in foods
    ]


def get_food_nutrients(fdc_id: int) -> dict[str, Any]:
    """
    Fetch the full nutrient profile for a single food by its FDC ID.

    Values are expressed per 100 g of food.

    Parameters
    ----------
    fdc_id : int
        The USDA FoodData Central identifier returned by search_food().

    Returns
    -------
    dict with keys:
        food (str), fdc_id (int), per_100g (str),
        calories (float), protein_g (float), carbs_g (float), fat_g (float),
        fiber_g (float | None), sugar_g (float | None), sodium_mg (float | None)

    On error returns: {"error": "<message>"}
    """
    if not _API_KEY:
        return {"error": "USDA_API_KEY is not set. Add it to your .env file."}

    try:
        response = requests.get(
            f"{_BASE_URL}/food/{fdc_id}",
            params={"api_key": _API_KEY},
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return {"error": "Network error — could not reach the USDA API."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out fetching nutrient details."}
    except requests.exceptions.HTTPError as exc:
        if exc.response.status_code == 404:
            return {"error": f"Food ID {fdc_id} not found in the USDA database."}
        return {"error": f"USDA API returned HTTP {exc.response.status_code}."}

    raw = response.json()

    # --- Flatten the deeply nested foodNutrients list ---
    # Each element looks like:
    #   { "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
    #     "amount": 52.0 }
    nutrient_lookup: dict[int, float] = {}
    for item in raw.get("foodNutrients", []):
        nutrient_info = item.get("nutrient", {})
        nutrient_id   = nutrient_info.get("id")
        amount        = item.get("amount")
        if nutrient_id and amount is not None:
            nutrient_lookup[nutrient_id] = float(amount)

    # Check that we at least got the four core macros
    core_ids = {1008, 1003, 1005, 1004}
    missing  = core_ids - nutrient_lookup.keys()
    if len(missing) == len(core_ids):
        return {"error": f"No recognisable nutrient data found for FDC ID {fdc_id}."}

    return {
        "food":       raw.get("description", "Unknown"),
        "fdc_id":     fdc_id,
        "per_100g":   "All values are per 100 g of food",
        # Core macros — default to 0 if a value is absent
        "calories":   round(nutrient_lookup.get(1008, 0.0), 1),
        "protein_g":  round(nutrient_lookup.get(1003, 0.0), 1),
        "carbs_g":    round(nutrient_lookup.get(1005, 0.0), 1),
        "fat_g":      round(nutrient_lookup.get(1004, 0.0), 1),
        # Bonus nutrients — None if not present in this food entry
        "fiber_g":    round(nutrient_lookup[1079], 1) if 1079 in nutrient_lookup else None,
        "sugar_g":    round(nutrient_lookup[2000], 1) if 2000 in nutrient_lookup else None,
        "sodium_mg":  round(nutrient_lookup[1093], 1) if 1093 in nutrient_lookup else None,
    }


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUERY = "Broccoli"

    print(f"=== Searching USDA for: '{TEST_QUERY}' ===\n")
    results = search_food(TEST_QUERY, page_size=3)

    for r in results:
        if "error" in r:
            print(f"[ERROR] {r['error']}")
            raise SystemExit(1)
        print(f"  [{r['fdc_id']}] {r['description']}  ({r['data_type']})")

    # Use the first clean result for the nutrient deep-dive
    first_id: int = results[0]["fdc_id"]
    print(f"\n=== Nutrient profile for FDC ID {first_id} (per 100 g) ===\n")
    nutrients = get_food_nutrients(first_id)

    if "error" in nutrients:
        print(f"[ERROR] {nutrients['error']}")
    else:
        for key, value in nutrients.items():
            if value is not None:
                print(f"  {key:<14} {value}")
