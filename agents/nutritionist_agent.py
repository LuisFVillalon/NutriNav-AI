import json
from typing import Any, Dict, List, Optional, Tuple

from tools.calculate_daily_calories import calculate_daily_calories
from tools.compliance_checker import ComplianceChecker
from tools.nutrition_api import search_food, get_food_nutrients
from agents.prompts import AgentPrompts

# ---------------------------------------------------------------------------
# Macro ratio presets keyed by dietary framework (protein%, carb%, fat%)
# ---------------------------------------------------------------------------
_MACRO_RATIOS: Dict[str, Dict[str, float]] = {
    "keto":          {"protein": 0.25, "carbs": 0.05, "fat": 0.70},
    "vegan":         {"protein": 0.30, "carbs": 0.45, "fat": 0.25},
    "vegetarian":    {"protein": 0.28, "carbs": 0.42, "fat": 0.30},
    "paleo":         {"protein": 0.35, "carbs": 0.25, "fat": 0.40},
    "carnivore":     {"protein": 0.35, "carbs": 0.00, "fat": 0.65},
    "pescatarian":   {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
    "mediterranean": {"protein": 0.20, "carbs": 0.50, "fat": 0.30},
    "high protein":  {"protein": 0.40, "carbs": 0.30, "fat": 0.30},
    "low-lectin":    {"protein": 0.28, "carbs": 0.37, "fat": 0.35},
    "whole30":       {"protein": 0.30, "carbs": 0.35, "fat": 0.35},
    "dash":          {"protein": 0.18, "carbs": 0.55, "fat": 0.27},
    "fodmap":        {"protein": 0.25, "carbs": 0.50, "fat": 0.25},
    "default":       {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
}

_CAL_PER_GRAM = {"protein": 4.0, "carbs": 4.0, "fat": 9.0}

# ---------------------------------------------------------------------------
# Keyword → canonical framework key.  First match wins.
# Lets "high-protein", "High Protein", etc. all resolve to "high protein".
# ---------------------------------------------------------------------------
_KEYWORD_MACRO_PATTERNS: List[Tuple[List[str], str]] = [
    (["carnivore", "zero carb", "zero-carb", "all meat", "all-meat"], "carnivore"),
    (["pescatarian", "pesce"],                                         "pescatarian"),
    (["mediterranean", "med diet"],                                    "mediterranean"),
    (["high protein", "high-protein"],                                 "high protein"),
    (["low-lectin", "low lectin"],                                     "low-lectin"),
    (["whole30", "whole 30"],                                          "whole30"),
    (["fodmap", "low fodmap", "low-fodmap"],                           "fodmap"),
    (["dash"],                                                         "dash"),
    (["keto", "ketogenic", "ketosis"],                                 "keto"),
    (["vegan", "plant-based", "plant based"],                          "vegan"),
    (["vegetarian"],                                                   "vegetarian"),
    (["paleo", "ancestral"],                                           "paleo"),
]

# All frameworks for which we hold explicit macro ratios (excluding fallback)
_KNOWN_FRAMEWORKS: frozenset = frozenset(_MACRO_RATIOS.keys()) - {"default"}


def _resolve_macro_ratios(framework: str) -> Dict[str, float]:
    """
    Resolve macro ratios for *framework* using a 3-tier fallback:
      1. Exact key match in _MACRO_RATIOS
      2. Keyword scan via _KEYWORD_MACRO_PATTERNS
      3. "default" ratios
    """
    key = framework.strip().lower()
    if key in _MACRO_RATIOS:
        return _MACRO_RATIOS[key]
    for keywords, canonical in _KEYWORD_MACRO_PATTERNS:
        if any(kw in key for kw in keywords):
            return _MACRO_RATIOS[canonical]
    return _MACRO_RATIOS["default"]


def _is_known_framework(framework: str) -> bool:
    """Return True if we have explicit macro ratios (or keyword match) for *framework*."""
    key = framework.strip().lower()
    if key in _KNOWN_FRAMEWORKS:
        return True
    for keywords, _ in _KEYWORD_MACRO_PATTERNS:
        if any(kw in key for kw in keywords):
            return True
    return False


# ---------------------------------------------------------------------------
# Framework-specific bullet points injected into the constraint block
# ---------------------------------------------------------------------------
_FRAMEWORK_NOTES: Dict[str, List[str]] = {
    "keto": [
        "Stay UNDER the carb ceiling -- hard limit for maintaining ketosis.",
        "Prioritise healthy fats: avocado, olive oil, nuts, fatty fish.",
        "Avoid grains, legumes, starchy vegetables, and added sugars.",
    ],
    "vegan": [
        "All protein must come from plant sources.",
        "Combine complementary proteins (e.g. rice + lentils, tofu + edamame).",
        "Ensure adequate B12, iron, and omega-3 from fortified foods or seeds.",
        "No animal products, including dairy and eggs.",
    ],
    "vegetarian": [
        "Eggs and dairy are permitted; no meat or fish.",
        "Prioritise legumes, Greek yoghurt, cottage cheese for protein.",
        "Whole-food plant sources preferred over processed alternatives.",
    ],
    "paleo": [
        "Exclude grains, legumes, refined sugar, and all dairy.",
        "Prioritise lean meats, fish, eggs, vegetables, fruits, and nuts.",
        "Carbs must come from whole-food sources only.",
    ],
    "carnivore": [
        "Animal products ONLY -- zero plant foods.",
        "Carbohydrates must be 0 g; any plant carbs are a hard violation.",
        "Primary protein sources: beef, lamb, pork, poultry, eggs, fatty fish.",
        "Fat provides the vast majority of energy -- do not reduce it.",
    ],
    "pescatarian": [
        "Fish and seafood are the primary animal proteins; no red meat or poultry.",
        "Plant proteins (legumes, tofu) supplement fish-based meals.",
        "Prioritise omega-3-rich fish: salmon, mackerel, sardines, trout.",
    ],
    "mediterranean": [
        "Emphasise olive oil, legumes, whole grains, vegetables, and fruits.",
        "Fish 2-3 times per week; red meat limited to occasional servings.",
        "Moderate dairy (Greek yoghurt, feta); minimal processed foods.",
    ],
    "high protein": [
        "Protein drives satiety -- distribute across all three meals.",
        "Lean sources preferred: chicken breast, turkey, egg whites, Greek yoghurt.",
        "Keep carbs and fat moderate; do not let either crowd out protein.",
    ],
    "low-lectin": [
        "Avoid high-lectin foods: beans, lentils, nightshades, grains with gluten.",
        "Permitted carb sources: sweet potatoes, white rice, peeled cucumbers.",
        "Prioritise pasture-raised animal proteins and pressure-cooked legumes.",
    ],
    "whole30": [
        "No grains, dairy, legumes, added sugar, or alcohol for 30 days.",
        "Carbs come exclusively from vegetables and whole fruits.",
        "Read all labels -- many sauces and seasonings contain excluded ingredients.",
    ],
    "dash": [
        "Emphasise fruits, vegetables, whole grains, and low-fat dairy.",
        "Strictly limit sodium to 1,500-2,300 mg/day to support blood pressure.",
        "Limit red meat, sweets, and sugar-sweetened beverages.",
    ],
    "fodmap": [
        "Avoid high-FODMAP foods: garlic, onion, wheat, lactose, stone fruits.",
        "Permitted: rice, oats, firm tofu, lactose-free dairy, most vegetables.",
        "Portion sizes matter -- even low-FODMAP foods can trigger symptoms in bulk.",
    ],
    "default": [
        "Balanced macros -- no strict food-group exclusions.",
        "Whole foods preferred; minimise ultra-processed items.",
        "Adequate protein supports satiety and muscle maintenance.",
    ],
}


class NutritionistAgent:
    """
    Scientific Controller that translates a user profile into precise daily
    calorie and macronutrient targets.

    Prompts are sourced from prompts.AgentPrompts so that instruction changes
    never require edits to this class.
    """

    SYSTEM_PROMPT: str = AgentPrompts.NUTRITIONIST_SYSTEM_PROMPT
    _checker = ComplianceChecker()

    def __init__(self, profile_path: str) -> None:
        """
        Parameters
        ----------
        profile_path : str
            Path to a single-user JSON profile (not the aggregate
            database/users.json list).
        """
        self.profile_path = profile_path

        with open(profile_path, "r") as fh:
            self._profile: Dict[str, Any] = json.load(fh)

        self._targets: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily_targets(self) -> Dict[str, Any]:
        """
        Run the calculator tool and derive macronutrient targets.

        Returns a dict with keys:
            user_id, dietary_framework, bmr, maintenance_tdee,
            daily_calorie_target, macros {protein_g, carbs_g, fat_g},
            warning (str | None)
        """
        if self._targets is not None:
            return self._targets

        calc = calculate_daily_calories(self.profile_path)
        daily_kcal: int = calc["daily_calorie_target"]
        framework: str = calc["dietary_framework"].strip().lower()

        ratios = _MACRO_RATIOS.get(framework, _MACRO_RATIOS["default"])
        macros = {
            "protein_g": round((daily_kcal * ratios["protein"]) / _CAL_PER_GRAM["protein"]),
            "carbs_g":   round((daily_kcal * ratios["carbs"])   / _CAL_PER_GRAM["carbs"]),
            "fat_g":     round((daily_kcal * ratios["fat"])     / _CAL_PER_GRAM["fat"]),
        }

        # Run the compliance engine — replaces the old inline safety check
        compliance = self._checker.verify_daily_goal(
            user_profile=self._profile,
            calculated_calories=daily_kcal,
        )

        self._targets = {
            "user_id":              calc["user_id"],
            "dietary_framework":    framework,
            "bmr":                  calc["bmr"],
            "maintenance_tdee":     calc["maintenance_tdee"],
            "daily_calorie_target": daily_kcal,
            "macros":               macros,
            # Full compliance report — callers can inspect status / issues / suggestions
            "compliance":           compliance.to_dict(),
        }
        return self._targets

    def format_instructions_for_chef(self) -> str:
        """
        Render the NUTRITIONIST_CONSTRAINT_TEMPLATE from prompts.py with
        live user data and return the filled string.

        The ChefAgent (or the orchestrator) passes this directly into the
        Chef's user prompt so constraints are unambiguous.
        """
        t = self.get_daily_targets()

        notes_text = "\n".join(
            f"  • {note}"
            for note in _FRAMEWORK_NOTES.get(
                t["dietary_framework"], _FRAMEWORK_NOTES["default"]
            )
        )

        compliance    = t.get("compliance", {})
        issues        = compliance.get("issues", [])
        warning_block = ("\n" + "\n".join(f"⚠  {i}" for i in issues) + "\n") if issues else ""

        return AgentPrompts.render(
            AgentPrompts.NUTRITIONIST_CONSTRAINT_TEMPLATE,
            daily_calorie_target=t["daily_calorie_target"],
            dietary_framework=t["dietary_framework"].capitalize(),
            protein_g=t["macros"]["protein_g"],
            carbs_g=t["macros"]["carbs_g"],
            fat_g=t["macros"]["fat_g"],
            framework_notes=notes_text,
            warning_block=warning_block,
        )

    def validate_ingredient(self, food_name: str) -> Dict[str, Any]:
        """
        Look up a food item in the USDA FoodData Central database and return
        its nutrient profile per 100 g.

        Used to cross-reference the calorie density of key ingredients that
        the Chef Agent proposes, providing the Nutritionist with real-world
        data to back up (or challenge) the Chef's macro estimates.

        Parameters
        ----------
        food_name : str
            Plain-language food name, e.g. "chicken breast" or "brown rice".

        Returns
        -------
        dict with keys:
            status   : "found" | "not_found" | "lookup_failed"
            usda_name: str   (official USDA description, only when found)
            per_100g : dict  (calories, protein_g, carbs_g, fat_g, ...)
            error    : str   (only present when status != "found")
        """
        results = search_food(food_name, page_size=1)

        if not results or "error" in results[0]:
            return {
                "food":   food_name,
                "status": "not_found",
                "error":  results[0].get("error") if results else "No results returned.",
            }

        best     = results[0]
        nutrients = get_food_nutrients(best["fdc_id"])

        if "error" in nutrients:
            return {
                "food":   food_name,
                "status": "lookup_failed",
                "error":  nutrients["error"],
            }

        return {
            "food":      food_name,
            "status":    "found",
            "usda_name": nutrients["food"],
            "fdc_id":    best["fdc_id"],
            "per_100g": {
                "calories":  nutrients["calories"],
                "protein_g": nutrients["protein_g"],
                "carbs_g":   nutrients["carbs_g"],
                "fat_g":     nutrients["fat_g"],
                "fiber_g":   nutrients.get("fiber_g"),
                "sugar_g":   nutrients.get("sugar_g"),
                "sodium_mg": nutrients.get("sodium_mg"),
            },
        }

    def generate_feedback_note(self, compliance_report: Dict[str, Any]) -> str:
        """
        Convert a ComplianceReport dict into a clear, actionable feedback
        message that the Orchestrator passes back to the Chef Agent as the
        next user turn in the conversation.

        Parameters
        ----------
        compliance_report : dict
            Output of ComplianceReport.to_dict() — must have keys
            "status", "issues", "suggestions".

        Returns
        -------
        str — a formatted feedback message ready to append to the LLM
              conversation history as a user-role message.
        """
        issues      = compliance_report.get("issues",      [])
        suggestions = compliance_report.get("suggestions", [])
        status      = compliance_report.get("status",      "UNKNOWN")
        framework   = self._targets["dietary_framework"].capitalize() if self._targets else "Unknown"

        lines = [
            "NUTRITIONIST FEEDBACK -- Your meal plan has been reviewed and rejected.",
            f"Compliance status : {status}",
            f"Dietary framework : {framework}",
            "",
            "Violations that must be corrected:",
        ]

        for idx, issue in enumerate(issues, 1):
            lines.append(f"  {idx}. {issue}")

        if suggestions:
            lines += ["", "Required corrections (address each one):"]
            for idx, suggestion in enumerate(suggestions, 1):
                lines.append(f"  {idx}. {suggestion}")

        lines += [
            "",
            "Action required:",
            "  Regenerate the COMPLETE meal plan from scratch addressing every",
            "  point above. All original constraints (calorie target, dietary",
            "  framework, allergies, liked/disliked foods) remain in force.",
            "  Include the [JSON_START]...[JSON_END] compliance block at the end.",
        ]

        return "\n".join(lines)
