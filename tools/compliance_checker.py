"""
tools/compliance_checker.py — Nutritional Compliance & Safety Engine
=====================================================================
Validates calorie targets and meal plans against established clinical
guidelines before they reach the user.

Guidelines sources
------------------
  • USDA Dietary Guidelines for Americans, 2020–2025
    https://www.dietaryguidelines.gov/
  • WHO healthy-diet fact sheet (no. 394, 2020)
    https://www.who.int/news-room/fact-sheets/detail/healthy-diet
  • Institute of Medicine (IOM) Acceptable Macronutrient Distribution
    Ranges (AMDR): protein 10–35 %, carbs 45–65 %, fat 20–35 %

Usage
-----
    from tools.compliance_checker import ComplianceChecker

    checker = ComplianceChecker()

    # Validate a calorie target
    report = checker.verify_daily_goal(user_profile, calculated_calories=1850)

    # Validate a full day's nutrients
    report = checker.verify_meal_plan(meal_plan_data, dietary_framework="keto")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_APPROVED = "APPROVED"
STATUS_WARNING  = "WARNING"
STATUS_REJECTED = "REJECTED"

# ---------------------------------------------------------------------------
# Hard safety thresholds (clinical minimums)
# ---------------------------------------------------------------------------

# Minimum kcal/day below which a plan is flagged UNSAFE
# Sources: National Institutes of Health, British Dietetic Association
_MIN_CALORIES_FEMALE = 1_200   # kcal
_MIN_CALORIES_MALE   = 1_500   # kcal
_MIN_CALORIES_DEFAULT = 1_200  # kcal (used when gender is unknown)

# Maximum safe weekly weight-loss rate (kg)
# Source: WHO — losing >1 kg/week is unsustainable without medical supervision
_MAX_WEEKLY_LOSS_KG = 1.0

# Maximum safe weekly weight-gain rate (kg)
# Source: general hypertrophy guidelines — >0.5 kg/week is primarily fat gain
_MAX_WEEKLY_GAIN_KG = 0.5

# ---------------------------------------------------------------------------
# Macro balance thresholds
# ---------------------------------------------------------------------------

# IOM AMDR for protein as % of total calories
_PROTEIN_PCT_MIN = 0.10
_PROTEIN_PCT_MAX = 0.35

# USDA / WHO: sodium ≤ 2,300 mg/day for general population
_MAX_SODIUM_MG = 2_300

# USDA: saturated fat < 10 % of daily calories
# (1 g saturated fat = 9 kcal)
_MAX_SAT_FAT_PCT = 0.10

# WHO: free/added sugar < 10 % of daily calories (< 5 % for additional benefit)
# (1 g sugar = 4 kcal)
_MAX_SUGAR_PCT         = 0.10
_IDEAL_SUGAR_PCT_UPPER = 0.05

# ---------------------------------------------------------------------------
# Framework-specific carbohydrate reference bands (grams per day)
# These define what is LOW / TARGET / HIGH for each dietary pattern.
# Used by verify_meal_plan() to contextualise carb levels appropriately.
# ---------------------------------------------------------------------------

_FRAMEWORK_CARB_BANDS: dict[str, dict[str, int]] = {
    # Keto: net carbs must stay extremely low to maintain ketosis
    "keto": {
        "low":    0,
        "target": 25,
        "high":   50,    # >50 g/day risks kicking users out of ketosis
        "ceiling": 50,
    },
    # Paleo: no grains/legumes so carbs naturally lower than standard
    "paleo": {
        "low":    50,
        "target": 150,
        "high":   200,
        "ceiling": 225,
    },
    # Vegan: higher reliance on carbohydrate-dense whole plants
    "vegan": {
        "low":    130,   # below 130 g/day may indicate inadequate fibre
        "target": 275,
        "high":   375,
        "ceiling": 400,
    },
    # Vegetarian: similar to vegan but slightly lower due to dairy/egg protein
    "vegetarian": {
        "low":    130,
        "target": 230,
        "high":   350,
        "ceiling": 380,
    },
    # Standard / balanced macros
    "default": {
        "low":    130,   # IOM minimum for adequate brain glucose
        "target": 225,
        "high":   325,
        "ceiling": 350,
    },
}

# Calories per gram for each macro
_KCAL_PER_G = {"protein": 4.0, "carbs": 4.0, "fat": 9.0, "sat_fat": 9.0, "sugar": 4.0}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComplianceReport:
    """
    Structured result returned by every ComplianceChecker method.

    status      : "APPROVED" | "WARNING" | "REJECTED"
    issues      : Violations that triggered the current status.
    suggestions : Actionable fixes for every issue raised.
    metadata    : Extra context (thresholds used, checked_at timestamp, etc.)
    """
    status:      str
    issues:      list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata:    dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status":      self.status,
            "issues":      self.issues,
            "suggestions": self.suggestions,
            "metadata":    self.metadata,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ComplianceChecker:
    """
    Validates nutritional plans against USDA and WHO clinical safety standards.

    Designed to be instantiated once and called repeatedly:

        checker = ComplianceChecker()
        report  = checker.verify_daily_goal(profile, 1450).to_dict()
    """

    # ------------------------------------------------------------------
    # verify_daily_goal
    # ------------------------------------------------------------------

    def verify_daily_goal(
        self,
        user_profile: dict[str, Any],
        calculated_calories: int | float,
    ) -> ComplianceReport:
        """
        Audit a proposed daily calorie target against the user's biometrics
        and stated goal.

        Parameters
        ----------
        user_profile : dict
            Raw user profile dict (same schema as database/users.json).
        calculated_calories : int | float
            The kcal/day figure produced by calculate_daily_calories().

        Returns
        -------
        ComplianceReport
            status APPROVED  — no violations found
            status WARNING   — technically within range but worth flagging
            status REJECTED  — hard safety limit breached; plan must not proceed
        """
        issues:      list[str] = []
        suggestions: list[str] = []

        kcal   = float(calculated_calories)
        bio    = user_profile.get("biometrics", {})
        goals  = user_profile.get("goals", {})
        gender = bio.get("gender", "").strip().lower()

        # --- 1. Gender-specific calorie floor (REJECTED) ---
        min_kcal = (
            _MIN_CALORIES_MALE   if gender == "male"   else
            _MIN_CALORIES_FEMALE if gender == "female" else
            _MIN_CALORIES_DEFAULT
        )
        if kcal < min_kcal:
            issues.append(
                f"UNSAFE: {kcal:.0f} kcal/day is below the clinical minimum of "
                f"{min_kcal} kcal/day for {gender or 'this user'}."
            )
            suggestions.append(
                f"Raise the daily target to at least {min_kcal} kcal. "
                "Consider reducing the weekly deficit rate instead."
            )

        # --- 2. Weekly rate of change (REJECTED for loss; WARNING for gain) ---
        target         = goals.get("target", "maintain_weight")
        weekly_rate_kg = float(goals.get("weekly_rate_kg", 0))

        if target == "lose_weight" and weekly_rate_kg > _MAX_WEEKLY_LOSS_KG:
            issues.append(
                f"UNSUSTAINABLE: A loss rate of {weekly_rate_kg} kg/week exceeds "
                f"the WHO safe maximum of {_MAX_WEEKLY_LOSS_KG} kg/week."
            )
            suggestions.append(
                f"Reduce the weekly loss target to ≤ {_MAX_WEEKLY_LOSS_KG} kg/week "
                "(approx. 550 kcal/day deficit) to preserve lean muscle mass."
            )

        if target == "gain_weight" and weekly_rate_kg > _MAX_WEEKLY_GAIN_KG:
            issues.append(
                f"WARNING: A gain rate of {weekly_rate_kg} kg/week exceeds "
                f"{_MAX_WEEKLY_GAIN_KG} kg/week — surplus may be stored as fat."
            )
            suggestions.append(
                f"Cap the weekly surplus at {_MAX_WEEKLY_GAIN_KG} kg/week "
                "(approx. 275 kcal/day surplus) for lean bulk quality."
            )

        # --- 3. Calorie target below BMR (WARNING — not always wrong) ---
        bmr_approx = self._estimate_bmr(bio)
        if bmr_approx and kcal < bmr_approx and kcal >= min_kcal:
            issues.append(
                f"WARNING: Target of {kcal:.0f} kcal is below estimated BMR "
                f"({bmr_approx:.0f} kcal). Extended periods below BMR can slow "
                "metabolism and cause micronutrient deficiencies."
            )
            suggestions.append(
                "Unless under direct clinical supervision, avoid sustained intake "
                "below BMR. A moderate deficit from TDEE is safer long-term."
            )

        # --- Determine final status ---
        status = self._resolve_status(issues)

        return ComplianceReport(
            status=status,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "checked_at":         datetime.now(timezone.utc).isoformat(),
                "guidelines":         "USDA Dietary Guidelines 2020-2025 / WHO",
                "calculated_calories": kcal,
                "min_safe_calories":  min_kcal,
                "weekly_rate_kg":     weekly_rate_kg,
                "goal":               target,
            },
        )

    # ------------------------------------------------------------------
    # verify_meal_plan
    # ------------------------------------------------------------------

    def verify_meal_plan(
        self,
        meal_plan_data: dict[str, Any],
        dietary_framework: str = "default",
    ) -> ComplianceReport:
        """
        Audit a full day's nutrient totals against macro-balance and
        micronutrient safety thresholds.

        Parameters
        ----------
        meal_plan_data : dict
            Expected keys (all values in grams unless noted):
                total_calories   (kcal)   — required
                protein_g                 — required
                carbs_g                   — required
                fat_g                     — required
                saturated_fat_g           — optional
                sodium_mg                 — optional
                sugar_g                   — optional
        dietary_framework : str
            The user's dietary pattern; used to contextualise carb levels.
            One of: "keto", "paleo", "vegan", "vegetarian", "default".

        Returns
        -------
        ComplianceReport
        """
        issues:      list[str] = []
        suggestions: list[str] = []

        framework = dietary_framework.strip().lower()
        kcal      = float(meal_plan_data.get("total_calories", 0))

        if kcal <= 0:
            return ComplianceReport(
                status=STATUS_REJECTED,
                issues=["Meal plan data is missing 'total_calories' or value is zero."],
                suggestions=["Provide a valid total_calories value before running compliance."],
            )

        protein_g = float(meal_plan_data.get("protein_g", 0))
        carbs_g   = float(meal_plan_data.get("carbs_g",   0))
        fat_g     = float(meal_plan_data.get("fat_g",     0))

        # --- 1. Protein within IOM AMDR (10–35 % of total calories) ---
        protein_kcal = protein_g * _KCAL_PER_G["protein"]
        protein_pct  = protein_kcal / kcal

        if protein_pct < _PROTEIN_PCT_MIN:
            issues.append(
                f"LOW PROTEIN: Protein supplies {protein_pct:.1%} of calories "
                f"({protein_g:.0f} g). IOM minimum is {_PROTEIN_PCT_MIN:.0%}."
            )
            suggestions.append(
                f"Increase protein to at least {kcal * _PROTEIN_PCT_MIN / 4:.0f} g "
                f"({_PROTEIN_PCT_MIN:.0%} of {kcal:.0f} kcal) to meet AMDR."
            )
        elif protein_pct > _PROTEIN_PCT_MAX:
            issues.append(
                f"HIGH PROTEIN: Protein supplies {protein_pct:.1%} of calories "
                f"({protein_g:.0f} g). IOM upper limit is {_PROTEIN_PCT_MAX:.0%}."
            )
            suggestions.append(
                f"Reduce protein to ≤ {kcal * _PROTEIN_PCT_MAX / 4:.0f} g "
                f"({_PROTEIN_PCT_MAX:.0%} of {kcal:.0f} kcal) or redistribute "
                "calories toward healthy fats or complex carbohydrates."
            )

        # --- 2. Carbohydrate level vs dietary-framework reference band ---
        band = _FRAMEWORK_CARB_BANDS.get(framework, _FRAMEWORK_CARB_BANDS["default"])

        if framework == "keto" and carbs_g > band["ceiling"]:
            issues.append(
                f"KETO VIOLATION: {carbs_g:.0f} g of carbs exceeds the "
                f"{band['ceiling']} g ceiling required to maintain ketosis."
            )
            suggestions.append(
                f"Reduce net carbs to ≤ {band['target']} g to sustain ketosis. "
                "Swap high-carb items for non-starchy vegetables or healthy fats."
            )
        elif carbs_g < band["low"] and framework not in ("keto",):
            issues.append(
                f"LOW CARBS for {framework.capitalize()}: {carbs_g:.0f} g is below "
                f"the recommended minimum of {band['low']} g for this dietary framework."
            )
            suggestions.append(
                f"Add more complex carbohydrates (target ≥ {band['low']} g) to "
                "support energy levels and adequate dietary fibre intake."
            )

        # --- 3. Sodium (USDA: ≤ 2,300 mg/day) ---
        sodium_mg = meal_plan_data.get("sodium_mg")
        if sodium_mg is not None:
            sodium_mg = float(sodium_mg)
            if sodium_mg > _MAX_SODIUM_MG:
                issues.append(
                    f"HIGH SODIUM: {sodium_mg:.0f} mg exceeds the USDA limit of "
                    f"{_MAX_SODIUM_MG} mg/day. Excess sodium raises blood pressure."
                )
                suggestions.append(
                    "Replace processed or cured foods with fresh alternatives. "
                    "Use herbs/spices instead of salt at cooking and the table."
                )

        # --- 4. Saturated fat (USDA: < 10 % of daily calories) ---
        sat_fat_g = meal_plan_data.get("saturated_fat_g")
        if sat_fat_g is not None:
            sat_fat_g    = float(sat_fat_g)
            sat_fat_kcal = sat_fat_g * _KCAL_PER_G["sat_fat"]
            sat_fat_pct  = sat_fat_kcal / kcal
            if sat_fat_pct > _MAX_SAT_FAT_PCT:
                issues.append(
                    f"HIGH SATURATED FAT: {sat_fat_g:.0f} g ({sat_fat_pct:.1%} of "
                    f"calories) exceeds the USDA cap of {_MAX_SAT_FAT_PCT:.0%}."
                )
                suggestions.append(
                    "Replace saturated fat sources (butter, full-fat dairy, fatty "
                    "red meat) with unsaturated alternatives: olive oil, avocado, "
                    "nuts, or oily fish."
                )

        # --- 5. Added / free sugar (WHO: < 10 % of daily calories) ---
        sugar_g = meal_plan_data.get("sugar_g")
        if sugar_g is not None:
            sugar_g    = float(sugar_g)
            sugar_kcal = sugar_g * _KCAL_PER_G["sugar"]
            sugar_pct  = sugar_kcal / kcal
            if sugar_pct > _MAX_SUGAR_PCT:
                issues.append(
                    f"HIGH SUGAR: {sugar_g:.0f} g ({sugar_pct:.1%} of calories) "
                    f"exceeds the WHO free-sugar limit of {_MAX_SUGAR_PCT:.0%}."
                )
                suggestions.append(
                    "Reduce added-sugar sources (sweetened drinks, desserts, sauces). "
                    "WHO notes additional benefit when free sugars fall below 5 % of calories."
                )
            elif sugar_pct > _IDEAL_SUGAR_PCT_UPPER:
                issues.append(
                    f"ELEVATED SUGAR: {sugar_g:.0f} g ({sugar_pct:.1%} of calories). "
                    "Within the WHO hard limit but above the 5 % ideal threshold."
                )
                suggestions.append(
                    "Aim to reduce free sugars below 5 % of total calories for "
                    "optimal metabolic health (WHO recommendation)."
                )

        status = self._resolve_status(issues)

        return ComplianceReport(
            status=status,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "checked_at":        datetime.now(timezone.utc).isoformat(),
                "guidelines":        "USDA Dietary Guidelines 2020-2025 / WHO",
                "dietary_framework": framework,
                "total_calories":    kcal,
                "carb_band_used":    band,
                "protein_pct":       round(protein_pct, 4),
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_status(issues: list[str]) -> str:
        """
        Derive the overall status from the accumulated issue strings.

        REJECTED — any issue prefixed with "UNSAFE", "UNSUSTAINABLE", or
                   a hard-limit keyword (keto violation, low protein, etc.)
        WARNING  — at least one issue but none are hard violations
        APPROVED — no issues
        """
        if not issues:
            return STATUS_APPROVED

        rejected_keywords = (
            "UNSAFE", "UNSUSTAINABLE", "KETO VIOLATION",
            "HIGH SODIUM", "HIGH SATURATED FAT", "HIGH SUGAR",
            "HIGH PROTEIN", "LOW PROTEIN",
        )
        for issue in issues:
            for kw in rejected_keywords:
                if kw in issue:
                    return STATUS_REJECTED

        return STATUS_WARNING

    @staticmethod
    def _estimate_bmr(biometrics: dict[str, Any]) -> float | None:
        """
        Estimate BMR using the Mifflin-St Jeor equation.
        Returns None if required biometric fields are missing.
        """
        w = biometrics.get("weight_kg")
        h = biometrics.get("height_cm")
        a = biometrics.get("age")
        g = biometrics.get("gender", "").strip().lower()

        if None in (w, h, a) or not g:
            return None

        if g == "male":
            return (10 * w) + (6.25 * h) - (5 * a) + 5
        return (10 * w) + (6.25 * h) - (5 * a) - 161


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    checker = ComplianceChecker()

    # --- Test 1: unsafe calorie target for a female user ---
    test_profile = {
        "user_id": "student_02",
        "biometrics": {
            "age": 30, "weight_kg": 65, "height_cm": 170,
            "gender": "female", "activity_level": 1.375,
        },
        "goals": {"target": "lose_weight", "weekly_rate_kg": 1.5},
    }
    print("=== Test 1: Daily Goal Validation ===")
    report = checker.verify_daily_goal(test_profile, calculated_calories=1050)
    print(json.dumps(report.to_dict(), indent=2))

    # --- Test 2: Keto meal plan with a carb violation ---
    print("\n=== Test 2: Keto Meal Plan — Carb Violation ===")
    meal = {
        "total_calories": 2100,
        "protein_g":      131,
        "carbs_g":        80,     # too high for keto
        "fat_g":          163,
        "saturated_fat_g": 24,
        "sodium_mg":      2100,
        "sugar_g":        12,
    }
    report2 = checker.verify_meal_plan(meal, dietary_framework="keto")
    print(json.dumps(report2.to_dict(), indent=2))

    # --- Test 3: Clean balanced plan — should be APPROVED ---
    print("\n=== Test 3: Balanced Plan — Expect APPROVED ===")
    clean_meal = {
        "total_calories": 2000,
        "protein_g":      130,
        "carbs_g":        220,
        "fat_g":          60,
        "saturated_fat_g": 16,
        "sodium_mg":      1800,
        "sugar_g":        35,
    }
    report3 = checker.verify_meal_plan(clean_meal, dietary_framework="default")
    print(json.dumps(report3.to_dict(), indent=2))
