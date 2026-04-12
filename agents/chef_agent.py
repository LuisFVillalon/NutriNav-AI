"""
agents/chef_agent.py — Chef Agent
==================================
Generates an original 1-day meal plan via an LLM, constrained by the
targets produced by NutritionistAgent.

Key behaviours
--------------
  * Invents recipes from scratch -- no database or API lookups.
  * Accepts pre-formatted Nutritionist constraints via nutritionist_constraints
    parameter so format_instructions_for_chef() output is used directly.
  * Every ingredient measurement is expressed in metric AND imperial units
    (via tools.unit_converter.UnitConverter).
  * Allergy safety block is injected into the prompt; nut allergies trigger
    an explicit per-recipe confirmation line.
  * A mandatory Macro Verification Table + [JSON_START]...[JSON_END] compliance
    block enables the Orchestrator to run ComplianceChecker.verify_meal_plan()
    on the Chef's output and drive the self-correcting feedback loop.
"""

from __future__ import annotations

import json as _json
import re
from typing import Any

from agents.prompts import AgentPrompts
from tools.unit_converter import UnitConverter

# Allergen keywords that trigger the per-recipe nut-free confirmation line
_NUT_ALLERGEN_KEYWORDS = frozenset({
    "peanut", "peanuts", "nut", "nuts", "tree nut", "tree nuts",
    "almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
    "pecan", "pecans", "pistachio", "pistachios", "hazelnut", "hazelnuts",
    "macadamia", "brazil nut", "brazil nuts",
})


class ChefAgent:
    """
    Creative Controller that turns structured nutritional targets into an
    original 1-day meal plan via an LLM call.

    Prompt templates live in agents/prompts.py — change the Chef's rules
    or tone by editing that file only.

    Parameters
    ----------
    targets : dict
        Output of NutritionistAgent.get_daily_targets().
        Must contain: daily_calorie_target, dietary_framework, macros
        (protein_g / carbs_g / fat_g).
    profile : dict
        Raw user profile dict so the Chef can access preferences and
        allergies.

    Usage
    -----
        chef   = ChefAgent(targets, profile)
        plan   = chef.generate_meal_plan(openai_client)
        prompt = chef.build_user_prompt()   # inspect before paying for API
    """

    # Shared converter — stateless, safe as a class-level singleton
    _converter = UnitConverter()

    # Class-level reference so callers can read the system prompt without
    # instantiating (useful in tests and the orchestrator)
    SYSTEM_PROMPT: str = AgentPrompts.CHEF_SYSTEM_PROMPT

    def __init__(
        self,
        targets: dict[str, Any],
        profile: dict[str, Any],
        nutritionist_constraints: str = "",
    ) -> None:
        """
        Parameters
        ----------
        targets : dict
            Output of NutritionistAgent.get_daily_targets().
        profile : dict
            Raw user profile dict (preferences, allergies, etc.).
        nutritionist_constraints : str
            Pre-rendered output of NutritionistAgent.format_instructions_for_chef().
            When provided, this replaces the inline constraint block in the prompt,
            ensuring the Orchestrator's call to format_instructions_for_chef() is
            the single source of truth for what the Chef receives.
            Falls back to rendering from targets if empty.
        """
        self._targets                 = targets
        self._profile                 = profile
        self._nutritionist_constraints = nutritionist_constraints
        self._allergies: list[str]    = self._parse_allergies()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """Static system role — sourced from AgentPrompts."""
        return AgentPrompts.CHEF_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        """
        Render CHEF_USER_PROMPT with all dynamic values:
          * nutritionist_constraints — pre-formatted block from NutritionistAgent
          * user preferences (liked / disliked / allergies)
          * allergy safety block + per-recipe confirmation line
          * unit conversion reference table (from UnitConverter)
          * JSON compliance block specification

        Returns the completed prompt string ready to send to the LLM.
        """
        prefs  = self._profile.get("preferences", {})
        macros = self._targets["macros"]

        liked     = ", ".join(prefs.get("liked_foods",    [])) or "no preference"
        disliked  = ", ".join(prefs.get("disliked_foods", [])) or "none"
        allergies = ", ".join(self._allergies) if self._allergies else "none"

        # Use the pre-rendered nutritionist constraint block when available;
        # fall back to a minimal inline block if the Orchestrator did not provide one.
        constraints = self._nutritionist_constraints or (
            f"Daily Calorie Target: {self._targets['daily_calorie_target']} kcal\n"
            f"Protein: {macros['protein_g']} g | "
            f"Carbs: {macros['carbs_g']} g | "
            f"Fat: {macros['fat_g']} g"
        )

        allergy_block        = self._build_allergy_block()
        allergy_confirmation = self._build_allergy_confirmation()
        unit_reference       = "\n" + self._converter.build_reference_block()

        return AgentPrompts.render(
            AgentPrompts.CHEF_USER_PROMPT,
            dietary_framework=self._targets["dietary_framework"].capitalize(),
            nutritionist_constraints=constraints,
            daily_calorie_target=self._targets["daily_calorie_target"],
            protein_g=macros["protein_g"],
            carbs_g=macros["carbs_g"],
            fat_g=macros["fat_g"],
            liked_foods=liked,
            disliked_foods=disliked,
            allergies=allergies,
            allergy_block=allergy_block,
            allergy_confirmation=allergy_confirmation,
            unit_reference=unit_reference,
            json_output_spec=AgentPrompts.CHEF_JSON_OUTPUT_SPEC,
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def generate_meal_plan(self, client: Any, model: str = "gpt-4o") -> str:
        """
        Send the constructed prompt to the LLM and return the meal plan text.

        Parameters
        ----------
        client : openai.OpenAI
            An already-configured OpenAI (or compatible) client instance.
        model : str
            Model identifier; defaults to "gpt-4o".

        Returns
        -------
        str — the raw LLM response containing the full meal plan.
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self.build_user_prompt()},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Structured output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_structured_output(raw: str) -> dict[str, Any] | None:
        """
        Extract the daily_totals dict from the Chef's
        [JSON_START]...[JSON_END] compliance block.

        The block is required at the end of every Chef response so that
        ComplianceChecker.verify_meal_plan() can validate the output
        programmatically without parsing free-form text.

        Parameters
        ----------
        raw : str
            The full text response from the LLM.

        Returns
        -------
        dict  — the "daily_totals" sub-dict, e.g.
                {"total_calories": 2100, "protein_g": 131, ...}
        None  — if the block is absent or the JSON is malformed.
        """
        match = re.search(
            r"\[JSON_START\](.*?)\[JSON_END\]",
            raw,
            re.DOTALL,
        )
        if not match:
            return None

        try:
            data = _json.loads(match.group(1).strip())
            return data.get("daily_totals")
        except (ValueError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_allergies(self) -> list[str]:
        """Extract and normalise the allergy list from the user profile."""
        raw = self._profile.get("preferences", {}).get("allergies", [])
        # Filter out placeholder values like "none"
        return [a.strip() for a in raw if a.strip().lower() not in ("", "none", "n/a")]

    def _has_nut_allergy(self) -> bool:
        """Return True if any listed allergy maps to a nut allergen."""
        lowered = {a.lower() for a in self._allergies}
        return bool(lowered & _NUT_ALLERGEN_KEYWORDS)

    def _build_allergy_block(self) -> str:
        """
        Build the ALLERGY SAFETY NOTICE block injected near the top of the
        user prompt.  Returns an empty string when there are no allergies.
        """
        if not self._allergies:
            return ""

        allergy_list = "\n".join(f"    ✗ {a.capitalize()}" for a in self._allergies)
        lines = [
            "⚠  ALLERGY SAFETY NOTICE — STRICT AVOIDANCE REQUIRED",
            "   The following ingredients and ALL derivatives must be completely",
            "   excluded from every recipe, including hidden sources:",
            allergy_list,
        ]

        if self._has_nut_allergy():
            lines += [
                "",
                "   NUT ALLERGY DETECTED: After each recipe section you MUST",
                '   include the exact line: "✔ Confirmed: This recipe is 100% nut-free."',
            ]

        lines.append("")   # blank line before next section
        return "\n".join(lines) + "\n"

    def _build_allergy_confirmation(self) -> str:
        """
        Return the confirmation placeholder line that appears inside the
        per-meal template block.  The LLM is expected to fill this in.
        """
        if self._has_nut_allergy():
            return "✔ Confirmed: This recipe is 100% nut-free."
        if self._allergies:
            confirmed = " and ".join(a.lower() for a in self._allergies)
            return f"✔ Confirmed: This recipe contains no {confirmed}."
        return ""
