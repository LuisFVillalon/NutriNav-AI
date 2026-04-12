"""
agentic_orchestrator.py -- Self-Correcting Multi-Agent Workflow
===============================================================
Implements a 3-attempt feedback loop between the NutritionistAgent and
the ChefAgent so that the final meal plan is guaranteed to pass (or
have made its best effort to pass) clinical compliance checks.

Workflow
--------
1. NutritionistAgent  ->  deterministic calorie / macro targets
2. ComplianceChecker  ->  GATE: reject dangerously low targets before the Chef is called
3. format_instructions_for_chef()  ->  structured constraint block passed to Chef
4. ChefAgent (up to 3 attempts):
     a. Chef generates meal plan + [JSON_START]...[JSON_END] compliance block
     b. Orchestrator parses the JSON block
     c. ComplianceChecker.verify_meal_plan() audits the parsed totals
     d. If APPROVED  -> done
     e. If REJECTED  -> NutritionistAgent.generate_feedback_note() -> re-route to Chef
5. If all 3 attempts fail -> output best attempt with a clinical disclaimer

Data contract (Nutritionist -> Chef)
--------------------------------------
NutritionistAgent.get_daily_targets() returns:
    {
        "user_id":              str,
        "dietary_framework":    str,
        "bmr":                  float,
        "maintenance_tdee":     float,
        "daily_calorie_target": int,
        "macros":               {"protein_g": int, "carbs_g": int, "fat_g": int},
        "compliance":           ComplianceReport.to_dict(),
    }

NutritionistAgent.format_instructions_for_chef() renders that dict into a
human-readable constraint block.  ChefAgent accepts it via its
``nutritionist_constraints`` parameter and injects it verbatim into the
LLM user prompt, so there is a single source of truth for what the Chef sees.

The Chef's LLM response MUST include a [JSON_START]...[JSON_END] block
whose "daily_totals" sub-dict is parsed by ChefAgent.parse_structured_output()
and fed back into ComplianceChecker.verify_meal_plan().
"""

from __future__ import annotations

import sys
from pathlib import Path

# When run directly (python agents/agentic_orchestrator.py) the interpreter
# puts the agents/ directory on sys.path, making "from agents.xxx" fail.
# Insert the project root so both invocation styles work:
#   python agents/agentic_orchestrator.py          (direct)
#   python -m agents.agentic_orchestrator          (module)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

from agents.nutritionist_agent import NutritionistAgent
from agents.chef_agent import ChefAgent
from agents.visionary_agent import VisionaryAgent

load_dotenv()

MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class NutriNavError(Exception):
    """Base exception for all NutriNav pipeline errors."""


class ProfileLoadError(NutriNavError):
    """Raised when a user profile JSON file cannot be read or parsed."""


class ComplianceGateError(NutriNavError):
    """
    Raised when the Nutritionist's calorie target fails the compliance gate
    (i.e. the goal itself is clinically unsafe before the Chef is involved).
    """
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class ChefGenerationError(NutriNavError):
    """
    Raised when the Chef fails to produce a compliant plan within MAX_ATTEMPTS
    AND raises an exception (as opposed to returning a warning with a disclaimer).
    Not raised in the normal fallback path — only used for unexpected failures.
    """


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Manages the full NutriNav meal-plan pipeline:

        NutritionistAgent -> ComplianceGate -> ChefAgent (self-correcting loop)
                                                         -> save .md to outputs/

    Usage (async)
    -------------
        orch   = Orchestrator(api_key="...", model="gpt-4o")
        result = await orch.run("user_profiles/student_01.json")

    Usage (sync convenience)
    ------------------------
        result = Orchestrator.run_sync("user_profiles/student_01.json")
    """

    def __init__(
        self,
        api_key:    str | None = None,
        model:      str | None = None,
        output_dir: str        = "outputs",
    ) -> None:
        """
        Parameters
        ----------
        api_key : str | None
            OpenAI API key; falls back to OPENAI_API_KEY env var.
        model : str | None
            LLM model ID; falls back to OPENAI_MODEL env var, then "gpt-4o".
        output_dir : str
            Directory where finished meal plans are written.  Created
            automatically if it does not exist.
        """
        self._api_key    = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model      = model   or os.getenv("OPENAI_MODEL",   "gpt-4o")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Async client — used for all LLM calls inside the pipeline
        self._async_client = openai.AsyncOpenAI(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, profile_path: str) -> dict[str, Any]:
        """
        Execute the full self-correcting pipeline for one user profile.

        Parameters
        ----------
        profile_path : str
            Path to a single-user JSON profile file.

        Returns
        -------
        dict with keys:
            status      : "APPROVED" | "WARNING" | "REJECTED" | "GATE_REJECTED"
            meal_plan   : str   (final meal plan text, with disclaimer if needed)
            output_path : str   (absolute path to the saved .md file)
            attempts    : int   (how many Chef LLM calls were made)
            compliance  : dict  (final ComplianceReport as dict)

        Raises
        ------
        ProfileLoadError
            If the profile JSON cannot be opened or parsed.
        ComplianceGateError
            If the nutritionist's calorie target is clinically unsafe.
        """
        # ------------------------------------------------------------------
        # STEP 1: Load profile + run NutritionistAgent (deterministic, no LLM)
        # ------------------------------------------------------------------
        _log("Orchestrator", f"Loading profile: {profile_path}")

        profile = _load_profile(profile_path)   # raises ProfileLoadError on failure

        nutritionist = NutritionistAgent(profile_path)
        targets      = nutritionist.get_daily_targets()

        _log(
            "Nutritionist",
            f"{targets['user_id']} | {targets['daily_calorie_target']} kcal | "
            f"{targets['dietary_framework'].capitalize()}",
        )

        # ------------------------------------------------------------------
        # STEP 2: Compliance gate — block dangerously low calorie targets
        # ------------------------------------------------------------------
        goal_compliance = targets["compliance"]
        goal_status     = goal_compliance["status"]

        if goal_status == "REJECTED":
            _log("Orchestrator", "COMPLIANCE GATE: Initial calorie goal is REJECTED.")
            for issue in goal_compliance["issues"]:
                _log("  [XX]", issue)
            for hint in goal_compliance["suggestions"]:
                _log("  [->]", hint)
            _log("Orchestrator", "Halting workflow. Adjust the user profile and retry.")
            raise ComplianceGateError(
                f"Calorie target for {targets['user_id']} failed the compliance gate.",
                report=goal_compliance,
            )

        if goal_status == "WARNING":
            for issue in goal_compliance["issues"]:
                _log("Nutritionist [!!]", issue)

        # ------------------------------------------------------------------
        # STEP 3: Build Chef with pre-rendered Nutritionist constraint block
        # ------------------------------------------------------------------
        nutritionist_constraints = nutritionist.format_instructions_for_chef()
        _log("Nutritionist", "Formatted constraint block ready for Chef.")

        chef = ChefAgent(
            targets=targets,
            profile=profile,
            nutritionist_constraints=nutritionist_constraints,
        )

        # ------------------------------------------------------------------
        # STEP 4: Self-correcting feedback loop (max MAX_ATTEMPTS async calls)
        # ------------------------------------------------------------------
        messages: list[dict[str, str]] = [
            {"role": "system", "content": chef.system_prompt},
            {"role": "user",   "content": chef.build_user_prompt()},
        ]

        best_output:   str | None  = None
        best_report:   dict | None = None
        final_status:  str         = "FAILED"
        attempts_made: int         = 0

        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts_made = attempt
            _log(
                "Orchestrator",
                f"Attempt {attempt}/{MAX_ATTEMPTS} — sending prompt to {self._model}...",
            )

            try:
                raw_output = await self._chef_llm_call(messages)
            except openai.OpenAIError as exc:
                _log("Orchestrator [ERROR]", f"LLM call failed on attempt {attempt}: {exc}")
                if attempt == MAX_ATTEMPTS:
                    raise ChefGenerationError(
                        f"OpenAI API error on final attempt: {exc}"
                    ) from exc
                # Retry without appending anything to messages — the prompt is unchanged
                continue

            # ---- Parse the structured compliance block ----
            daily_totals = ChefAgent.parse_structured_output(raw_output)

            if daily_totals is None:
                _log(
                    f"Orchestrator [Attempt {attempt}]",
                    "Chef response missing [JSON_START]...[JSON_END] block.",
                )
                feedback = (
                    "NUTRITIONIST FEEDBACK -- Your response did not include the required "
                    "[JSON_START]...[JSON_END] compliance block.\n"
                    "Regenerate the COMPLETE meal plan and append the block at the end, "
                    "replacing all <integer> placeholders with real numbers."
                )
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user",      "content": feedback})
                best_output = raw_output   # keep as fallback even without the block
                continue

            # ---- Run ComplianceChecker on Chef's reported totals ----
            meal_report = nutritionist._checker.verify_meal_plan(
                daily_totals,
                dietary_framework=targets["dietary_framework"],
            )
            best_output = raw_output
            best_report = meal_report.to_dict()

            _log(
                f"Orchestrator [Attempt {attempt}]",
                f"Meal plan compliance: {meal_report.status}",
            )

            if meal_report.status == "APPROVED":
                final_status = "APPROVED"
                _log("Orchestrator", f"Plan APPROVED on attempt {attempt}. Loop complete.")
                break

            # ---- Generate feedback and continue the conversation ----
            for issue in meal_report.issues:
                _log(f"  Attempt {attempt} [!!]", issue)

            if attempt < MAX_ATTEMPTS:
                feedback_note = nutritionist.generate_feedback_note(meal_report.to_dict())
                _log(
                    "Orchestrator",
                    f"Attempt {attempt}: Routing Nutritionist feedback to Chef...",
                )
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user",      "content": feedback_note})
            else:
                final_status = meal_report.status
                _log(
                    "Orchestrator",
                    f"Maximum attempts ({MAX_ATTEMPTS}) reached. "
                    "Using best output with disclaimer.",
                )

        # ------------------------------------------------------------------
        # STEP 5: Attach clinical disclaimer when plan did not fully pass
        # ------------------------------------------------------------------
        disclaimer = ""
        if final_status != "APPROVED":
            remaining = "\n".join(
                f"  - {i}"
                for i in (best_report.get("issues", []) if best_report else [])
            )
            disclaimer = (
                f"\n\n{'=' * 60}\n"
                f"CLINICAL DISCLAIMER\n"
                f"This meal plan did not pass all compliance checks after "
                f"{MAX_ATTEMPTS} attempt(s).\n"
                f"Remaining issues:\n{remaining or '  (unknown)'}\n"
                f"Please consult a registered dietitian before following this plan.\n"
                f"{'=' * 60}"
            )

        final_output = (best_output or "") + disclaimer

        # ------------------------------------------------------------------
        # STEP 6: Save to outputs/<user_id>_meal_plan_<YYYYMMDD>.md
        # ------------------------------------------------------------------
        output_path = self._build_output_path(targets["user_id"])
        self._save_plan(output_path, final_output)
        _log("Orchestrator", f"Meal plan saved -> {output_path}")

        heading = (
            "APPROVED MEAL PLAN"
            if final_status == "APPROVED"
            else "MEAL PLAN (WITH DISCLAIMER)"
        )
        print(f"\n{'=' * 60}")
        print(f"  {heading}")
        print(f"{'=' * 60}\n")
        print(final_output)

        return {
            "status":      final_status,
            "meal_plan":   final_output,
            "output_path": str(output_path),
            "attempts":    attempts_made,
            "compliance":  best_report or {},
        }

    @classmethod
    def run_sync(
        cls,
        profile_path: str,
        api_key:      str | None = None,
        model:        str | None = None,
        output_dir:   str        = "outputs",
    ) -> dict[str, Any]:
        """
        Synchronous convenience wrapper around ``run()``.

        Suitable for scripts, tests, and the ``__main__`` block.
        """
        orch = cls(api_key=api_key, model=model, output_dir=output_dir)
        return asyncio.run(orch.run(profile_path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _chef_llm_call(self, messages: list[dict[str, str]]) -> str:
        """
        Send *messages* to the LLM asynchronously and return the reply text.

        Raises openai.OpenAIError on any API failure so the loop can decide
        whether to retry or surface the error.
        """
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content

    def _build_output_path(self, user_id: str) -> Path:
        """
        Construct a timestamped Markdown path inside ``output_dir``.

        Format: outputs/mealplan_<user_id>_<YYYYMMDD>.md
        """
        datestamp = datetime.now().strftime("%Y%m%d")
        filename  = f"mealplan_{user_id}_{datestamp}.md"
        return self._output_dir / filename

    def _save_plan(self, path: Path, content: str) -> None:
        """
        Write *content* to *path* as UTF-8 text.

        Raises NutriNavError (wrapping OSError) on I/O failure so callers
        get a consistent exception type.
        """
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise NutriNavError(f"Failed to write meal plan to {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Vision workflow
# ---------------------------------------------------------------------------

def run_vision_workflow(
    image_path:    str,
    api_key:       str | None = None,
    model:         str | None = None,
    extra_context: str        = "",
) -> str:
    """
    Vision pipeline: image -> VisionaryAgent -> structured calorie breakdown.

    Parameters
    ----------
    image_path : str
        Path to the food image (jpg / png / webp).
    api_key : str | None
        OpenAI API key; falls back to OPENAI_API_KEY env var.
    model : str | None
        Vision-capable model; falls back to OPENAI_MODEL env var, then "gpt-4o".
    extra_context : str
        Optional hint injected into the Visionary's user prompt.

    Returns
    -------
    str -- the structured calorie breakdown text.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    model   = model   or os.getenv("OPENAI_MODEL",   "gpt-4o")

    client = openai.OpenAI(api_key=api_key)

    _log("Orchestrator", f"Analysing image: {image_path}")
    visionary = VisionaryAgent(image_path, extra_context=extra_context)
    _log("Visionary", "Sending image to LLM...")
    result = visionary.analyze(client, model=model)

    print("\n--- VISUAL CALORIE ANALYSIS ---")
    print(result)
    return result


# ---------------------------------------------------------------------------
# Backwards-compatible function wrapper
# ---------------------------------------------------------------------------

def run_meal_plan_workflow(
    profile_path: str,
    api_key:      str | None = None,
    model:        str | None = None,
    output_path:  str        = "",      # ignored — path is now auto-generated
) -> dict[str, Any]:
    """
    Backwards-compatible synchronous entry point.

    Delegates to ``Orchestrator.run_sync()``.  The ``output_path`` parameter
    is accepted but ignored; the file is always written to outputs/ with an
    auto-generated timestamped name.

    Returns
    -------
    dict with keys: status, meal_plan, output_path, attempts, compliance
    """
    if output_path:
        _log(
            "Orchestrator [NOTICE]",
            f"output_path='{output_path}' is ignored; "
            "file is saved to outputs/ with a timestamped name.",
        )
    return Orchestrator.run_sync(profile_path, api_key=api_key, model=model)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_profile(profile_path: str) -> dict[str, Any]:
    """
    Read and parse a user profile JSON file.

    Raises
    ------
    ProfileLoadError
        If the file does not exist, is unreadable, or contains invalid JSON.
    """
    try:
        with open(profile_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise ProfileLoadError(f"Profile not found: {profile_path}")
    except json.JSONDecodeError as exc:
        raise ProfileLoadError(
            f"Profile is not valid JSON ({profile_path}): {exc}"
        ) from exc
    except OSError as exc:
        raise ProfileLoadError(
            f"Could not read profile ({profile_path}): {exc}"
        ) from exc


def _log(agent: str, message: str) -> None:
    """Print a labelled log line to stdout."""
    print(f"[{agent}] {message}")


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point.

    Examples
    --------
        python -m agents.agentic_orchestrator user_profiles/student_01.json
        python -m agents.agentic_orchestrator user_profiles/student_02.json --model gpt-4o-mini
    """
    parser = argparse.ArgumentParser(
        description="NutriNav-AI — generate a personalised meal plan for one user profile.",
    )
    parser.add_argument(
        "profile",
        help="Path to a user profile JSON file (e.g. user_profiles/student_01.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model ID (default: OPENAI_MODEL env var or gpt-4o)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for saved meal plans (default: outputs/)",
    )
    args = parser.parse_args()

    try:
        result = Orchestrator.run_sync(
            args.profile,
            model=args.model,
            output_dir=args.output_dir,
        )
    except ProfileLoadError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ComplianceGateError as exc:
        print(f"\n[COMPLIANCE GATE] {exc}", file=sys.stderr)
        print(f"Report: {json.dumps(exc.report, indent=2)}", file=sys.stderr)
        sys.exit(2)
    except ChefGenerationError as exc:
        print(f"\n[CHEF ERROR] {exc}", file=sys.stderr)
        sys.exit(3)
    except NutriNavError as exc:
        print(f"\n[NUTRINAV ERROR] {exc}", file=sys.stderr)
        sys.exit(4)

    print(f"\nFinal status  : {result['status']}")
    print(f"Attempts made : {result['attempts']}")
    print(f"Saved to      : {result['output_path']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
