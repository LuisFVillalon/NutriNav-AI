# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NutriNav-AI is a Python-based multi-agent system that generates personalised, clinically-safe meal plans. It chains a deterministic nutrition calculator → an LLM-based chef → a compliance validator in a self-correcting feedback loop.

## Setup & Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a meal plan from the CLI
python -m agents.agentic_orchestrator user_profiles/student_01.json
python -m agents.agentic_orchestrator user_profiles/student_02.json --model gpt-4o-mini

# Smoke-test individual tools (no API key needed)
python tools/compliance_checker.py
python tools/unit_converter.py
```

**Environment:** Create a `.env` file and set `OPENAI_API_KEY`. Optionally set `USDA_API_KEY` and `OPENAI_MODEL` (defaults to `gpt-4o`). The `anthropic` SDK is installed in `requirements.txt` for future Claude-powered agents.

**Model requirement:** GPT-4o or Gemini 1.5 Pro minimum. Smaller models reliably fail at least one constraint (macros, framework rules, or allergy avoidance) in the Chef prompt. The Visionary Agent additionally requires multimodal/vision capability.

**Existing user profiles:** `user_profiles/student_01.json`, `student_02.json`, `student_03.json`

## Architecture

### Agent Pipeline

The `Orchestrator` class in `agents/agentic_orchestrator.py` runs this sequence:

```
User Profile JSON
    → NutritionistAgent       (deterministic BMR/TDEE math, no LLM)
    → ComplianceChecker        (GATE: raises ComplianceGateError if calorie target is unsafe)
    → ChefAgent                (async LLM call: generates 1-day meal plan)
    → ComplianceChecker        (validate meal plan macro totals from JSON block)
    → if REJECTED: inject feedback → retry ChefAgent (max 3 attempts)
    → save to outputs/mealplan_<user_id>_<YYYYMMDD>.md
```

There is also a separate vision pipeline via `run_vision_workflow()` in the same file:

```
Food image file
    → VisionaryAgent           (base64-encodes image, builds prompt)
    → OpenAI gpt-4o vision     (sync LLM call)
    → structured calorie/macro breakdown text
```

### Orchestrator Class

`Orchestrator` is the primary interface. Its `run()` method is async; `run_sync()` is a classmethod that calls `asyncio.run()`:

```python
# Async (preferred for integration)
orch   = Orchestrator(api_key="sk-...", model="gpt-4o", output_dir="outputs")
result = await orch.run("user_profiles/student_01.json")

# Sync convenience (scripts, tests, __main__)
result = Orchestrator.run_sync("user_profiles/student_01.json")
```

`run_meal_plan_workflow()` is kept as a backwards-compatible sync wrapper that delegates to `Orchestrator.run_sync()`. Its `output_path` parameter is accepted but ignored — paths are always auto-generated.

### Exception Hierarchy

```
NutriNavError          — base; all pipeline failures
  ProfileLoadError     — file missing / unreadable / invalid JSON
  ComplianceGateError  — calorie target clinically unsafe; carries .report dict
  ChefGenerationError  — OpenAI API error persisted through all retry attempts
```

`main()` catches each type and exits with a distinct status code (1–4).

### Agent Responsibilities

| Agent | File | LLM? | Role |
|-------|------|-------|------|
| NutritionistAgent | `agents/nutritionist_agent.py` | No | Mifflin-St Jeor BMR → TDEE → macro targets by dietary framework |
| ChefAgent | `agents/chef_agent.py` | Yes (async, gpt-4o) | Generates recipes hitting calorie/macro targets; mandatory `[JSON_START]...[JSON_END]` structured output |
| VisionaryAgent | `agents/visionary_agent.py` | Yes (sync, gpt-4o vision) | Analyses base64-encoded food photos to estimate calories/macros |
| Orchestrator | `agents/agentic_orchestrator.py` | — | Chains agents, drives retry loop, saves output |

### Key Design Invariants

- **All LLM prompts live in `agents/prompts.py`** as a static class with `AgentPrompts.render()`. Never hardcode prompts inside agent files.
- **ChefAgent requires structured output:** Every Chef LLM response must contain a `[JSON_START]...[JSON_END]` block with `daily_totals` (total_calories, protein_g, carbs_g, fat_g). `ChefAgent.parse_structured_output()` extracts it; without it, compliance checking fails and the retry loop injects feedback asking for the block.
- **Nutritionist → Chef data contract:** `get_daily_targets()` returns a targets dict; `format_instructions_for_chef()` renders it into a human-readable constraint block. The Orchestrator passes both to `ChefAgent.__init__()`. The constraint block is the single source of truth for what the LLM sees — do not duplicate it inline.
- **Nut allergy safety:** If a user profile lists peanut or tree nut allergies, the Chef prompt is modified to require an explicit `✔ Confirmed: This recipe is 100% nut-free.` line after each recipe section.

### Dietary Framework Resolution

`NutritionistAgent` uses a 3-tier fallback to resolve the `dietary_framework` field from the user profile:
1. Exact key match in `_MACRO_RATIOS` (12 frameworks + `"default"`)
2. Keyword scan via `_KEYWORD_MACRO_PATTERNS` (handles aliases like `"high-protein"` → `"high protein"`)
3. Falls back to `"default"` (30/40/30 protein/carbs/fat)

`ComplianceChecker.verify_meal_plan()` uses `_FRAMEWORK_CARB_BANDS` to apply framework-appropriate carb limits. Keto has a hard 50 g/day ceiling that triggers `REJECTED`; other frameworks use soft minimums that trigger `WARNING`.

### ComplianceChecker Two-Layer Validation

1. **Goal-level** (`verify_daily_goal`): checks calorie floor by gender (1,200 F / 1,500 M), weekly rate limits (loss ≤ 1 kg/wk → REJECTED; gain > 0.5 kg/wk → WARNING), and below-BMR warnings.
2. **Meal-level** (`verify_meal_plan`): checks IOM AMDR protein range (10–35%), framework carb band, sodium (≤ 2,300 mg), saturated fat (< 10% kcal), and sugar (< 10% kcal WHO limit).

Status determination: any issue string containing `UNSAFE`, `UNSUSTAINABLE`, `KETO VIOLATION`, `HIGH SODIUM`, `HIGH SATURATED FAT`, `HIGH SUGAR`, `HIGH PROTEIN`, or `LOW PROTEIN` → `REJECTED`; otherwise `WARNING`; no issues → `APPROVED`.

### Supporting Tools

- `tools/calculate_daily_calories.py` — pure Mifflin-St Jeor function called by NutritionistAgent; no LLM
- `tools/unit_converter.py` — `UnitConverter` class + `convert_units()` module function; `build_reference_block()` embeds a conversion table directly in Chef prompts
- `tools/nutrition_api.py` — optional USDA FoodData Central client used by `NutritionistAgent.validate_ingredient()`

### Data

- Individual profiles in `user_profiles/` are the runtime input to the pipeline
- Generated meal plans are saved to `outputs/mealplan_<user_id>_<YYYYMMDD>.md` (directory auto-created by Orchestrator)
- Food images for the Visionary Agent go in `uploads/`

## Constraint Types to Test

Three canonical constraint types cover the main compliance paths:
1. **Keto** — carbs ≤ 50 g hard ceiling; triggers `KETO VIOLATION` → `REJECTED` if exceeded
2. **Vegan** — plant protein combos; carbs ≥ 130 g soft minimum (WARNING if below)
3. **Allergy** — peanut/tree nut profiles require `✔ Confirmed: This recipe is 100% nut-free.` in every meal section; verified by `ChefAgent._has_nut_allergy()`
