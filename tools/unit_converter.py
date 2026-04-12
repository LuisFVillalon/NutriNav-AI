"""
tools/unit_converter.py — Cooking Unit Conversion Tool
=======================================================
Provides metric ↔ imperial conversions for ingredient measurements.
Designed to be used by the ChefAgent when building ingredient lists
so every quantity is expressed in both unit systems.

Usage
-----
    from tools.unit_converter import UnitConverter, convert_units

    uc = UnitConverter()
    uc.convert(200, "g", "oz")      # → 7.05
    uc.dual_label(200, "g")         # → "200 g / 7.1 oz"

    convert_units(1, "cup", "ml")   # → 240.0  (module-level tool function)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Conversion table — all factors are (from_unit → to_unit) multipliers
# ---------------------------------------------------------------------------

# Weight
_G_PER_OZ  = 28.3495
_G_PER_LB  = 453.592
_G_PER_KG  = 1_000.0

# Volume
_ML_PER_TSP    = 4.92892
_ML_PER_TBSP   = 14.7868
_ML_PER_FL_OZ  = 29.5735
_ML_PER_CUP    = 240.0
_ML_PER_PINT   = 473.176
_ML_PER_LITRE  = 1_000.0

# Build a flat lookup: (normalised_from, normalised_to) → float multiplier
_CONVERSION_TABLE: dict[tuple[str, str], float] = {
    # ---------- weight ----------
    ("g",   "oz"):  1 / _G_PER_OZ,
    ("oz",  "g"):   _G_PER_OZ,
    ("g",   "lb"):  1 / _G_PER_LB,
    ("lb",  "g"):   _G_PER_LB,
    ("g",   "kg"):  1 / _G_PER_KG,
    ("kg",  "g"):   _G_PER_KG,
    ("kg",  "lb"):  _G_PER_KG / _G_PER_LB,
    ("lb",  "kg"):  _G_PER_LB / _G_PER_KG,
    ("oz",  "lb"):  1 / 16,
    ("lb",  "oz"):  16.0,
    ("kg",  "oz"):  _G_PER_KG / _G_PER_OZ,
    ("oz",  "kg"):  _G_PER_OZ / _G_PER_KG,
    # ---------- volume ----------
    ("ml",     "tsp"):    1 / _ML_PER_TSP,
    ("tsp",    "ml"):     _ML_PER_TSP,
    ("ml",     "tbsp"):   1 / _ML_PER_TBSP,
    ("tbsp",   "ml"):     _ML_PER_TBSP,
    ("ml",     "fl oz"):  1 / _ML_PER_FL_OZ,
    ("fl oz",  "ml"):     _ML_PER_FL_OZ,
    ("ml",     "cup"):    1 / _ML_PER_CUP,
    ("cup",    "ml"):     _ML_PER_CUP,
    ("ml",     "pint"):   1 / _ML_PER_PINT,
    ("pint",   "ml"):     _ML_PER_PINT,
    ("l",      "ml"):     _ML_PER_LITRE,
    ("ml",     "l"):      1 / _ML_PER_LITRE,
    ("l",      "cup"):    _ML_PER_LITRE / _ML_PER_CUP,
    ("cup",    "l"):      _ML_PER_CUP   / _ML_PER_LITRE,
    ("l",      "fl oz"):  _ML_PER_LITRE / _ML_PER_FL_OZ,
    ("fl oz",  "l"):      _ML_PER_FL_OZ / _ML_PER_LITRE,
    ("cup",    "fl oz"):  _ML_PER_CUP   / _ML_PER_FL_OZ,
    ("fl oz",  "cup"):    _ML_PER_FL_OZ / _ML_PER_CUP,
    ("cup",    "tbsp"):   _ML_PER_CUP   / _ML_PER_TBSP,
    ("tbsp",   "cup"):    _ML_PER_TBSP  / _ML_PER_CUP,
    ("tbsp",   "tsp"):    _ML_PER_TBSP  / _ML_PER_TSP,
    ("tsp",    "tbsp"):   _ML_PER_TSP   / _ML_PER_TBSP,
    ("fl oz",  "tbsp"):   _ML_PER_FL_OZ / _ML_PER_TBSP,
    ("tbsp",   "fl oz"):  _ML_PER_TBSP  / _ML_PER_FL_OZ,
}

# Canonical imperial counterpart for each metric unit (used by dual_label)
_METRIC_TO_IMPERIAL: dict[str, str] = {
    "g":  "oz",
    "kg": "lb",
    "ml": "fl oz",
    "l":  "cup",
}


class UnitConverter:
    """
    Stateless converter for common cooking measurement units.

    All methods are pure functions — safe to share as a class-level
    singleton in agents.
    """

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert *value* from *from_unit* to *to_unit*.

        Parameters
        ----------
        value : float
        from_unit : str  e.g. "g", "oz", "ml", "cup", "tbsp"
        to_unit   : str

        Returns
        -------
        float — converted value, rounded to 2 decimal places

        Raises
        ------
        ValueError  if the conversion pair is not supported
        """
        from_unit = from_unit.strip().lower()
        to_unit   = to_unit.strip().lower()

        if from_unit == to_unit:
            return round(float(value), 2)

        key = (from_unit, to_unit)
        if key not in _CONVERSION_TABLE:
            supported = ", ".join(f"{a}→{b}" for a, b in sorted(_CONVERSION_TABLE))
            raise ValueError(
                f"No conversion defined for '{from_unit}' → '{to_unit}'. "
                f"Supported pairs: {supported}"
            )

        return round(float(value) * _CONVERSION_TABLE[key], 2)

    def dual_label(self, value: float, metric_unit: str) -> str:
        """
        Return a dual-unit label string for ingredient lists, e.g.
        ``dual_label(200, "g")  →  "200 g / 7.1 oz"``

        Only works for metric input units that have a defined imperial
        counterpart (g, kg, ml, l).  For all other units, returns the
        metric value only.

        Parameters
        ----------
        value       : float   The numeric quantity in the metric unit.
        metric_unit : str     One of: "g", "kg", "ml", "l".
        """
        metric_unit = metric_unit.strip().lower()
        imperial_unit = _METRIC_TO_IMPERIAL.get(metric_unit)

        if imperial_unit is None:
            return f"{value} {metric_unit}"

        imperial_value = self.convert(value, metric_unit, imperial_unit)

        # Format: suppress unnecessary trailing zeros
        metric_str   = f"{value:g} {metric_unit}"
        imperial_str = f"{imperial_value:g} {imperial_unit}"
        return f"{metric_str} / {imperial_str}"

    def build_reference_block(self) -> str:
        """
        Build a compact unit-conversion reference table for embedding
        in LLM prompts so the model knows what scale to use.

        Returns a multi-line string.
        """
        rows = [
            ("Weight",  [("50 g",  "g",  "oz"),
                          ("100 g", "g",  "oz"),
                          ("200 g", "g",  "oz"),
                          ("500 g", "g",  "oz"),
                          ("1 kg",  "kg", "lb")]),
            ("Volume",  [("60 ml",   "ml",  "fl oz"),
                          ("120 ml",  "ml",  "fl oz"),
                          ("240 ml",  "ml",  "fl oz"),
                          ("1 cup",   "cup", "ml"),
                          ("1 tbsp",  "tbsp","ml"),
                          ("1 tsp",   "tsp", "ml")]),
        ]

        lines = ["UNIT CONVERSION REFERENCE (use for all ingredient measurements):"]
        for category, conversions in rows:
            lines.append(f"  {category}:")
            for label, from_u, to_u in conversions:
                numeric = float(label.split()[0])
                converted = self.convert(numeric, from_u, to_u)
                lines.append(f"    {label:>8}  =  {converted:g} {to_u}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level tool function (matches the interface requested in the task)
# ---------------------------------------------------------------------------

_default_converter = UnitConverter()


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert *value* from *from_unit* to *to_unit*.

    This is the module-level tool function intended for import by agents.
    Internally delegates to a shared UnitConverter instance.

    Example
    -------
        convert_units(200, "g", "oz")    # → 7.05
        convert_units(1,   "cup", "ml")  # → 240.0
    """
    return _default_converter.convert(value, from_unit, to_unit)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uc = UnitConverter()

    tests = [
        (200,   "g",    "oz"),
        (500,   "g",    "oz"),
        (1,     "kg",   "lb"),
        (240,   "ml",   "fl oz"),
        (1,     "cup",  "ml"),
        (1,     "tbsp", "ml"),
        (1,     "tsp",  "ml"),
    ]

    print("=== Unit Conversion Tests ===")
    for value, from_u, to_u in tests:
        result = uc.convert(value, from_u, to_u)
        print(f"  {value} {from_u:>6}  →  {result} {to_u}")

    print("\n=== Dual Labels ===")
    for value, unit in [(200, "g"), (1000, "ml"), (1.5, "kg"), (0.5, "l")]:
        print(f"  {uc.dual_label(value, unit)}")

    print(f"\n{uc.build_reference_block()}")
