"""
Verification test for the annotations auto-fill behavior in both
content/shared/gemini_writer._validate_script and content/shared/claude_writer._validate_script.

Covers:
  1. Missing 'annotations' → auto-filled to {} + warning logged (no raise)
  2. Missing some OTHER required key (e.g. 'ticker') → still raises
  3. Present-and-populated 'annotations' → left untouched
  4. Both gemini_writer and claude_writer behave identically

Run with:
    python3 tests/test_script_validator.py
or:
    python3 -m unittest tests.test_script_validator -v
"""
import logging
import sys
import unittest
from pathlib import Path

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The writer modules import paid SDKs at the top (vertexai / anthropic).
# Stub them so we can import the pure Python validator.
import types


def _stub_module(name: str):
    mod = types.ModuleType(name)
    mod.__path__ = []

    class _Dummy:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return _Dummy()

        def __getattr__(self, _):
            return _Dummy()

    for attr in (
        "GenerativeModel", "GenerationConfig", "Anthropic",
        "Part", "Credentials",
    ):
        setattr(mod, attr, _Dummy)
    sys.modules[name] = mod


for name in (
    "vertexai",
    "vertexai.generative_models",
    "anthropic",
):
    _stub_module(name)


# Now import both validators
from content.shared import gemini_writer  # noqa: E402
from content.shared import claude_writer  # noqa: E402


VALID_SEGMENT = {
    "topic": "Market open recap",
    "narration": "Short narration text.",
    "chart_spec": {
        "ticker": "SPY",
        "timeframe": "5min",
        "lookback": 78,
        "overlays": [],
        "annotations": {"entry": 553.80},
    },
    "on_screen_text": "SPY 5min",
    "duration_seconds": 60,
}


def _valid_script():
    """Return a minimal valid script dict (fresh copy each time)."""
    return {
        "title": "Test Report",
        "hook": "Hook line.",
        "segments": [dict(VALID_SEGMENT, chart_spec=dict(VALID_SEGMENT["chart_spec"]))],
        "outro": "Outro line.",
    }


class ValidatorBehaviorTests(unittest.TestCase):
    """Run the same tests against both validator modules."""

    # ──────────────────────────────────────────────────────────────
    # Missing annotations → auto-filled + warning
    # ──────────────────────────────────────────────────────────────

    def _assert_autofills_missing_annotations(self, module):
        script = _valid_script()
        # Remove the annotations key from segment 0
        del script["segments"][0]["chart_spec"]["annotations"]
        self.assertNotIn("annotations", script["segments"][0]["chart_spec"])

        # Capture the warning log
        with self.assertLogs(module.logger, level="WARNING") as cm:
            module._validate_script(script)

        # Annotations must now be an empty dict in-place
        self.assertEqual(script["segments"][0]["chart_spec"]["annotations"], {})

        # Warning line must mention 'annotations' and segment index 0
        warn_text = "\n".join(cm.output)
        self.assertIn("annotations", warn_text)
        self.assertIn("Segment 0", warn_text)

    def test_gemini_writer_autofills_missing_annotations(self):
        self._assert_autofills_missing_annotations(gemini_writer)

    def test_claude_writer_autofills_missing_annotations(self):
        self._assert_autofills_missing_annotations(claude_writer)

    # ──────────────────────────────────────────────────────────────
    # Missing some OTHER required key → still raises
    # ──────────────────────────────────────────────────────────────

    def _assert_still_strict_on_other_keys(self, module):
        script = _valid_script()
        del script["segments"][0]["chart_spec"]["ticker"]
        with self.assertRaises(module.ScriptValidationError) as ctx:
            module._validate_script(script)
        self.assertIn("ticker", str(ctx.exception))

    def test_gemini_writer_still_strict_on_other_keys(self):
        self._assert_still_strict_on_other_keys(gemini_writer)

    def test_claude_writer_still_strict_on_other_keys(self):
        self._assert_still_strict_on_other_keys(claude_writer)

    # ──────────────────────────────────────────────────────────────
    # Present annotations → left untouched
    # ──────────────────────────────────────────────────────────────

    def _assert_preserves_present_annotations(self, module):
        script = _valid_script()
        original = {"entry": 553.80, "stop": 552.50}
        script["segments"][0]["chart_spec"]["annotations"] = dict(original)
        module._validate_script(script)
        self.assertEqual(
            script["segments"][0]["chart_spec"]["annotations"], original
        )

    def test_gemini_writer_preserves_present_annotations(self):
        self._assert_preserves_present_annotations(gemini_writer)

    def test_claude_writer_preserves_present_annotations(self):
        self._assert_preserves_present_annotations(claude_writer)

    # ──────────────────────────────────────────────────────────────
    # Missing multiple fields at once — annotations auto-filled,
    # other missing fields still trigger the error
    # ──────────────────────────────────────────────────────────────

    def _assert_mixed_missing_raises_on_other(self, module):
        script = _valid_script()
        del script["segments"][0]["chart_spec"]["annotations"]
        del script["segments"][0]["chart_spec"]["lookback"]
        with self.assertRaises(module.ScriptValidationError) as ctx:
            module._validate_script(script)
        # The error should mention lookback, NOT annotations (annotations
        # was auto-filled before the strict check ran)
        err = str(ctx.exception)
        self.assertIn("lookback", err)
        self.assertNotIn("annotations", err)

    def test_gemini_writer_mixed_missing(self):
        self._assert_mixed_missing_raises_on_other(gemini_writer)

    def test_claude_writer_mixed_missing(self):
        self._assert_mixed_missing_raises_on_other(claude_writer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
