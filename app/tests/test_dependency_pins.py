"""Dependency-pin regression guard — hosted deployment failure 001 (2026-10-29).

Root cause this file prevents from recurring: the app pinned
``pg8000==1.30.5`` but left scramp unbounded (``scramp>=1.4.4``). pip resolved
scramp to 1.5.0, which REMOVED ``make_channel_binding()`` — an API that
pg8000 1.30.5 calls during SCRAM authentication at connection time. The
hosted app then failed on every Lakebase connection with::

    AttributeError: module 'scramp' has no attribute 'make_channel_binding'

Compatibility map (verified against PyPI metadata 2026-10-29):
  * pg8000 1.30.x  requires ``scramp>=1.4.4``  -> compatible with the 1.4 line
    ONLY; scramp 1.5.0 removed make_channel_binding.
  * pg8000 1.31.6  requires ``scramp==1.5.0``  -> the line adapted to scramp 1.5.

These tests enforce that the app's requirements.txt keeps a scramp pin that
is compatible with its pg8000 pin, so an unbounded scramp (or a mismatched
driver upgrade) fails CI instead of the hosted runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent
_REQUIREMENTS = _APP_DIR / "requirements.txt"


def _parse_requirements() -> dict[str, str]:
    """Return {package: requirement-spec} from the app's requirements.txt."""
    specs: dict[str, str] = {}
    for line in _REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", line)
        if m:
            specs[m.group(1).lower().replace("_", "-")] = m.group(2).strip()
    return specs


def _pinned_version(spec: str) -> str | None:
    """Exact version if the spec pins one (==X.Y.Z), else None."""
    m = re.search(r"==([0-9][^;\s]*)", spec)
    return m.group(1) if m else None


def test_requirements_file_exists():
    assert _REQUIREMENTS.is_file(), f"missing {_REQUIREMENTS}"


def test_scramp_is_exactly_pinned_not_bounded():
    """scramp must be an exact pin — a lower-only bound (>=1.4.4) lets pip
    resolve 1.5.0, which removes make_channel_binding (deployment failure 001)."""
    specs = _parse_requirements()
    assert "scramp" in specs, "scramp missing from requirements.txt entirely"
    spec = specs["scramp"]
    assert "==" in spec, (
        f"scramp requirement {spec!r} is not an exact pin; an unbounded/bounded-only "
        "spec can resolve to scramp 1.5.0, which is incompatible with pg8000 1.30.x "
        "(removed make_channel_binding — hosted deployment failure 001)"
    )


def test_pg8000_is_exactly_pinned():
    """pg8000 must be an exact pin so the scramp compatibility map stays valid."""
    specs = _parse_requirements()
    assert "pg8000" in specs, "pg8000 missing from requirements.txt"
    assert "==" in specs["pg8000"], (
        f"pg8000 requirement {specs['pg8000']!r} is not an exact pin; a floating "
        "driver version makes the scramp compatibility window unpredictable"
    )


def test_scramp_pin_compatible_with_pg8000_pin():
    """The known-good compatibility map between pg8000 and scramp lines.

    * pg8000 1.30.x  <->  scramp 1.4.x  (make_channel_binding API)
    * pg8000 1.31.x  <->  scramp 1.5.x  (redesigned channel-binding API)

    If either pin moves to a new line, this test fails until BOTH are updated
    together — exactly the failure mode that took down the hosted deployment.
    """
    specs = _parse_requirements()
    pg = _pinned_version(specs["pg8000"])
    sc = _pinned_version(specs["scramp"])
    assert pg and sc, "both pins must be exact for the compatibility check"
    pg_major_minor = tuple(int(x) for x in pg.split(".")[:2])
    sc_major_minor = tuple(int(x) for x in sc.split(".")[:2])
    if pg_major_minor == (1, 30):
        assert sc_major_minor == (1, 4), (
            f"pg8000 {pg} requires the scramp 1.4 line (scramp {sc} is incompatible: "
            "1.5.0 removed make_channel_binding, which pg8000 1.30.x calls at "
            "connection time)"
        )
    elif pg_major_minor == (1, 31):
        assert sc_major_minor == (1, 5), (
            f"pg8000 {pg} is the scramp-1.5-adapted line; scramp {sc} does not match"
        )
    else:
        pytest.fail(
            f"pg8000 {pg} is a NEW major/minor line with an UNKNOWN scramp "
            "compatibility window — update the compatibility map in this test "
            "after verifying against PyPI metadata, then update the pins."
        )


def test_installed_env_matches_pins_when_present():
    """Live guard: when the running environment ACTUALLY has the pinned
    pg8000+scramp installed (e.g. a hosted-like environment or the deploy
    build), the exact failing call path must resolve.

    In other environments (notebook sessions with different versions or none
    at all) this test skips with a reason — it never fabricates a pass."""
    specs = _parse_requirements()
    want_pg = _pinned_version(specs["pg8000"])
    want_sc = _pinned_version(specs["scramp"])

    try:
        import importlib.metadata as md
        have_pg = md.version("pg8000")
        have_sc = md.version("scramp")
    except md.PackageNotFoundError:
        pytest.skip("pg8000/scramp not installed in this environment (static pin checks above still ran)")

    if (have_pg, have_sc) != (want_pg, want_sc):
        pytest.skip(
            f"environment has pg8000 {have_pg} + scramp {have_sc}, not the pinned "
            f"{want_pg} + {want_sc}; hosted-runtime guard not applicable here"
        )

    import scramp  # noqa: E402
    assert callable(getattr(scramp, "make_channel_binding", None)), (
        "pinned scramp is installed but does not export make_channel_binding() — "
        "this is the exact hosted failure 001; the pin and the installed package "
        "disagree"
    )
    import pg8000.native  # noqa: E402,F401  — the failing import chain
