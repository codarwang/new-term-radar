"""Shared data-source helpers.

Keyword metrics no longer go through a paid provider layer: volume, trends,
SERP and KD all ride the free browser-tool lane (webtool.*). `call()` remains
as a loud stub because aisa.call() still delegates here when no router
credentials exist — any post_dataforseo_* tool call reaching it is a bug, and
failing loud beats silently routing a metric to a billable endpoint.
"""
from __future__ import annotations

from .aisa import AisaError


def call(calls: list[dict], provider: str | None = None) -> tuple[dict, int]:
    raise AisaError(
        "no data provider: keyword metrics moved to the free webtool lane "
        "(webtool.kd/series/pair_series) — a post_dataforseo_* call reaching "
        "this stub means a call site was not rewired")


def _gl(location_code) -> str:
    """DataForSEO numeric location → web.cafe's `gl` country code."""
    if not location_code:
        return "us"
    gl = {2840: "us", 2826: "gb", 2124: "ca", 2036: "au", 2276: "de",
          2392: "jp", 2702: "sg"}.get(location_code)
    if gl is None:
        # a real-but-unmapped code silently pulling US data is worse than a
        # loud error — extend the map when adding geos
        raise AisaError(f"unmapped location_code {location_code!r} (extend _gl)")
    return gl
