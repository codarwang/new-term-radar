"""Can a new site get onto page one? The other half of "do it or not".

Rising is not enough: a term whose first page is held by strong sites is a fight a new site
loses. For the table's rows that would be acted on (prioritize, then watch), up to `max_rows`
a run, this reads the free browser-tool audit — ONE webtool.kd() call per term carries the
whole picture: page-1 results with inline DR + monthly visits, the difficulty score, keyword
volume, and the backlink budget. There is no paid lane: when the audit can't run the row
simply gets no competition block and the error is named in the report.

`open`: three or more weak results, or difficulty under 20. `crowded`: no weak result and a
difficulty of 40 or more, or unknown. `mixed` in between. A crowded `prioritize` becomes
`watch`: still rising, but page one belongs to strong sites. In a name's first week page one
is still forming (news, the maker's own pages): read it again at the second reading.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import datasrc, webtool

VIDEO = ("youtube.com", "tiktok.com", "vimeo.com")
FORUM = ("reddit.com", "quora.com", "ycombinator.com", "stackoverflow.com", "stackexchange.com", "x.com", "facebook.com")
TOOL = re.compile(r"\b(generator|maker|editor|converter|creator|online|free|app|tool|detector|remover|upscaler|translator)\b", re.I)
LIST = re.compile(r"\b(best|top|alternatives?|vs\.?|compared?|review(?:s|ed)?)\b|\b\d+\s+\w+", re.I)


def page_kind(domain: str, url: str, title: str) -> str:
    """tool, list, article, video or forum, from the address and the title."""
    if domain.endswith(VIDEO):
        return "video"
    if domain.endswith(FORUM):
        return "forum"
    if LIST.search(title):
        return "list"
    path = re.sub(r"^https?://[^/]+", "", url).strip("/")
    return "tool" if TOOL.search(title) or not path else "article"


def verdict(weak: int | None, kd: int | None) -> str:
    if (weak is not None and weak >= 3) or (kd is not None and kd < 20):
        return "open"
    if (weak in (0, None)) and (kd is None or kd >= 40):
        return "crowded"
    return "mixed"


def _assess_web(picked: list[dict], c: dict) -> list[str]:
    """One webtool.kd() call per term carries the entire audit — page-1 results
    with inline DR + monthly visits, difficulty score, keyword volume, link
    budget. Rows that arrive already audited (volumes() stashed `_audit` on the
    way through) reuse it — one call, two consumers. `paid` advertisers is the
    one signal the audit does not carry; it stays absent honestly."""
    errors = []
    for r in picked:
        audit = r.pop("_audit", None)
        if audit is None:
            try:
                audit = webtool.kd(r["phrase"], datasrc._gl(c["location_code"]))
            except webtool.PoolExhausted:
                raise   # the shared pool is dead — stop, don't fake rows
            except webtool.WebToolError as exc:
                errors.append(f"{r['phrase']}: {exc}")
                continue
        rated = [{"domain": x["domain"], "url": x["url"], "title": x["title"],
                  "position": x["position"], "dr": x["dr"], "visits": x["visits"],
                  "kind": page_kind(x["domain"], x["url"], x["title"])}
                 for x in audit["results"]]
        known = [x for x in rated if x["dr"] is not None]
        weak = sum(1 for x in known if x["dr"] < c["weak_dr"]) if known else None
        weakest = min(known, key=lambda x: (x["dr"], x["position"])) if known else None
        kd = round(audit["score"]) if isinstance(audit.get("score"), (int, float)) else None
        kinds = {}
        for x in rated:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        r["competition"] = {"verdict": verdict(weak, kd), "weak": weak, "kd": kd,
                            "ads": None, "kinds": kinds, "features": [],
                            "results": rated,
                            "weakest": {k: weakest[k] for k in ("domain", "dr", "position", "url")} if weakest else None,
                            # kw_hit homepage ≈ someone built a dedicated site on
                            # the term — the intitle "built" signal's free twin
                            "built": sum(1 for x in audit["results"] if x.get("kw_hit") and x.get("is_homepage")),
                            "link_budget": audit["link_budget"], "level": audit["level"],
                            "audit_volume": audit["volume"], "audit_reasons": audit["reasons"],
                            "source": "webcafe_web"}
        if not r.get("volume") and audit.get("volume"):
            r["volume"] = audit["volume"]     # free volume backfill — costs 0 extra
        if r["decision"] == "prioritize" and r["competition"]["verdict"] == "crowded":
            r["decision"], r["demoted"] = "watch", True
    return errors


def assess(rows: list[dict], cfg: dict, state: Path) -> tuple[int, list[str]]:
    """Adds `competition` to the rows that would be acted on and demotes a
    crowded prioritize. Free lane only — cost is always 0. (cost, errors)"""
    c = cfg["competition"]
    picked = [r for r in rows if r.get("decision") == "prioritize"] + [r for r in rows if r.get("decision") == "watch"]
    picked = picked[:c["max_rows"]]
    if not picked:
        return 0, []
    return 0, _assess_web(picked, c)
