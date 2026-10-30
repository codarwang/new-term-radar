"""seo.web.cafe's browser-facing tool endpoints — the zero-cost data lane.

Every tool page on the site ("/trends/", "/kd/", "/serp/", "/ideas/", "/mine/")
embeds a one-shot page token in a <meta name="{tool}-token"> tag; the page's own
JS sends it back as an X-{Tool}-Token header when calling the tool's JSON
endpoint. Calls ride the account's daily GIFTED pool — 10 credits anonymous,
30 logged-in, 500 VIP — shared across all tools, resetting 08:00 Beijing time.
Each call bills its normal credit price against the pool (cache hit 2, live
trends 4, +4 for related), so VIP ≈ 60–200 calls/day depending on mix.

Authenticate by exporting the site session cookie once (DevTools → Network →
copy the Cookie header of any api/* request) into WEBCAFE_COOKIE; without it the
client is anonymous and shares a 10-call IP pool with every other visitor of
this machine's egress — fine for a smoke test, not for a pipeline.

MONEY-SAFETY RAIL: when the gifted pool runs out, the site silently starts
charging the paid wallet ("赠送已满，正在用积分余额"). Two guards prevent the
radar from ever spending a cent on this lane:
  1. quota() is consulted before every call and the client refuses to fire when
     the call's worst-case charge could cross what the pool has left;
  2. every response's `charged` field is reconciled against the remaining pool —
     a charge that exceeds it means wallet money moved; the run aborts loudly.

HTTP is pinned to 1.1: several developer-network setups (local VPN/proxy with
fake-IP DNS) mangle h2 bodies to this host, returning empty responses. urllib
speaks 1.1 anyway; the note is for anyone porting this to a smarter client.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse

from . import UA

HOST = urllib.parse.urlparse(
    os.environ.get("WEBCAFE_SITE_URL", "https://seo.web.cafe")).hostname or "seo.web.cafe"
BASE_PATH = urllib.parse.urlparse(
    os.environ.get("WEBCAFE_SITE_URL", "https://seo.web.cafe")).path.rstrip("/")
MIN_INTERVAL = 6.5           # polite human pace; also the paid API's req/min cap
TIMEOUT = 180                # live audits (kd) can take a minute-plus
TRIES = 2
# the gift pool is credit-denominated; a single call bills 2–8 (cache hit →
# live+related). 10 is the worst plausible charge — conservative pre-checks
# cost nothing at VIP headroom and keep anon runs from tipping into the wallet
EST_MAX_CHARGE = 10
UA_BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# tool page path → (meta token name, header name)
TOOLS = {
    "kd":     ("kd-token", "X-KD-Token"),
    "trends": ("gt-token", "X-GT-Token"),
    "serp":   ("sr-token", "X-SR-Token"),
    "ideas":  ("id-token", "X-ID-Token"),
    "mine":   ("mn-token", "X-MN-Token"),
    "kgr":    ("kgr-token", "X-KGR-Token"),
}

_EGO_BIN = shutil.which("ego-browser")
_transport: str | None = None    # resolved once per process: "ego" | "http"


def _use_ego() -> bool:
    """Transport pick, resolved once. ego-browser is the preferred lane — the
    real browser carries the user's own logged-in session, real TLS/JS
    fingerprint, and needs no exported cookie. RADAR_WEB_TRANSPORT=http|ego
    pins the choice explicitly; auto falls back to HTTP (cookie, else anonymous)
    when the binary is absent."""
    global _transport
    if _transport:
        return _transport == "ego"
    forced = os.environ.get("RADAR_WEB_TRANSPORT", "auto")
    if forced == "http":
        _transport = "http"
    elif forced == "ego" or (forced == "auto" and _EGO_BIN):
        _transport = "ego" if _EGO_BIN else "http"
    else:
        _transport = "http"
    return _transport == "ego"


class WebToolError(RuntimeError):
    pass


class WalletSpend(WebToolError):
    """A call charged the paid wallet — the pool was exhausted. Abort the run."""


class PoolExhausted(WebToolError):
    """quota() reported used >= limit before the call was even attempted."""


_last_call = 0.0
_gate_lock = threading.Lock()         # callers fan out on ThreadPoolExecutor
_tokens: dict[str, str] = {}          # tool → page token
_quota_cache: tuple[float, dict] | None = None


def cookie() -> str:
    return os.environ.get("WEBCAFE_COOKIE", "").strip()


def available() -> bool:
    """True when a session cookie is configured. Anonymous calls still work for
    smoke tests (10/day, IP-shared) — callers decide whether that counts."""
    return bool(cookie())


def _gate() -> None:
    global _last_call
    with _gate_lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _request(method: str, path: str, body: dict | None = None,
             headers: dict | None = None) -> tuple[int, str]:
    """One HTTPS request, HTTP/1.1. Returns (status, body text)."""
    _gate()
    h = {"user-agent": UA_BROWSER, "accept": "application/json, text/html",
         "referer": f"https://{HOST}{BASE_PATH or '/'}"}
    if cookie():
        h["cookie"] = cookie()
    if headers:
        h.update(headers)
    conn = http.client.HTTPSConnection(HOST, timeout=TIMEOUT)
    try:
        conn.request(method, path,
                     body=json.dumps(body) if body is not None else None,
                     headers=h)
        res = conn.getresponse()
        return res.status, res.read().decode("utf-8", "replace")
    finally:
        conn.close()


def _ego_request(tool: str, method: str, path: str,
                 body: dict | None = None, params: dict | None = None,
                 reload: bool = False) -> tuple[int, str]:
    """One request executed inside the real Chromium page. The tab navigates to
    the tool page (once per tool switch — the space persists across calls), the
    page's own JS context reads its meta token and issues the fetch, so session
    cookie, token lifecycle and TLS fingerprint are all the site's native
    ones. `taskSpace` is shared per machine — sequential radar runs reuse the
    same tab; concurrent runs would interleave navigation on one tab, so the
    sweep and the daily chain must not overlap (they never do)."""
    meta, header = TOOLS[tool]
    page_url = f"https://{HOST}{BASE_PATH}/{tool}/"
    if params:
        path += "?" + urllib.parse.urlencode(params)
    script = f"""
const task = await taskSpace("webcafe-lane");
const page = task.page("p1");
const PAGE_URL = {json.dumps(page_url)};
const here = await page.evaluate(() => location.origin + location.pathname + "/");
if ({str(reload).lower()} || !here.startsWith(PAGE_URL)) {{ await page.goto(PAGE_URL); }}
const r = await page.evaluate(async (a) => {{
  const token = document.querySelector('meta[name="' + a.meta + '"]')?.content || "";
  const opt = {{method: a.method, cache: "no-store",
                headers: {{[a.header]: token, "Content-Type": "application/json"}}}};
  if (a.body != null) opt.body = JSON.stringify(a.body);
  const res = await fetch(a.path, opt);
  return {{status: res.status, text: await res.text()}};
}}, {{meta: {json.dumps(meta)}, header: {json.dumps(header)},
    method: {json.dumps(method)}, path: {json.dumps(path)},
    body: {json.dumps(body)}}});
console.log("EGO_RESULT:" + JSON.stringify(r));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        r = subprocess.run([_EGO_BIN, "nodejs"], input=script.encode(),
                           capture_output=True, timeout=TIMEOUT + 60)
        # ego-browser prints console.log on stderr, diagnostics on stdout —
        # scan both for the result marker
        for line in (r.stdout + r.stderr).decode("utf-8", "replace").splitlines():
            if line.startswith("EGO_RESULT:"):
                d = json.loads(line[len("EGO_RESULT:"):])
                return d["status"], d["text"]
        raise WebToolError(f"ego-browser: no result "
                           f"(stderr: {r.stderr.decode('utf-8', 'replace')[:200]})")
    except subprocess.TimeoutExpired:
        raise WebToolError(f"ego-browser: timeout after {TIMEOUT + 60}s")
    finally:
        os.unlink(fname)


def _page_token(tool: str) -> str:
    """GET the tool page, pull its embedded page token. Cached until a call 403s."""
    meta, _ = TOOLS[tool]
    status, html = _request("GET", f"{BASE_PATH}/{tool}/")
    if status != 200:
        raise WebToolError(f"{tool} page: HTTP {status}")
    m = re.search(rf'name="{meta}" content="([^"]+)"', html)
    if not m:
        raise WebToolError(f"{tool} page: no {meta} meta tag")
    _tokens[tool] = m.group(1)
    return _tokens[tool]


def quota(force: bool = False) -> dict:
    """{"used", "limit", "tier", ...} from api/me. Cached for the process unless
    force — the pool only moves when we move it."""
    global _quota_cache
    if _quota_cache and not force and time.monotonic() - _quota_cache[0] < 30:
        return _quota_cache[1]
    if _use_ego():
        status, text = _ego_request("kd", "GET", f"{BASE_PATH}/kd/api/me")
    else:
        status, text = _request("GET", f"{BASE_PATH}/kd/api/me")
    if status != 200:
        raise WebToolError(f"api/me: HTTP {status}")
    data = (json.loads(text).get("quota") or {})
    if cookie() and data.get("tier") == "游客":
        # a configured cookie that still reads anonymous means the session
        # expired — loud, because 10/day IP-shared is not the deal we made
        raise WebToolError("WEBCAFE_COOKIE expired — api/me still reports 游客")
    _quota_cache = (time.monotonic(), data)
    return data


def _call(tool: str, method: str, path: str,
          body: dict | None = None, params: dict | None = None) -> dict:
    """Authenticated tool call with quota + wallet guards and token refresh."""
    q = quota()
    used, limit = q.get("used", 0), q.get("limit", 0)
    # the pool is CREDIT-denominated (a live trends call bills 4–8 credits):
    # refuse to fire when the call's plausible worst case would cross the line —
    # whatever the pool doesn't cover is pulled from the paid wallet
    if used + EST_MAX_CHARGE > limit:
        raise PoolExhausted(f"gifted pool low: {used}/{limit} credits "
                            f"({q.get('tier')}) — resets 08:00 CST")
    _, header = TOOLS[tool]
    token = "" if _use_ego() else (_tokens.get(tool) or _page_token(tool))
    last_error = ""
    for attempt in range(TRIES):
        if _use_ego():
            status, text = _ego_request(tool, method, path, body, params,
                                        reload=attempt > 0)
        else:
            full = path + ("?" + urllib.parse.urlencode(params) if params else "")
            status, text = _request(method, full, body,
                                    {header: token, "content-type": "application/json"})
        if status == 403:
            if _use_ego():                  # ego reads the token from the live
                continue                    # DOM each call — reload handles it
            token = _page_token(tool)       # page tokens rotate; refetch once
            continue
        if status != 200:
            last_error = f"HTTP {status}: {text[:160]}"
            continue
        try:
            data = json.loads(text)
        except ValueError:
            last_error = f"non-JSON response: {text[:120]}"
            continue
        if data.get("error"):
            code = str(data.get("code") or "")
            if code == "token":
                token = _page_token(tool)
                continue
            raise WebToolError(f"{path}: {str(data.get('error'))[:160]}")
        # `charged` is the credits this call consumed — from the gift pool while
        # it lasts, then silently from the wallet. Any charge beyond the pool's
        # remaining headroom is real money: abort loudly, never "cost of doing
        # business". This is the rail that makes the lane safe to run unattended.
        charged = data.get("charged")
        if charged not in (None, 0, "0", "0.0"):
            try:
                spent = float(charged)
            except (TypeError, ValueError):
                spent = 0.0
            if spent > max(0.0, float(limit) - float(used)):
                raise WalletSpend(f"{path} charged {charged} credits but only "
                                  f"{float(limit) - float(used):.0f} of the gift "
                                  f"pool was left — wallet drained; aborting")
            used += spent
            if _quota_cache:
                _quota_cache[1]["used"] = used
        return data
    raise WebToolError(last_error or f"{path}: all {TRIES} tries failed")


def kd(keyword: str, gl: str = "us") -> dict:
    """哥飞版 KD audit in ONE free call: difficulty score/level, monthly volume,
    trend, the whole page-1 SERP with per-domain DR + visits + page type, and the
    backlink budget. Replaces serp+KD+DR×10+intitle of the paid bundle."""
    d = _call("kd", "GET", f"{BASE_PATH}/kd/api/kd",
              params={"keyword": keyword, "gl": gl})
    details = []
    for r in d.get("details") or []:
        if not isinstance(r, dict):
            continue
        details.append({
            "position": r.get("position"),
            "domain": re.sub(r"^www\.", "", str(r.get("domain") or "").lower()),
            "url": r.get("url") or "",
            "title": r.get("title") or "",
            "page_type": r.get("pageType") or "",
            "is_homepage": bool(r.get("isHomepage")),
            "dr": r.get("dr"),
            "visits": r.get("visits"),
            "kw_hit": bool(r.get("kwHit")),
            "strength": r.get("strength"),
        })
    return {"keyword": keyword, "score": d.get("score"), "level": d.get("level"),
            "volume": d.get("keywordVolume"), "trend": d.get("keywordTrend") or {},
            "link_budget": d.get("linkBudget") or {}, "reasons": d.get("reasons") or [],
            "type": d.get("keywordType"), "cache_hit": d.get("cacheHit") or "",
            "computed_at": d.get("computedAt"), "results": details}


def trends(keywords: list[str], range: str = "30d", geo: str = "US",
           related: bool = False) -> dict:
    """Google Trends via the free tool: relative-interest points plus, for a
    single keyword with related=True, rising/top queries with growth values.
    Up to 5 keywords share one call and one 0-100 scale (pair comparisons)."""
    d = _call("trends", "POST", f"{BASE_PATH}/trends/api/query",
              body={"keywords": keywords[:5], "range": range, "geo": geo,
                    "related": related and len(keywords) == 1})
    return {"keywords": d.get("keywords") or keywords, "geo": d.get("geo"),
            "range": d.get("range"), "granularity": d.get("granularity"),
            "source": d.get("source"), "points": d.get("points") or [],
            "related": d.get("related") or {}, "insight": d.get("insight") or "",
            "live_today": d.get("liveToday")}


def series(keyword: str, geo: str = "US", range: str = "30d") -> list[tuple]:
    """[(date, value)] — drop-in for goanyapi.daily_series."""
    from datetime import date as _date
    d = trends([keyword], range=range, geo=geo)
    out = []
    for p in d["points"]:
        try:
            v = p.get("value", (p.get("values") or [0])[0])
            out.append((_date.fromisoformat(str(p.get("time") or "")[:10]), int(v)))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def pair_series(keyword: str, benchmark: str, range: str = "90d", geo: str = "") -> dict:
    """{"solo": [(date, v)], "pair": [(date, v, bench_v)]} from ONE free call —
    the web tool normalizes all keywords in a query to a shared 0-100 scale, so
    [keyword, benchmark] in one shot is exactly the old solo+pair pair of calls.
    solo renormalized to its own top reproduces the term-alone curve (Trends
    scaling is linear), pair rides the shared scale as before."""
    from datetime import date as _date
    d = trends([keyword, benchmark], range=range, geo=geo)
    solo_raw, pair = [], []
    for p in d["points"]:
        try:
            vals = p.get("values") or []
            if len(vals) < 2:                    # single-keyword shape fallback
                vals = [p.get("value", 0), 0]
            day = _date.fromisoformat(str(p.get("time") or "")[:10])
            solo_raw.append((day, int(vals[0] or 0)))
            pair.append((day, int(vals[0] or 0), int(vals[1] or 0)))
        except (TypeError, ValueError, IndexError):
            continue
    top = max((v for _, v in solo_raw), default=0) or 1
    return {"solo": [(day, round(v * 100 / top)) for day, v in solo_raw],
            "pair": pair}


def related_queries(term: str, geo: str = "US") -> dict[str, list[str]]:
    """{"rising": [...], "top": [...]} — drop-in for goanyapi.related_queries,
    richer: the web tool returns growth numbers and Breakout flags too, but the
    normalized shape keeps the paid-path contract."""
    d = trends([term], range="12m", geo=geo, related=True)
    norm = lambda xs: [" ".join(str(x.get("query", "")).lower().split())
                       for x in xs if isinstance(x, dict)]  # noqa: E731
    return {"rising": norm(d["related"].get("rising") or []),
            "top": norm(d["related"].get("top") or [])}


def ideas(keyword: str, gl: str = "us") -> dict | None:
    """Ads-snapshot metrics for ONE keyword through the ideas expander —
    the lane's only CPC/bids source (kd audits don't carry them).
    {volume, cpc, low_bid, high_bid, competition, competition_level,
    kd_prescreen, intent, months, trend} or None when the planner library
    doesn't know the word at all (the honest answer, same as kd's gaps).

    The tool expands seeds into its ~2M-keyword monthly Ads snapshot; the
    seed itself comes back as a plain item whenever the library covers it,
    so one seed per call + exact-match is the reliable lookup shape (multi-
    seed calls dilute the per-seed budget and silently drop exacts). `kd`
    here is a PRE-SCREEN score on a different scale from kd() audits — the
    tool's own note says海选 only, don't compare across the two scales."""
    d = _call("ideas", "POST", f"{BASE_PATH}/ideas/api/ideas",
              body={"seeds": [keyword], "mode": "suggestions", "gl": gl,
                    "limit": 5, "minVolume": 0, "maxKd": None})
    want = " ".join(keyword.lower().split())
    for x in d.get("items") or []:
        if not isinstance(x, dict):
            continue
        if " ".join(str(x.get("keyword") or "").lower().split()) == want:
            return {"volume": x.get("volume"), "cpc": x.get("cpc"),
                    "low_bid": x.get("lowBid"), "high_bid": x.get("highBid"),
                    "competition": x.get("competition"),
                    "competition_level": x.get("competitionLevel"),
                    "kd_prescreen": x.get("kd"), "intent": x.get("intent"),
                    "months": x.get("months") or [], "trend": x.get("trend") or {}}
    return None
