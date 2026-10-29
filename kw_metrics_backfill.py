"""One-off: backfill cpc (avgCpc) + kd for the existing demand_keywords corpus,
then emit UPDATE statements for D1. ~87 internal WebCafe kd calls at the 6.1s
gate — expect ~10 min runtime and a few hundred member credits.

Usage: WEBCAFE_TOKEN=... uv run python kw_metrics_backfill.py > /tmp/kw_metrics.sql
Reads /tmp/backfill-kws.json — [[keyword, keyword_slug], ...]
"""
import json
import sys

from radar import datasrc


def main() -> int:
    kws = json.load(open("/tmp/backfill-kws.json"))
    print(f"# {len(kws)} keywords", file=sys.stderr)

    data, spent = datasrc.call([{"call_id": "vol", "tool": datasrc.DFS_VOLUME, "arguments": {
        "body": [{"keywords": [k for k, _ in kws],
                  "location_code": 2840, "language_code": "en"}]}}])
    result = ((data.get("vol") or {}).get("tasks") or [{}])[0].get("result") or []
    cpc_of = {r["keyword"]: r.get("cpc") for r in result if r.get("keyword")}
    print(f"# volume: {len(cpc_of)} rows, {spent} micros", file=sys.stderr)

    data, spent = datasrc.call([{"call_id": "kd", "tool": datasrc.DFS_KD, "arguments": {
        "body": [{"keywords": [k for k, _ in kws],
                  "location_code": 2840, "language_code": "en"}]}}])
    items = ((((data.get("kd") or {}).get("tasks") or [{}])[0]
              .get("result") or [{}])[0].get("items") or [])
    kd_of = {i["keyword"]: (i.get("keyword_properties") or {}).get("keyword_difficulty")
             for i in items if i.get("keyword")}
    print(f"# kd: {len(kd_of)} rows, {spent} micros", file=sys.stderr)

    stmts = []
    for kw, _slug in kws:
        cpc, kd = cpc_of.get(kw), kd_of.get(kw)
        sets = []
        if isinstance(cpc, (int, float)):
            sets.append(f"cpc = {round(float(cpc), 4)}")
        if isinstance(kd, (int, float)):
            sets.append(f"kd = {round(float(kd), 2)}")
        if sets:
            stmts.append("UPDATE demand_keywords SET " + ", ".join(sets)
                         + f" WHERE keyword = '{kw.replace(chr(39), chr(39)*2)}';")
        else:
            print(f"  no data: {kw}", file=sys.stderr)
    print("\n".join(stmts))
    print(f"# {len(stmts)} UPDATEs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
