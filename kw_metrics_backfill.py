"""One-off: backfill kd (+ volume audit) for the existing demand_keywords
corpus via the FREE browser-tool lane — one webtool.kd audit per keyword,
then emit UPDATE statements for D1. No paid API anywhere.

Usage: uv run python kw_metrics_backfill.py > /tmp/kw_metrics.sql
Reads /tmp/backfill-kws.json — [[keyword, keyword_slug], ...]
"""
import json
import sys
import time

from radar import webtool


def main() -> int:
    kws = json.load(open("/tmp/backfill-kws.json"))
    print(f"# {len(kws)} keywords", file=sys.stderr)
    stmts = []
    for kw, _slug in kws:
        try:
            a = webtool.kd(kw, "us")
        except webtool.PoolExhausted as exc:
            print(f"# pool exhausted — stopping: {exc}", file=sys.stderr)
            break
        except webtool.WebToolError as exc:
            print(f"  {kw}: {exc}", file=sys.stderr)
            continue
        sets = []
        if isinstance(a.get("score"), (int, float)):
            sets.append(f"kd = {round(float(a['score']), 2)}")
        if sets:
            kw_sql = kw.replace("'", "''")
            stmts.append(f"UPDATE demand_keywords SET {', '.join(sets)} WHERE keyword = '{kw_sql}';")
            print(f"  {kw}: kd={a.get('score')} vol={a.get('volume')}", file=sys.stderr)
        else:
            print(f"  no data: {kw}", file=sys.stderr)
        time.sleep(0.5)   # the lane is free, not free-for-all
    print("\n".join(stmts))
    print(f"# {len(stmts)} UPDATEs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
