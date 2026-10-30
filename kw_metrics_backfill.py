"""One-off: backfill kd + cpc for the existing demand_keywords corpus via the
GIFT-POOL tool lane — one webtool.kd audit plus one webtool.ideas lookup
per keyword, then emit UPDATE statements for D1. No paid API anywhere.

Usage: uv run python kw_metrics_backfill.py > /tmp/kw_metrics.sql
Reads /tmp/backfill-kws.json — [[keyword, keyword_slug], ...]
"""
import json
import os
import sys
import time

from radar import webtool

WANT_KD = os.environ.get("BF_KD", "1") != "0"    # kd already backfilled 87/87 —
WANT_CPC = os.environ.get("BF_CPC", "1") != "0"  # this run is BF_KD=0, cpc only


def main() -> int:
    kws = json.load(open("/tmp/backfill-kws.json"))
    print(f"# {len(kws)} keywords (kd={'on' if WANT_KD else 'off'}, cpc={'on' if WANT_CPC else 'off'})", file=sys.stderr)
    stmts = []
    dead = False
    for kw, _slug in kws:
        sets = []
        if WANT_KD:
            try:
                a = webtool.kd(kw, "us")
                if isinstance(a.get("score"), (int, float)):
                    sets.append(f"kd = {round(float(a['score']), 2)}")
            except (webtool.PoolExhausted, webtool.WalletSpend) as exc:
                print(f"# pool/wallet guard — stopping: {exc}", file=sys.stderr)
                dead = True
            except webtool.WebToolError as exc:
                print(f"  {kw}: kd: {exc}", file=sys.stderr)
        if WANT_CPC and not dead:
            try:
                x = webtool.ideas(kw, "us")
                if x and isinstance(x.get("cpc"), (int, float)):
                    sets.append(f"cpc = {round(float(x['cpc']), 2)}")
            except (webtool.PoolExhausted, webtool.WalletSpend) as exc:
                print(f"# pool/wallet guard — stopping: {exc}", file=sys.stderr)
                dead = True
            except webtool.WebToolError as exc:
                print(f"  {kw}: ideas: {exc}", file=sys.stderr)
        if sets:
            kw_sql = kw.replace("'", "''")
            stmts.append(f"UPDATE demand_keywords SET {', '.join(sets)} WHERE keyword = '{kw_sql}';")
            print(f"  {kw}: {' '.join(sets)}", file=sys.stderr)
        else:
            print(f"  no data: {kw}", file=sys.stderr)
        if dead:
            break
        time.sleep(0.5)   # the lane is free, not free-for-all
    print("\n".join(stmts))
    print(f"# {len(stmts)} UPDATEs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
