"""The radar's pure logic: name extraction, signal scoring, the store, the judge's parsing,
the review's verdict, the reports. Network calls are not exercised here."""
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from radar import __main__ as cli
from radar import breakout, config, gsc, i18n, judge, new_sites, notify, report, review, sitemaps, store
from radar.sources import base_name, novel_names
from radar.terms import names

NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)


# --- names(): what a post can name -------------------------------------------------

def test_names_keeps_versioned_model_names_whole():
    found = names("Introducing Claude Opus 5.5, our best model.")
    assert "claude opus 5.5" in found and "opus 5.5" in found


def test_names_drops_phrases_made_only_of_stop_words():
    assert "claude code" not in names("Try it in Claude Code today")


def test_names_takes_a_name_that_starts_the_sentence():
    assert "strata" in names("Strata now runs Qwen3.8-Flash-Next on a 12GB RTX 5070. Big day.")


def test_names_stops_at_the_end_of_a_sentence():
    assert not any("big" in n and "5070" in n for n in names("on a RTX 5070. Big day."))


def test_names_ignores_urls_and_handles():
    assert names("see https://Example.com/Foo by @SomeOne") == set()


def test_names_breaks_on_chinese_text_between_words():
    found = names("用 Opus 5.5 配音（Gemini 3.8 Flash TTS，我提供了 API Key）")
    assert "gemini 3.8 flash" in found and not any("tts" in n and "api" in n for n in found)


# --- ① X: only names the accounts have not used before ------------------------------

def _post(account, text, likes=0, hours_ago=2):
    return {"account": account, "text": text, "likes": likes, "url": "", "created": NOW - timedelta(hours=hours_ago)}


def test_x_reports_a_new_name_two_accounts_used():
    rows = novel_names([_post("a", "Strata is fast"), _post("b", "Tried Strata today")], {}, NOW)
    assert [r["term"] for r in rows] == ["strata"] and rows[0]["sources"] == ["a", "b"]
    assert rows[0]["note"] == "X: 2 accounts, 0 likes"


def test_x_skips_names_already_in_the_vocabulary():
    rows = novel_names([_post("a", "Strata is fast"), _post("b", "Tried Strata today")], {"strata": "2026-09-01"}, NOW)
    assert rows == []


def test_x_one_account_needs_likes_and_an_unmistakable_name():
    assert novel_names([_post("a", "Qwen3.9 is fast", likes=20)], {}, NOW) == []
    assert [r["term"] for r in novel_names([_post("a", "Qwen3.9 is fast", likes=500)], {}, NOW)] == ["qwen3.9"]
    # a plain capitalised word may only be the start of a sentence: one account is not enough
    assert novel_names([_post("a", "Strata is fast", likes=500)], {}, NOW) == []


def test_x_one_post_yields_at_most_three_names():
    post = _post("a", "New: Alpha1 Beta2 Gamma3 Delta4 Eps5 Zeta6 models", likes=900)
    assert len(novel_names([post], {}, NOW)) <= 3


def test_x_ignores_old_posts_and_retweets():
    old = [_post("a", "Strata is fast", hours_ago=48), _post("b", "Strata rocks", hours_ago=48)]
    rts = [_post("a", "RT @x: Strata is fast"), _post("b", "RT @x: Strata rocks")]
    assert novel_names(old, {}, NOW) == [] and novel_names(rts, {}, NOW) == []


# --- ① HF: copies of one model group under its base name -----------------------------

def test_hf_base_name_strips_quantisation_and_size():
    assert base_name("abenzerps/Qwen-Image-2.1-Uncensored-GGUF") == "qwen image 2.1"
    assert base_name("Qwen/Qwen3.8-Flash-Next-A6B-Instruct") == "qwen3.8 flash next"
    assert base_name("Lightricks/LTX-2.5") == "ltx 2.5"


def test_hf_reads_trending_models_and_spaces_and_merges_a_model_with_its_demo(tmp_path, monkeypatch):
    import radar.sources as src
    lists = {"models": [{"id": "old/Old-Model"}], "spaces": [{"id": "old/old-demo"}]}
    monkeypatch.setattr(src, "_trending", lambda what: lists[what])
    first = src.hf_new({}, tmp_path)
    assert first["first_run"] and first["terms"] == [] and first["models"] == 1 and first["spaces"] == 1
    lists["models"] = [{"id": "krea/Krea-2", "pipeline_tag": "text-to-image", "likes": 900}, {"id": "old/Old-Model"}]
    lists["spaces"] = [{"id": "krea/Krea-2-demo", "sdk": "gradio", "likes": 300}, {"id": "lighton/LightOnOCR-3-webgpu", "sdk": "static"}]
    later = src.hf_new({}, tmp_path)
    terms = {r["term"]: r for r in later["terms"]}
    assert set(terms) == {"krea 2", "lightonocr 3"} and not later["first_run"]
    assert terms["krea 2"]["examples"] == ["krea/Krea-2 (text-to-image, ♥900)", "krea/Krea-2-demo (gradio, ♥300)"]
    assert terms["lightonocr 3"]["note"] == "Hugging Face Spaces: new at #2"


# --- ② breakout: one repo taking off --------------------------------------------------

def _repo(name, stars, days_old):
    created = (NOW - timedelta(days=days_old)).isoformat().replace("+00:00", "Z")
    return {"full_name": name, "stargazers_count": stars, "created_at": created, "description": ""}


def test_breakout_flags_a_fast_repo_and_skips_a_slow_one():
    rows = breakout.score([_repo("Niko1221/Strata", 7734, 9), _repo("a/slow", 900, 12)], None, NOW)
    assert [r["repo"] for r in rows] == ["Niko1221/Strata"]


def test_breakout_flags_a_jump_since_yesterday_even_when_the_average_is_low():
    rows = breakout.score([_repo("a/late", 1500, 13)], {"a/late": 900}, NOW)
    assert rows and rows[0]["delta"] == 600


def test_breakout_variants_include_handle_and_display_name():
    row = {"owner": "Niko1221", "name": "Strata"}
    assert breakout.variants(row, "Niko") == ["strata", "niko1221 strata", "niko strata"]
    assert breakout.variants(row, "") == ["strata", "niko1221 strata"]


# --- ③ sitemaps: one entry in 17 languages counts once ---------------------------------

def test_locale_prefix_is_stripped():
    urls = ["https://lobehub.com/skills/x", "https://lobehub.com/zh/skills/x", "https://lobehub.com/zh-TW/skills/x"]
    assert {re.sub(sitemaps._LOCALE, r"\1", u) for u in urls} == {"https://lobehub.com/skills/x"}


def test_paths_are_not_mistaken_for_locales():
    assert re.sub(sitemaps._LOCALE, r"\1", "https://lobehub.com/mcp/foo") == "https://lobehub.com/mcp/foo"


def test_sitemap_diff_reports_a_word_new_urls_share(cfg, tmp_path, monkeypatch):
    cfg["sitemaps"] = {"site": {"index": "https://example.com/sitemap.xml", "pause": 0}}
    old = {f"https://example.com/tools/thing-{i}" for i in range(40)}
    new = old | {f"https://example.com/tools/hyperframes-{w}" for w in ("starter", "captions", "studio")}
    answers = iter([old, new])
    monkeypatch.setattr(sitemaps, "site_urls", lambda *a: next(answers))
    first = sitemaps.diff(cfg, tmp_path)
    assert first["sites"]["site"]["first_run"] and first["terms"] == []
    second = sitemaps.diff(cfg, tmp_path)
    assert second["sites"]["site"]["new"] == 3 and second["terms"][0]["term"] == "hyperframes"


def test_a_site_that_fails_keeps_its_snapshot(cfg, tmp_path, monkeypatch):
    cfg["sitemaps"] = {"site": {"index": "https://example.com/sitemap.xml", "pause": 0}}
    monkeypatch.setattr(sitemaps, "site_urls", lambda *a: {"https://example.com/a"})
    sitemaps.diff(cfg, tmp_path)

    def down(*a):
        raise OSError("503")
    monkeypatch.setattr(sitemaps, "site_urls", down)
    assert "error" in sitemaps.diff(cfg, tmp_path)["sites"]["site"]
    monkeypatch.setattr(sitemaps, "site_urls", lambda *a: {"https://example.com/a"})
    assert sitemaps.diff(cfg, tmp_path)["sites"]["site"]["new"] == 0


# --- ③ new sites (whois overview, weekly) -----------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures/whois_new_ai_sites.json"   # a real answer, 2026-10-06


def test_new_site_name_terms():
    assert new_sites.name_terms("ai-hailuo.com") == ["hailuo"]
    assert new_sites.name_terms("livephoto.video") == ["livephoto"]
    assert new_sites.name_terms("ai-video-tool.com") == []   # only stop words left


def test_new_sites_drop_domains_ranking_for_a_handful_of_terms(cfg, tmp_path, monkeypatch):
    answer = {"tasks": [{"result": [{"items": json.loads(FIXTURE.read_text())}]}]}
    monkeypatch.setattr(new_sites.aisa, "call_all", lambda calls: ({c["call_id"]: answer for c in calls}, 127000))
    out = new_sites.new_sites(cfg, tmp_path, date(2026, 10, 6))
    domains = [r["domain"] for r in out["sites"]]
    assert "ai-hailuo.com" in domains and "neonwin386.ai" not in domains and "oyo88.ai" not in domains
    assert all(r["new"] for r in out["sites"]) and "hailuo" in [t["term"] for t in out["terms"]]
    again = new_sites.new_sites(cfg, tmp_path, date(2026, 10, 13))
    assert not any(r["new"] for r in again["sites"]) and again["terms"] == []


def test_new_sites_query_needs_both_dates(cfg):
    body = new_sites._calls(cfg["new_sites"], date(2026, 10, 6))[0]["arguments"]["body"][0]
    fields = [f[0] for f in body["filters"] if isinstance(f, list)]
    assert fields[:2] == ["created_datetime", "first_seen"] and body["filters"][0][2].startswith("2026-07-08")


# --- ④ Search Console ---------------------------------------------------------------------

def test_gsc_keeps_queries_the_earlier_window_never_saw():
    recent = [{"query": "hyperframes skill", "page": "/a", "impressions": 4, "clicks": 1},
              {"query": "hyperframes skill", "page": "/b", "impressions": 3, "clicks": 0},
              {"query": "ppt skill", "page": "/c", "impressions": 90, "clicks": 9}]
    fresh = gsc.fresh_queries(recent, [{"query": "ppt skill"}])
    assert fresh == [{"query": "hyperframes skill", "impressions": 7, "clicks": 1, "page": "/a"}]


def test_gsc_is_off_without_a_site(cfg):
    assert gsc.gsc_new(cfg) == {"off": True}


# --- the name judge ------------------------------------------------------------------------

def test_judge_parses_scores_and_drops_what_it_cannot_use():
    answer = 'Here you go:\n[{"i": 0, "score": 0.9}, {"i": 1, "score": "0.1"}, {"i": 7, "score": 1}, {"i": 2, "score": 3}, {"x": 1}]'
    assert judge.parse(answer, 3) == {0: 0.9, 1: 0.1}
    assert judge.parse("sorry", 3) == {}


def test_judge_parse_tolerates_fences_and_object_form():
    # free models sometimes wrap the payload in a markdown fence
    fenced = '```json\n[{"i": 0, "score": 0.9}, {"i": 1, "score": 0.2}]\n```'
    assert judge.parse(fenced, 3) == {0: 0.9, 1: 0.2}
    # ...or answer in object form {"index": score} (observed in the wild)
    assert judge.parse('{"0": 0.9, "1": 0.2}', 3) == {0: 0.9, 1: 0.2}
    # stray brackets in prose must not swallow the real span
    mixed = 'scores [per run] look like [{i}] — actually: [{"i": 1, "score": 0.7}]'
    assert judge.parse(mixed, 3) == {1: 0.7}
    # strictness kept: out-of-range scores still dropped even in object form
    assert judge.parse('{"0": 1.5, "1": 0.4}', 3) == {1: 0.4}
    # wrapper envelopes — models that nest the array under a key must still parse
    assert judge.parse('{"scores": [{"i": 0, "score": 0.9}]}', 3) == {0: 0.9}
    assert judge.parse('{"data": {"results": [{"i": 1, "score": 0.3}]}}', 3) == {1: 0.3}
    # a leading non-payload fence must not hide a later ```json block
    two_fences = '```thinking\nlet me reason\n```\n```json\n[{"i": 0, "score": 0.7}]\n```'
    assert judge.parse(two_fences, 3) == {0: 0.7}
    # a draft score array — in a thinking fence or in prose — must not win
    # over the final payload (reasoning models deliberate before answering)
    assert judge.parse('```thinking\npreliminary: [{"i": 0, "score": 0.1}]\n```\n'
                       'final: [{"i": 0, "score": 0.9}]', 3) == {0: 0.9}
    assert judge.parse('preliminary: [{"i": 0, "score": 0.1}] — final: '
                       '[{"i": 0, "score": 0.9}]', 3) == {0: 0.9}


def test_judge_scores_rows_in_chunks(monkeypatch):
    rows = [{"term": f"t{i}", "examples": []} for i in range(judge.CHUNK + 2)]
    sizes = []

    def chat(model, system, user):
        n = len(json.loads(user))
        sizes.append(n)
        return json.dumps([{"i": i, "score": 0.8} for i in range(n)])
    monkeypatch.setattr(judge.aisa, "chat", chat)
    assert judge.score(rows, "m") == {"judged": judge.CHUNK + 2} and sizes == [judge.CHUNK, 2]
    assert all(r["named"] == 0.8 for r in rows)


def test_judge_failure_leaves_rows_unscored(monkeypatch):
    def chat(*a):
        raise judge.aisa.AisaError("HTTP 402: no credit")
    monkeypatch.setattr(judge.aisa, "chat", chat)
    rows = [{"term": "strata", "examples": []}]
    assert "error" in judge.score(rows, "m") and "named" not in rows[0]


# --- merging the signals ---------------------------------------------------------------------

def _no_judge(monkeypatch):
    monkeypatch.setattr(report.judge, "score", lambda rows, model: {"judged": 0})


def test_leads_merge_a_term_seen_by_two_signals(cfg, monkeypatch):
    _no_judge(monkeypatch)
    blocks = {"x": {"terms": [{"term": "strata", "recent": 2, "note": "X: 2 accounts"}]},
              "breakout": {"terms": [{"term": "strata", "recent": 1, "note": "GitHub breakout",
                                      "variants": ["strata", "niko1221 strata"]}]},
              "sitemap": {"terms": [{"term": "other", "recent": 3, "lift": 4.0}]}}
    rows, _ = report.leads(cfg, blocks)
    assert rows[0]["term"] == "strata" and rows[0]["kinds"] == ["x", "breakout"]
    assert rows[0]["variants"] == ["strata", "niko1221 strata"]
    assert rows[1]["signals"] == ["③ sitemaps: 3 places (×4.0)"]


def test_names_lead_and_known_terms_are_not_recorded(cfg, monkeypatch):
    cfg["known_terms"] = ["Remotion"]

    def score(rows, model):
        for r in rows:
            r["named"] = {"dashboard": 0.1, "remotion skill": 0.9, "hyperframes": 0.95}[r["term"]]
        return {"judged": len(rows)}
    monkeypatch.setattr(report.judge, "score", score)
    blocks = {"sitemap": {"terms": [{"term": t, "recent": 3, "lift": 2.0} for t in ("dashboard", "remotion skill", "hyperframes")]}}
    rows, _ = report.leads(cfg, blocks)
    assert [r["term"] for r in report.named(cfg, rows)] == ["hyperframes", "remotion skill"]
    assert [r["term"] for r in report.to_record(cfg, rows)] == ["hyperframes"]


def test_known_terms_match_either_way(cfg):
    cfg["known_terms"] = ["ppt skills"]
    assert config.known(cfg, "ppt") == "ppt skills" and config.known(cfg, "best ppt skills") == "ppt skills"
    assert config.known(cfg, "hyperframes") == ""


# --- the store ---------------------------------------------------------------------------------

def _lead(term, kinds, **more):
    return {"term": term, "kinds": kinds, "signals": [f"note {term}"], **more}


def test_store_upserts_and_merges(tmp_path):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("strata", ["x"], named=0.9, variants=["strata"])], date(2026, 9, 25))
    store.record(conn, [_lead("strata", ["breakout"], variants=["strata", "niko strata"])], date(2026, 9, 26))
    row = conn.execute("SELECT first_seen, last_seen, signals, named, variants FROM leads").fetchone()
    assert row[:2] == ("2026-09-25", "2026-09-26") and json.loads(row[2]) == ["breakout", "x"]
    assert row[3] == 0.9 and json.loads(row[4]) == ["strata", "niko strata"]


def test_store_due_verdict_and_summary(tmp_path):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("strata", ["x", "breakout"]), _lead("flop", ["sitemap"])], date(2026, 9, 25))
    store.record(conn, [_lead("young", ["hf"])], date(2026, 10, 8))
    today = date(2026, 10, 10)
    assert [d[0] for d in store.due(conn, today, 14, 5)] == ["flop", "strata"]
    store.save_verdict(conn, "strata", "hit", 2, {}, today)
    store.save_verdict(conn, "flop", "miss", None, {}, today)
    assert store.due(conn, today, 14, 5) == []
    summ = store.summary(conn, today)
    assert summ["pending"] == 1 and summ["by_signal"]["x"] == {"checked": 1, "hit": 1, "median_lead": 2}
    assert summ["by_signal"]["sitemap"] == {"checked": 1, "hit": 0, "median_lead": None}


def test_store_recent_sources_are_only_x_and_hf(tmp_path):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("qwen3.9", ["x"]), _lead("strata", ["breakout"])], date(2026, 10, 10))
    assert [r["term"] for r in store.recent_sources(conn, date(2026, 10, 10))] == ["qwen3.9"]


# --- ⑤ review: did it rise, judged against its own baseline ----------------------------------

SEEN = date(2026, 9, 25)


def _series(values_by_day):
    start = date(2026, 9, 19)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values_by_day)]


def test_review_a_word_with_an_older_meaning_does_not_count_as_rising():
    # 'strata' alone, 2026-09-19..10-02, as DataForSEO returned it: high baseline throughout
    assert not review.rose(_series([40, 67, 70, 67, 76, 80, 68, 62, 70, 100, 89, 98, 36, 33]), SEEN)["rose"]


def test_review_a_new_name_rising_from_zero_counts_and_dates_the_rise():
    out = review.rose(_series([0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 100, 0, 0]), SEEN)
    assert out["rose"] and out["first_rise"] == "2026-09-28"


def test_review_all_zero_is_no_data():
    assert review.rose(_series([0] * 14), SEEN)["no_data"]


def test_review_verdict_takes_the_earliest_variant_that_rose():
    per = {"strata": {"rose": False, "no_data": False},
           "niko1221 strata": {"rose": True, "first_rise": "2026-09-27"},
           "niko strata": {"rose": True, "first_rise": "2026-09-30"}}
    assert review.verdict(per, SEEN) == ("hit", 2)


def test_review_verdict_miss_and_no_data():
    assert review.verdict({"a": {"rose": False, "no_data": False}}, SEEN) == ("miss", None)
    assert review.verdict({"a": {"rose": False, "no_data": True}}, SEEN) == ("no_data", None)


def test_variant_series_picks_the_smallest_daily_range_and_slices(monkeypatch):
    asked = []
    monkeypatch.setattr(review.webtool, "series", lambda kw, geo, rng: (
        asked.append(rng) or [(SEEN - timedelta(days=40) + timedelta(days=i), 5) for i in range(80)]))
    pts = review._variant_series("x", SEEN, date(2026, 10, 10), 14)
    assert asked == ["90d"]                       # 30+15d span — the 30d range can't reach the baseline
    assert pts[0][0] == SEEN - timedelta(days=30) and pts[-1][0] == SEEN + timedelta(days=14)


def test_variant_series_rejects_a_coarser_than_daily_grain(monkeypatch):
    weekly = [(SEEN - timedelta(days=35) + timedelta(days=7 * i), 5) for i in range(9)]
    monkeypatch.setattr(review.webtool, "series", lambda kw, geo, rng: weekly)
    assert review._variant_series("x", SEEN, TODAY, 14) is None   # ~6 points can't judge a 14-day window


def test_review_stores_the_verdict(cfg, tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("niko strata", ["breakout"])], SEEN)
    points = _series([0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 100, 0, 0])
    monkeypatch.setattr(review.webtool, "series", lambda kw, geo, rng: points)
    out = review.review(cfg, conn, date(2026, 10, 10))
    assert out["checked"] == [{"term": "niko strata", "first_seen": "2026-09-25", "verdict": "hit", "lead_days": 3}]
    assert conn.execute("SELECT verdict, lead_days FROM leads").fetchone() == ("hit", 3)


# --- the two commands ---------------------------------------------------------------------------

def _write_config(tmp_path, **more):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"x_accounts": {"labs": ["a"]}, **more}))
    return path


def test_sources_run_writes_no_report_when_nothing_is_new(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.sources, "x_new", lambda cfg, state: {"posts": 400, "terms": []})
    monkeypatch.setattr(cli.sources, "hf_new", lambda cfg, state: {"models": 100, "terms": []})
    out = tmp_path / "sources.md"
    assert cli.main(["sources", "--config", str(_write_config(tmp_path)), "--state", str(tmp_path / "s"), "--out", str(out)]) == 0
    assert not out.exists()


def test_sources_run_reports_and_keeps_a_new_name(tmp_path, monkeypatch):
    _no_judge(monkeypatch)
    monkeypatch.setattr(cli.sources, "x_new", lambda cfg, state: {"posts": 400, "cost_micros": 72000, "terms": [
        {"term": "qwen3.9", "recent": 2, "note": "X: 2 accounts, 900 likes", "examples": ["@Alibaba_Qwen: Qwen3.9 is out"],
         "posts": [{"account": "Alibaba_Qwen", "text": "Qwen3.9 is out", "url": "https://x.com/p/1", "likes": 900}]}]})
    monkeypatch.setattr(cli.sources, "hf_new", lambda cfg, state: {"models": 100, "terms": []})
    out, state = tmp_path / "sources.md", tmp_path / "s"
    cli.main(["sources", "--config", str(_write_config(tmp_path)), "--state", str(state), "--out", str(out)])
    text = out.read_text()
    assert "**qwen3.9**" in text and "$0.072" in text and "[post](https://x.com/p/1)" in text
    assert store.connect(state).execute("SELECT term FROM leads").fetchall() == [("qwen3.9",)]


def test_daily_run_writes_a_report_when_every_signal_is_down(tmp_path, monkeypatch):
    def down(*a, **k):
        raise OSError("network is down")
    for mod, fn in ((cli.sources, "x_new"), (cli.sources, "hf_new"), (cli.breakout, "breakouts"), (cli.sitemaps, "diff")):
        monkeypatch.setattr(mod, fn, down)
    out = tmp_path / "report.md"
    assert cli.main(["daily", "--config", str(_write_config(tmp_path)), "--state", str(tmp_path / "s"), "--out", str(out),
                     "--skip", "newsites,review"]) == 0
    text = out.read_text()
    assert text.count("network is down") >= 4 and out.with_suffix(".json").exists()


def test_daily_report_in_chinese(tmp_path, monkeypatch):
    _no_judge(monkeypatch)
    monkeypatch.setattr(cli.breakout, "breakouts", lambda cfg, state: {"searched": 120, "first_run": False, "terms": [], "repos": [
        {"repo": "Niko1221/Strata", "stars": 7734, "velocity": 859, "delta": 600, "age_days": 9.0, "description": "Qwen3.8", "new": True}]})
    out = tmp_path / "report.md"
    cli.main(["daily", "--config", str(_write_config(tmp_path, lang="zh")), "--state", str(tmp_path / "s"), "--out", str(out),
              "--skip", "x,hf,sitemaps,newsites,review"])
    text = out.read_text()
    assert "## 新词雷达" in text and "🆕 [Niko1221/Strata]" in text and "| +600 |" in text


# --- topics and mail -----------------------------------------------------------------------------

def test_topic_section_lists_matching_leads_repos_and_queries(cfg):
    cfg["topics"] = {"AI video": r"\bvideo"}
    lead = [{"term": "reelmimic", "examples": ["edenfunf/reelmimic: Show it a video you love"], "signals": ["② breakout"]},
            {"term": "strata", "examples": ["Niko1221/Strata: Qwen3.8 on a gaming PC"], "signals": ["② breakout"]}]
    blocks = {"breakout": {"repos": [{"repo": "a/motion-video-kit", "description": "video kit", "stars": 900, "velocity": 300},
                                     {"repo": "Niko1221/Strata", "description": "Qwen3.8", "stars": 7700, "velocity": 870}]}}
    found = {"queries": [{"query": "claude video skills", "impressions": 6, "page": "/best/x/"},
                         {"query": "best ai scraping tools", "impressions": 3, "page": "/best/web-scraping/"}]}
    text = "\n".join(report.topic_lines(cfg, lead, blocks, found))
    assert "reelmimic" in text and "motion-video-kit" in text and "claude video skills" in text
    assert "strata" not in text.lower() and "scraping" not in text


def test_every_text_has_both_languages():
    assert all(len(pair) == 2 and all(pair) for pair in i18n.T.values())


def test_mail_html_renders_tables():
    out = notify.html("| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<table style=" in out and ">1</td>" in out
    # Gmail drops a <style> block outside <head>: every tag carries its own style.
    assert "<style>" not in out and '<td style="border:1px solid' in out


def test_mail_without_recipient_is_skipped(monkeypatch):
    monkeypatch.delenv("RADAR_EMAIL_TO", raising=False)
    assert notify.send("s", "x").startswith("mail skipped")


def test_mail_recipients_tolerate_how_people_type_them():
    assert notify.recipients('"a@b.com"') == ["a@b.com"]
    assert notify.recipients("a@b.com，c@d.cn") == ["a@b.com", "c@d.cn"]
    assert notify.recipients("<a@b.com>; c@d.cn\n") == ["a@b.com", "c@d.cn"]


# --- the generic terms behind the names ---------------------------------------------------------

from radar import usecases  # noqa: E402

TODAY = date(2026, 10, 12)


def _daily(values, end=TODAY - timedelta(days=1)):
    """A daily series ending yesterday, as Trends returns it; its last two days are unsettled."""
    start = end - timedelta(days=len(values) - 1)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


def test_usecases_reads_new_names_first_then_those_due_again():
    seen = {"old": {"first": "2026-10-08", "runs": 1}, "done": {"first": "2026-10-01", "runs": 2},
            "fresh": {"first": "2026-10-11", "runs": 1}}
    assert usecases.due(["a", "old", "b"], seen, TODAY, 5, 3) == ["a", "b", "old"]
    assert usecases.due(["a", "b", "c"], seen, TODAY, 2, 3) == ["a", "b"]


def test_usecases_parse_drops_the_name_itself_sentences_and_repeats():
    answer = json.dumps({"use_cases": [
        {"phrase": "AI Image Detector", "evidence": "check if any image is AI generated", "brand": False},
        {"phrase": "synthid detector online", "evidence": "x", "brand": True},
        {"phrase": "detector", "evidence": "x"},
        {"phrase": "check whether a picture was made by a model today", "evidence": "x"},
        {"phrase": "ai image detector", "evidence": "again"},
        {"phrase": "photoshop alternative", "evidence": "free photoshop clone", "brand": True}]})
    out = usecases.parse("Sure:\n" + answer, "synthid detector")
    assert [(u["phrase"], u["brand"]) for u in out] == [("ai image detector", False), ("photoshop alternative", True)]
    assert usecases.parse("no json here", "x") == []


def test_trends_last_two_days_are_not_a_rise():
    # "ai image detector" on 2026-10-12: flat for 88 days, then 0 and 100 on the two unsettled days
    s = usecases.shape(_daily([5] * 88 + [0, 100]), TODAY)
    assert s["now"] == "flat" and not s["rose"]


def test_a_single_day_spike_on_a_small_term_is_not_a_rise():
    s = usecases.shape(_daily([0] * 80 + [100] + [0] * 9), TODAY)   # "debloat macos"
    assert not s["rose"]


def test_a_real_rise_is_dated_and_still_rising():
    # "synthid detector": nothing, then 9, 53, 40 on the last settled days, then the unsettled two
    s = usecases.shape(_daily([1] * 85 + [9, 53, 40, 59, 100]), TODAY)
    assert s["rose"] and s["new"] and s["now"] == "rising" and s["started"] == "2026-10-07" and s["peak_day"] == "2026-10-08"


def test_a_rise_that_came_back_down_is_fading():
    s = usecases.shape(_daily([1] * 70 + [60, 100, 70, 30, 10] + [5] * 15), TODAY)
    assert s["rose"] and s["now"] == "fading"


def test_searches_a_day_follow_the_terms_own_volume_along_its_curve():
    # 30,000 searches in August at level 10; level 20 in the last settled week: twice as many a day
    points = _daily([10] * 81 + [20] * 9)
    assert usecases.own_per_day(points, 30000, "2026-08", TODAY) == 2000
    assert usecases.own_per_day(points, 0, "2026-08", TODAY) is None


def test_the_benchmark_is_sized_from_terms_whose_volume_is_known():
    def row(volume, term_level, bench_level):
        return {"volume": volume, "month": "2026-08", "_pair": [(d, term_level, bench_level) for d, _ in _daily([0] * 90)]}
    # a term with 3,000 searches a day at a third of the benchmark's level puts the benchmark at 9,000
    assert usecases.bench_daily([row(90000, 10, 30), row(900, 1, 50)]) == 9000   # the 900/month term is too small to count
    assert usecases.bench_daily([row(900, 1, 50)]) is None


def test_decisions():
    up, flat, down = {"now": "rising"}, {"now": "flat"}, {"now": "fading"}
    assert usecases.decide(up, 5000, 300) == "prioritize" and usecases.decide(up, 40, 300) == "watch"
    assert usecases.decide(flat, 5000, 300) == "watch" and usecases.decide(flat, 40, 300) == "skip"
    assert usecases.decide(down, 5000, 300) == "skip" and usecases.decide({"no_data": True}, None, 300) == "watch"


def _fake_usecase_network(monkeypatch):
    rise = _daily([1] * 85 + [9, 53, 40, 59, 100])
    flat = _daily([20] * 90)
    monkeypatch.setattr(usecases, "evidence", lambda names, repos, today, texts=None: (
        {n: {"posts": [{"text": "check if any image is AI generated", "likes": 900, "by": "a", "url": ""}], "readme": "", "repo": ""} for n in names}, 3000))
    monkeypatch.setattr(usecases, "propose", lambda model, name, ev: [
        {"phrase": "ai image detector", "evidence": "check if any image is AI generated", "brand": False},
        {"phrase": "photoshop alternative", "evidence": "free clone", "brand": True},
        {"phrase": "nobody searches this", "evidence": "x", "brand": False}])
    monkeypatch.setattr(usecases, "screen", lambda model, name, ev, cands: cands)
    monkeypatch.setattr(usecases, "volumes", lambda phrases, loc: (
        {"ai image detector": {"volume": 60000, "month": "2026-08"}, "photoshop alternative": {"volume": 6000, "month": "2026-08"},
         "nobody searches this": {"volume": 0, "month": ""}}, 0, []))
    monkeypatch.setattr(usecases, "fetch_series", lambda keywords, bench, today: (
        {k: {"solo": rise if k == "synthid detector" else flat,
             "pair": [(d, v, 40) for d, v in (rise if k == "synthid detector" else flat)]} for k in keywords}, 7200, []))
    monkeypatch.setattr(usecases, "fill_cpc", lambda rows, loc, *a, **k: [])


def test_usecases_run_builds_one_table_and_reads_a_name_twice(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    rows = {r["phrase"]: r for r in uc["rows"]}
    assert uc["proposed"] == 3 and uc["relevant"] == 3 and uc["kept"] == 2 and "nobody searches this" not in rows
    assert rows["synthid detector"]["kind"] == "name" and rows["synthid detector"]["decision"] == "prioritize"
    assert rows["ai image detector"]["per_day"] == 2000 and rows["ai image detector"]["decision"] == "watch"
    assert rows["photoshop alternative"]["brand"] and rows["photoshop alternative"]["decision"] == "skip"
    assert uc["bench_daily"] == 2200 and uc["cost_micros"] == 10200   # the median of 4,000 and 400
    assert usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY + timedelta(days=1))["names"] == []
    assert usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY + timedelta(days=3))["names"] == ["synthid detector"]
    assert usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY + timedelta(days=9))["names"] == []


def test_keyword_table_marks_brands_and_generic_terms_become_leads(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    text = "\n".join(report.keyword_table(cfg, uc))
    assert "| **synthid detector** | name |" in text and "**prioritize**" in text
    assert "| **photoshop alternative** ⚠ | generic | synthid detector |" in text and "someone's brand" in text
    assert "“check if any image is AI generated”" in text
    leads = report.usecase_leads(uc)
    assert [x["term"] for x in leads] == ["ai image detector", "photoshop alternative"] and leads[0]["kinds"] == ["usecase"]


def test_fill_cpc_annotates_rows_and_survives_gaps(monkeypatch):
    """fill_cpc: ideas() hits land cpc on the row; library misses and pre-set
    values are left alone — never a fabricated zero."""
    rows = [{"phrase": "ai headshot"}, {"phrase": "unknown word"}, {"phrase": "pre-set", "cpc": 2.0}]
    calls = []

    def ideas(kw, gl):
        calls.append(kw)
        if kw == "ai headshot":
            return {"cpc": 7.551, "volume": 8100}
        if kw == "unknown word":
            return None
        raise AssertionError("pre-set rows must not be re-fetched")
    monkeypatch.setattr(usecases.webtool, "ideas", ideas)
    assert usecases.fill_cpc(rows, 2840) == []
    assert rows[0]["cpc"] == 7.55 and "cpc" not in rows[1] and rows[2]["cpc"] == 2.0
    assert calls == ["ai headshot", "unknown word"]


def test_fill_cpc_lists_row_errors_and_pool_death_stops(monkeypatch):
    import pytest
    rows = [{"phrase": "a"}, {"phrase": "b"}, {"phrase": "c"}]

    def ideas(kw, gl):
        if kw == "b":
            raise usecases.webtool.WebToolError("HTTP 500")
        if kw == "c":
            raise usecases.webtool.PoolExhausted("gifted pool low")
        return {"cpc": 1.5}
    monkeypatch.setattr(usecases.webtool, "ideas", ideas)
    assert usecases.fill_cpc(rows[:2], 2840) == ["b: HTTP 500"] and rows[0]["cpc"] == 1.5
    with pytest.raises(usecases.webtool.PoolExhausted):
        usecases.fill_cpc(rows[2:], 2840)


def test_fill_cpc_daily_budget_caps_shared_pool(tmp_path, monkeypatch):
    """the state-file counter is a hard ceiling across entry points: a run at
    the cap stops before ideas() is even called, and the count persists so
    the next process (demand-sweep) sees the same spend."""
    import json as j
    rows = [{"phrase": "a"}, {"phrase": "b"}, {"phrase": "c"}]
    calls = []
    monkeypatch.setattr(usecases.webtool, "ideas",
                        lambda kw, gl: (calls.append(kw), {"cpc": 1.0})[1])
    # 2 of 3 rows fit under a cap of 2 — the third is deferred, not fetched
    errs = usecases.fill_cpc(rows, 2840, tmp_path, max_per_day=2,
                             today=usecases.date(2026, 10, 30))
    assert calls == ["a", "b"] and [r.get("cpc") for r in rows] == [1.0, 1.0, None]
    assert errs == ["daily cpc budget 2 reached — later rows unfilled"]
    # the file records today's spend — a "next run" the same day starts at 2/2
    assert j.loads((tmp_path / "cpc-budget.json").read_text())["calls"] == 2
    errs2 = usecases.fill_cpc([{"phrase": "d"}], 2840, tmp_path, max_per_day=2,
                              today=usecases.date(2026, 10, 30))
    assert calls == ["a", "b"] and errs2 == ["daily cpc budget 2 reached — later rows unfilled"]
    # a new day resets — the deferred row gets its call
    usecases.fill_cpc(rows[2:], 2840, tmp_path, max_per_day=2,
                      today=usecases.date(2026, 10, 31))
    assert calls[-1] == "c" and rows[2]["cpc"] == 1.0


def test_usecases_run_reports_cpc_errors(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    monkeypatch.setattr(usecases, "fill_cpc", lambda rows, loc, *a, **k: ["x: boom"])
    assert usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)["cpc_errors"] == ["x: boom"]


def test_review_ignores_the_unsettled_days(cfg, tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("ai image detector", ["usecase"])], date(2026, 9, 28))
    monkeypatch.setattr(review.webtool, "series",
                        lambda kw, geo, rng: _daily([5] * 43 + [0, 100]))
    assert review.review(cfg, conn, TODAY)["checked"][0]["verdict"] == "miss"


# --- remote due: D1-side calls merged into the review budget -----------------------

def _remote_stub(monkeypatch, items):
    class Resp:
        def read(self):
            return json.dumps({"items": items}).encode()
    monkeypatch.setattr(review.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    monkeypatch.setenv("INGEST_URL", "https://x.test")
    monkeypatch.setenv("INGEST_SECRET", "s")


def test_remote_due_merges_filters_and_carries_variants(tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    _remote_stub(monkeypatch, [
        {"term": "alpha", "first_seen": "2026-10-01", "variants": ["alpha ai", "alpha ai"]},
        {"term": "beta", "first_seen": "2026-10-02"},          # in `seen` — skipped
        {"term": "gamma", "first_seen": "not-a-date"},          # bad date — skipped
        "oops", 42,                                             # non-dict — skipped
        {"term": 7, "first_seen": "2026-10-03"}])               # non-str term — skipped
    out = review._remote_due(conn, TODAY, 10, 14, {"beta"})
    assert [(t, str(f)) for t, f, _ in out] == [("alpha", "2026-10-01")]
    assert out[0][2] == ["alpha", "alpha ai"]   # term leads its variants, deduped


def test_remote_due_caps_attempts_and_resets_after_expiry(tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    conn.execute(review._REMOTE_ATTEMPTS)
    conn.execute("INSERT INTO remote_attempts VALUES('capped', 3, ?)", (TODAY.isoformat(),))
    conn.execute("INSERT INTO remote_attempts VALUES('stale', 3, '2026-01-01')")
    _remote_stub(monkeypatch, [
        {"term": "capped", "first_seen": "2026-10-01"},   # capped, fresh — skipped
        {"term": "stale", "first_seen": "2026-10-02"},    # capped but lapsed — eligible
        {"term": "fresh", "first_seen": "2026-10-03"}])
    out = review._remote_due(conn, TODAY, 10, 14, set())
    assert [t for t, _, _ in out] == ["stale", "fresh"]


def test_remote_due_fails_soft_without_creds_or_on_fetch_error(tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    monkeypatch.delenv("INGEST_URL", raising=False)
    monkeypatch.delenv("INGEST_SECRET", raising=False)
    assert review._remote_due(conn, TODAY, 10, 14, set()) == []
    monkeypatch.setenv("INGEST_URL", "https://x.test")
    monkeypatch.setenv("INGEST_SECRET", "s")
    def boom(req, timeout=None):
        raise OSError("endpoint down")
    monkeypatch.setattr(review.urllib.request, "urlopen", boom)
    assert review._remote_due(conn, TODAY, 10, 14, set()) == []


def test_remote_due_survives_a_malformed_payload(tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    monkeypatch.setenv("INGEST_URL", "https://x.test")
    monkeypatch.setenv("INGEST_SECRET", "s")
    class Resp:
        def read(self):
            return b'{"items": ["oops", 42]}'
    monkeypatch.setattr(review.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    assert review._remote_due(conn, TODAY, 10, 14, set()) == []


def test_review_marks_remote_attempts_and_resets_after_expiry(cfg, tmp_path, monkeypatch):
    conn = store.connect(tmp_path)
    _remote_stub(monkeypatch, [{"term": "alpha", "first_seen": "2026-09-20", "variants": []}])
    monkeypatch.setattr(review.webtool, "series", lambda kw, geo, rng: [])
    review.review(cfg, conn, TODAY)
    assert conn.execute(
        "SELECT attempts, last_at FROM remote_attempts WHERE term='alpha'").fetchone() == \
        (1, TODAY.isoformat())
    # a cap older than the reset window restarts the counter rather than
    # accreting past it
    conn.execute("UPDATE remote_attempts SET attempts=3, last_at='2026-01-01' WHERE term='alpha'")
    review.review(cfg, conn, TODAY)
    assert conn.execute(
        "SELECT attempts FROM remote_attempts WHERE term='alpha'").fetchone()[0] == 1


def test_save_verdict_does_not_overwrite_a_judged_lead(tmp_path):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("claude code", ["x"])], date(2026, 9, 20))
    store.save_verdict(conn, "claude code", "hit", 3, {}, date(2026, 10, 4))
    store.save_verdict(conn, "claude code", "miss", None, {}, date(2026, 10, 29))
    row = conn.execute("SELECT verdict, lead_days FROM leads WHERE term='claude code'").fetchone()
    assert row == ("hit", 3)   # the remote re-review can't flip the local call


def test_screen_drops_loosely_related_keywords_and_keeps_all_when_the_call_fails(monkeypatch):
    cands = [{"phrase": p, "brand": False} for p in ("ai image detector", "chatgpt alternative", "ai audio detector")]
    ev = {"posts": [{"text": "check if any image is AI generated", "likes": 900, "by": "a"}], "readme": ""}
    monkeypatch.setattr(usecases.aisa, "chat", lambda *a, **k: '[{"i": 0, "score": 0.95}, {"i": 1, "score": 0.2}, {"i": 2, "score": 0.7}]')
    kept = usecases.relevant(usecases.screen("m", "synthid detector", ev, cands))
    assert [(c["phrase"], c["relevance"]) for c in kept] == [("ai image detector", 0.95), ("ai audio detector", 0.7)]

    def down(*a, **k):
        raise usecases.aisa.AisaError("HTTP 500")
    monkeypatch.setattr(usecases.aisa, "chat", down)
    assert usecases.screen("m", "synthid detector", ev, cands) == cands


def test_momentum_is_the_last_settled_week_against_the_week_before():
    assert usecases.shape(_daily([10] * 76 + [20] * 7 + [40] * 7 + [0, 100]), TODAY)["momentum"] == 2.0
    assert usecases.shape(_daily([0] * 83 + [40] * 7 + [0, 100]), TODAY)["momentum"] is None   # nothing the week before


# --- the optional second source: what people searched together with a name ----------------------

def test_related_queries_split_into_long_tails_and_candidates():
    related = {"rising": ["ai photo detector", "google synthid detector", "openai", "gpt image 2.5"],
               "top": ["ai detector", "synthid detector", "synthid detector online", "ai photo detector"]}
    tails, cands = usecases.split_related("synthid detector", related)
    assert tails == ["google synthid detector", "synthid detector online"]
    assert [c["phrase"] for c in cands] == ["ai photo detector", "gpt image 2.5", "ai detector"]   # "openai" is one word
    assert cands[0]["source"] == "searched" and "rising" in cands[0]["evidence"]


def test_screen_marks_brands_and_other_names(monkeypatch):
    cands = [{"phrase": "free up mac disk space", "brand": False, "source": "model"},
             {"phrase": "gpt image 2.5", "brand": False, "source": "searched"}]
    monkeypatch.setattr(usecases.aisa, "chat", lambda *a, **k:
                        '[{"i": 0, "score": 0.8, "brand": true, "name": false}, {"i": 1, "score": 0.1, "brand": true, "name": true}]')
    viewed = usecases.screen("m", "removemacai", {"posts": [], "readme": "frees 12 GB"}, cands)
    assert viewed[0]["brand"] and viewed[1]["is_name"]
    assert [c["phrase"] for c in usecases.relevant(viewed)] == ["free up mac disk space"]


def test_with_a_key_related_queries_join_the_table_and_other_names_become_leads(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    monkeypatch.setattr(usecases.webtool, "related_queries", lambda name, geo:
        {"rising": ["ai photo detector", "gpt image 2.5", "synthid detector online"], "top": []})
    monkeypatch.setattr(usecases, "screen", lambda model, name, ev, cands: [
        dict(c, relevance=0.9, is_name=c["phrase"] == "gpt image 2.5") for c in cands])
    monkeypatch.setattr(usecases, "volumes", lambda phrases, loc: (
        {p: {"volume": 9000, "month": "2026-08"} for p in phrases if p != "nobody searches this"}, 0, []))
    cfg["usecases"]["country_trends"] = False
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    rows = {r["phrase"]: r for r in uc["rows"]}
    assert rows["ai photo detector"]["source"] == "searched" and "gpt image 2.5" not in rows
    assert uc["cosearch"] == [{"term": "gpt image 2.5", "with": "synthid detector"}]
    assert uc["tails"] == {"synthid detector": ["synthid detector online"]} and uc["cost_micros"] == 10200
    text = "\n".join(report.keyword_table(cfg, uc))
    assert "| **ai photo detector** 🖼 🔎 |" in text and "gpt image 2.5 (with synthid detector)" in text and "synthid detector online" in text
    assert "incomplete today" not in text
    assert {"term": "gpt image 2.5", "kinds": ["cosearch"]}.items() <= report.usecase_leads(uc)[-1].items()


def test_related_failure_is_reported_and_the_table_still_builds(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)

    def down(name, geo):
        raise usecases.webtool.WebToolError("HTTP 502")
    monkeypatch.setattr(usecases.webtool, "related_queries", down)
    cfg["usecases"]["country_trends"] = False
    uc = usecases.run(cfg, tmp_path / "b", ["synthid detector"], {}, TODAY)
    assert uc["related"] and uc["related_errors"] == ["synthid detector: HTTP 502"] and uc["kept"] == 2
    assert "HTTP 502" in "\n".join(report.keyword_table(cfg, uc))


def test_other_names_leave_out_old_products_siblings_and_spelling_variants():
    # what the first real run kept beside filmcraft and pdfcraft (2026-10-13)
    viewed = [{"phrase": p, "is_name": True, "rising": True, "established": old} for p, old in (
        ("premiere pro", True), ("davinci resolve", True), ("photocraft github", False), ("gpt image 2.5", False),
        ("adobe artcraft", False), ("artcraft adobe", False))]
    assert [x["term"] for x in usecases.other_names("filmcraft", viewed, ["photocraft", "filmcraft"])] == ["gpt image 2.5", "adobe artcraft"]
    tails, cands = usecases.split_related("pdfcraft", {"rising": ["pdf craft", "pdf-craft review", "merge pdf files"], "top": []})
    assert tails == ["pdf craft", "pdf-craft review"] and [c["phrase"] for c in cands] == ["merge pdf files"]


def test_every_name_gets_the_country_series_and_the_cap_is_on_generic_terms(cfg, monkeypatch):
    rows = [{"phrase": f"name{i}", "kind": "name"} for i in range(5)] + \
           [{"phrase": f"g{v}", "kind": "generic", "volume": v} for v in (27100, 22200, 8100, 6600, 900)]
    cfg["usecases"]["country_max"] = 3
    asked = []
    monkeypatch.setattr(usecases.webtool, "series", lambda kw, geo, range: asked.append(kw) or [(TODAY, 1)])
    out, cost, errors = usecases.country_series(rows, cfg["usecases"])
    assert sorted(asked) == sorted([f"name{i}" for i in range(5)] + ["g27100", "g22200", "g8100"]) and cost == 0 and not errors


def test_other_names_must_be_rising_and_not_the_same_family():
    viewed = [{"phrase": p, "is_name": True, "rising": r} for p, r in (
        ("gpt image 2.5", True), ("google nano banana", True), ("nano banana pro", True), ("google gemini", False), ("mistral large 4", True))]
    assert [x["term"] for x in usecases.other_names("nano banana 2.1", viewed)] == ["gpt image 2.5", "mistral large 4"]


# --- one country's settled series for the rows that matter -------------------------------------

def _month(values):
    """A 30-day series ending today, as goanyapi.daily_series returns it."""
    return [(TODAY - timedelta(days=len(values) - 1 - i), v) for i, v in enumerate(values)]


def test_an_established_term_80_percent_up_for_five_days_is_a_rise_in_the_country_series():
    # "photoshop alternative" in the US around PhotoCraft: about 50, then 77, 86, 83, 100, 94, and 79 on the partial day
    points = _month([52, 48, 50, 47, 55, 51, 49, 53, 50, 46, 52, 50, 48, 54, 49, 51, 50, 47, 53, 46, 47, 53, 57, 55, 77, 86, 83, 100, 94, 79])
    s = usecases.shape(points, TODAY, usecases.COUNTRY)
    assert s["rose"] and not s["new"] and s["now"] == "rising" and s["started"] == "2026-10-07" and s["momentum"] >= 1.4


def test_a_flat_country_series_stays_flat():
    assert usecases.shape(_month([30, 34, 28, 31] * 7 + [33, 29]), TODAY, usecases.COUNTRY)["now"] == "flat"


def test_names_and_big_generic_terms_take_their_shape_from_the_country(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    cfg["usecases"]["related"] = False
    up = _month([50] * 24 + [77, 86, 83, 100, 94, 79])
    asked = []

    def series(kw, geo, range):
        asked.append(kw)
        if kw == "synthid detector":
            raise usecases.webtool.WebToolError("HTTP 502")
        return up
    monkeypatch.setattr(usecases.webtool, "series", series)
    uc = usecases.run(cfg, tmp_path, ["synthid detector"], {}, TODAY)
    rows = {r["phrase"]: r for r in uc["rows"]}
    # The lookups run three at a time, so the order they are recorded in is the threads' own.
    assert sorted(asked) == ["ai image detector", "photoshop alternative", "synthid detector"]
    assert rows["ai image detector"]["trend_geo"] and rows["ai image detector"]["now"] == "rising"
    assert rows["ai image detector"]["decision"] == "prioritize"      # flat worldwide, up in the US
    assert "trend_geo" not in rows["synthid detector"] and uc["country_errors"] == ["synthid detector: HTTP 502"]
    text = "\n".join(report.keyword_table(cfg, uc))
    assert "still rising (US)" in text and "worldwide data used instead: synthid detector" in text
    assert "This table is incomplete today" in text


def test_small_generic_terms_are_not_worth_a_country_lookup(cfg, monkeypatch):
    rows = [{"phrase": "a", "kind": "generic", "volume": 300}, {"phrase": "b", "kind": "generic", "volume": 5000}, {"phrase": "n", "kind": "name"}]
    asked = []
    monkeypatch.setattr(usecases.webtool, "series", lambda kw, geo, range: asked.append(kw) or [])
    usecases.country_series(rows, cfg["usecases"])
    assert asked == ["n", "b"]


def test_started_rising_is_the_run_that_holds_the_peak_not_an_earlier_blip():
    # a one-day blip on 10-01, then the real rise from 10-07
    points = _month([20] * 19 + [45] + [20] * 5 + [70, 100, 68, 51, 42])
    s = usecases.shape(points, TODAY, usecases.COUNTRY)
    assert s["rose"] and s["started"] == "2026-10-08" and s["peak_day"] == "2026-10-09"


def test_only_things_are_read_for_generic_terms_not_words_that_surged():
    today = [{"term": "mcd", "kinds": ["github"], "named": 0.75}, {"term": "kr.co.klisten.api", "kinds": ["sitemap"], "named": 0.78},
             {"term": "photocraft", "kinds": ["breakout"], "named": 0.95}, {"term": "dots", "kinds": ["breakout"], "named": 0.65},
             {"term": "hailuo", "kinds": ["newsite", "sitemap"]}]
    recent = [{"term": "qwen image 2.1", "kinds": ["hf"], "named": 0.96}, {"term": "use rea", "kinds": ["x"], "named": 0.72},
              {"term": "photocraft", "kinds": ["x"], "named": 0.9}]
    assert usecases.names_to_read(today, recent) == ["photocraft", "hailuo", "qwen image 2.1"]


def test_a_failed_post_search_costs_one_name_its_posts_not_the_step(monkeypatch):
    def call(calls):
        if "photocraft" in calls[0]["arguments"]["query"]:
            raise usecases.aisa.AisaError("HTTP 504: <html><h1>504 Gateway Time-out</h1></html>")
        return {"s": {"tweets": [{"text": "check any image", "likeCount": 900, "author": {"userName": "a"}}]}}, 3000
    monkeypatch.setattr(usecases.aisa, "call", call)
    monkeypatch.setattr(usecases, "_readme", lambda repo: "PhotoCraft opens real PSD files")
    ev, cost = usecases.evidence(["photocraft", "synthid detector"], {"photocraft": "storytold/photocraft"}, TODAY)
    assert cost == 3000 and ev["photocraft"]["readme"] and ev["photocraft"]["posts"] == []
    assert ev["photocraft"]["error"] == "HTTP 504: 504 Gateway Time-out" and len(ev["synthid detector"]["posts"]) == 1


# --- names someone asks about: the `keywords` command and the MCP tool ---------------------------

def test_keywords_on_request_reads_every_name_now_and_leaves_the_record_alone(cfg, tmp_path, monkeypatch):
    _fake_usecase_network(monkeypatch)
    uc, text = cli.keyword_table(cfg, tmp_path, ["SynthID Detector ", "synthid detector", ""], {})
    assert uc["names"] == ["synthid detector"] and "| **ai image detector** 🖼 |" in text
    again, _ = cli.keyword_table(cfg, tmp_path, ["synthid detector"], {})
    assert again["names"] == ["synthid detector"]                 # asked again, read again
    assert not (tmp_path / "usecases.json.gz").exists()           # the daily run's read-twice record is untouched
    many, _ = cli.keyword_table(cfg, tmp_path, [f"name {i}" for i in range(20)], {})
    assert len(many["names"]) == usecases.ASKED_MAX


def test_leads_command_lists_what_was_kept(tmp_path, capsys):
    conn = store.connect(tmp_path)
    store.record(conn, [_lead("strata", ["breakout"], named=0.9)], date.today() - timedelta(days=2))
    store.record(conn, [_lead("old one", ["x"])], date.today() - timedelta(days=30))
    data = cli.kept_leads(tmp_path, 7)
    assert [r["term"] for r in data["leads"]] == ["strata"] and data["hit_rates"]["pending"] == 2
    assert cli.main(["leads", "--state", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "strata  [breakout]" in out and "1 leads in the last 7 days" in out


# --- the MCP server -----------------------------------------------------------------------------

def test_mcp_tools_answer_and_never_print(cfg, tmp_path, monkeypatch, capsys):
    import pytest
    pytest.importorskip("mcp")
    from radar import mcp_server
    (tmp_path / "config.json").write_text(json.dumps({"x_accounts": {"labs": ["a"]}}))
    monkeypatch.setenv("RADAR_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RADAR_STATE", str(tmp_path / "state"))
    store.record(store.connect(tmp_path / "state"), [_lead("photocraft", ["breakout"], named=0.95)], date.today())
    assert [r["term"] for r in mcp_server.recent_leads(7)["leads"]] == ["photocraft"]

    _fake_usecase_network(monkeypatch)
    out = mcp_server._keyword_table(["synthid detector"], {})
    assert "| **ai image detector** 🖼 |" in out["table_markdown"] and out["incomplete"] is False and out["cost_usd"] == 0.1

    monkeypatch.setattr(cli.sources, "x_new", lambda cfg, state: {"posts": 400, "cost_micros": 3000, "terms": [
        {"term": "qwen3.9", "recent": 2, "note": "X: 2 accounts", "examples": ["@a: Qwen3.9 is out"]}]})
    monkeypatch.setattr(cli.sources, "hf_new", lambda cfg, state: {"models": 100, "terms": []})
    monkeypatch.setattr(report.judge, "score", lambda rows, model: {"judged": 0})
    ran = mcp_server._run("sources", [])
    assert ran["wrote_report"] and "**qwen3.9**" in ran["report_markdown"]
    assert capsys.readouterr().out == ""        # stdout is the protocol channel


def test_skills_have_a_name_and_a_description():
    root = Path(__file__).resolve().parents[1] / "skills"
    for name in ("new-term-radar", "ai-listening", "seo-new-keywords-research"):
        text = (root / name / "SKILL.md").read_text()
        head = re.match(r"---\nname: (\S+)\ndescription: (.+)\n---\n", text)
        assert head and head.group(1) == name and 100 < len(head.group(2)) < 1024
        assert "python -m radar" in text and "AISA_API_KEY" in text
        # Claude Code replaces $0, $1 ... in a skill's text with the user's arguments: "$0.05" came out
        # as "<what the user typed>.05" on the first real run. Amounts are written "0.05 USD".
        assert not re.search(r"\$\d", text)


def test_the_aisa_key_comes_from_the_environment_or_the_cli_login(tmp_path, monkeypatch):
    from radar import aisa
    monkeypatch.setattr(aisa.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("AISA_API_KEY", raising=False)
    import pytest
    with pytest.raises(aisa.AisaError, match="aisa login"):
        aisa.key()
    (tmp_path / ".aisa").mkdir()
    (tmp_path / ".aisa" / "key").write_text("from-cli\n")
    assert aisa.key() == "from-cli"
    monkeypatch.setenv("AISA_API_KEY", "from-env")
    assert aisa.key() == "from-env"


def test_every_manifest_carries_the_same_version():
    # A release is one number in five places; the registry rejects a server.json whose
    # version PyPI does not have.
    import radar
    root = Path(__file__).resolve().parents[1]
    server = json.loads((root / "server.json").read_text())
    plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text())
    pyproject = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
    assert {pyproject, server["version"], server["packages"][0]["version"], plugin["version"]} == {radar.__version__}
    assert len(server["description"]) <= 100                                  # the registry's limit
    assert f"mcp-name: {server['name']} -->" in (root / "README.md").read_text()   # how the registry checks who owns the PyPI package


def test_the_mcp_command_says_what_is_missing(monkeypatch, capsys):
    import builtins
    real = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    import sys
    import radar
    monkeypatch.delitem(sys.modules, "radar.mcp_server", raising=False)   # an earlier test may have loaded it;
    monkeypatch.delattr(radar, "mcp_server", raising=False)               # the package keeps its own reference
    monkeypatch.setattr(builtins, "__import__", without_mcp)
    assert cli.main(["mcp"]) == 1
    assert "new-term-radar[mcp]" in capsys.readouterr().err


# --- page one: can a new site get in? ------------------------------------------------------------

from radar import competition  # noqa: E402


def _serp(results, ads=False):
    items = [{"type": "ai_overview"}] + [{"type": "organic", "domain": d, "url": u, "title": t, "rank_group": i + 1}
                                         for i, (d, u, t) in enumerate(results)]
    return {"tasks": [{"result": [{"items": items + ([{"type": "paid"}] if ads else [])}]}]}


def test_page_one_verdicts():
    assert competition.verdict(3, None) == "open" and competition.verdict(0, 15) == "open"
    assert competition.verdict(0, 55) == "crowded" and competition.verdict(None, None) == "crowded"
    assert competition.verdict(1, 30) == "mixed" and competition.verdict(None, 30) == "mixed"
    assert competition.page_kind("youtube.com", "https://youtube.com/watch?v=1", "x") == "video"
    assert competition.page_kind("headshotpro.com", "https://headshotpro.com/", "Free AI Headshot Generator") == "tool"
    assert competition.page_kind("headshots.com", "https://headshots.com/best", "16 Best AI Headshot Generators") == "list"


def test_the_free_audit_lane_demotes_crowded_and_marks_its_source(cfg, tmp_path, monkeypatch):
    """_assess_web: one webtool.kd() call carries the whole audit — page-1 rows
    with inline DR, volume backfill, link budget — at zero reported cost."""
    cfg["competition"]["max_rows"] = 8

    def kd(keyword, gl):
        strong = [{"position": 1, "domain": "canva.com", "url": "https://canva.com/ai", "title": "Canva AI",
                   "page_type": "内页", "is_homepage": False, "dr": 94, "visits": 9_000_000, "kw_hit": False, "strength": 90},
                  {"position": 2, "domain": "adobe.com", "url": "https://adobe.com/x", "title": "Adobe",
                   "page_type": "内页", "is_homepage": False, "dr": 96, "visits": 400_000_000, "kw_hit": False, "strength": 95}]
        results = strong + ([{"position": i + 3, "domain": f"tiny{i}.ai", "url": f"https://tiny{i}.ai/",
                             "title": "Free Tool", "page_type": "首页", "is_homepage": True, "dr": d,
                             "visits": 500, "kw_hit": True, "strength": 10}
                            for i, d in enumerate([8, 12, 21])] if keyword == "open term" else [])
        return {"keyword": keyword, "score": 60 if keyword == "crowded term" else 25, "level": "中",
                "volume": 9000, "trend": {}, "link_budget": {"targetDr": 40}, "reasons": ["x"],
                "type": "generic", "cache_hit": "", "computed_at": 0, "results": results}
    monkeypatch.setattr(competition.webtool, "kd", kd)
    rows = [{"phrase": "crowded term", "decision": "prioritize"}, {"phrase": "open term", "decision": "prioritize"},
            {"phrase": "faded", "decision": "skip"}]
    cost, errors = competition.assess(rows, cfg, tmp_path)
    crowded, opened, faded = rows
    assert cost == 0 and errors == [] and "competition" not in faded
    assert crowded["decision"] == "watch" and crowded["demoted"] and crowded["competition"]["verdict"] == "crowded"
    assert crowded["competition"]["source"] == "webcafe_web" and crowded["competition"]["weak"] == 0
    assert crowded["competition"]["kd"] == 60 and crowded["competition"]["audit_volume"] == 9000
    assert opened["competition"]["weak"] == 3 and opened["competition"]["built"] == 3   # kw_hit homepages
    assert opened["competition"]["weakest"] == {"domain": "tiny0.ai", "dr": 8, "position": 3, "url": "https://tiny0.ai/"}
    assert opened["volume"] == 9000                                                  # free backfill


from radar import asks, pains  # noqa: E402


def _reddit_post(pid, title, score=12, sub="VideoEditors"):
    return {"id": pid, "title": title, "text": "", "score": score, "comments": 3, "subreddit": sub,
            "url": f"https://www.reddit.com/r/{sub}/{pid}", "created": 0}


def test_asks_turns_reddit_requests_into_sized_jobs(cfg, tmp_path, monkeypatch):
    cfg["asks"]["patterns"] = ['"is there an ai tool"', '"how do i automate"']
    posts = {'"is there an ai tool"': [_reddit_post("a", "Is there an AI tool to add b-roll automatically?", 45),
                                       _reddit_post("b", "Any AI that turns slides into notes?", 35, "college"),
                                       _reddit_post("c", "Rate my setup", 1)],
             '"how do i automate"': [_reddit_post("a", "Is there an AI tool to add b-roll automatically?", 45)]}
    monkeypatch.setattr(asks.social, "reddit", lambda q, **k: (posts[q], 1880))
    monkeypatch.setattr(asks.aisa, "chat", lambda model, system, user: json.dumps(
        [{"i": i, "kind": "request", "job": job, "ai": True} for i, job in enumerate(
            ["add b roll automatically", "convert slides to notes"])]))
    monkeypatch.setattr(asks.usecases, "volumes", lambda phrases, loc: (
        {"add b roll automatically": {"volume": 880}, "convert slides to notes": {"volume": 10}}, 0, []))
    first = asks.asks(cfg, tmp_path)
    assert first["read"] == 2 and first["asks"] == 2 and first["unsized"] == 1          # "rate my setup" scored under the floor
    assert [r["phrase"] for r in first["rows"]] == ["add b roll automatically"] and first["cost_micros"] == 2 * 1880
    assert first["terms"][0]["note"] == "Reddit: asked 1× (r/VideoEditors)"
    assert [q["phrase"] for q in first["quiet"]] == ["convert slides to notes"]           # asked, too few searches yet
    text = "\n".join(report._asks_lines(first, cfg))
    assert "| **add b roll automatically** | 1 (45) | r/VideoEditors | 880 |" in text and "do not republish" in text
    assert "| convert slides to notes | 1 (35) | r/college |" in text and "ideas rather than keywords" in text
    again = asks.asks(cfg, tmp_path)
    assert again["rows"][0]["again"] and again["terms"] == []                            # reported once


def test_pains_groups_reviews_posts_and_comments_into_gaps(cfg, monkeypatch):
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")
    ext = "a" * 32

    def serp(q, gl="US"):
        if q.startswith("site:chromewebstore"):
            return [{"url": f"https://chromewebstore.google.com/detail/headshot-ai/{ext}", "title": "x", "rank": 1}], 2000
        return [{"url": "https://www.youtube.com/watch?v=abcdefghijk", "title": "tutorial", "rank": 1}], 2000
    monkeypatch.setattr(pains.goanyapi, "serp", serp)
    monkeypatch.setattr(pains.goanyapi, "chrome_reviews", lambda e, c="": (
        [{"rating": 1, "text": "Needs a photo of your face, no way to upload a group shot", "date": "2026-10-13"},
         {"rating": 5, "text": "Great!", "date": "2026-10-12"}], "", 3000))
    monkeypatch.setattr(pains.social, "reddit", lambda q, **k: ([_reddit_post("r1", "Headshot apps all look plastic")], 1880))
    monkeypatch.setattr(pains.social, "youtube_comments", lambda vid: ([{"text": "Is there a way to do this for a whole team at once?", "likes": 40},
                                                                        {"text": "nice", "likes": 0}], 1000))
    monkeypatch.setattr(pains.aisa, "chat", lambda model, system, user, max_tokens=2500: json.dumps([
        {"gap": "No team or group headshots", "kind": "missing", "items": [0, 2], "phrase": "ai headshots for teams"},
        {"gap": "Results look plastic", "kind": "quality", "items": [1]}]))
    result = pains.run(cfg, "ai headshot")
    assert result["by_source"] == {"chrome": 1, "reddit": 1, "youtube": 1}           # 5 stars, "nice" and repeats are left out
    assert result["gaps"][0]["count"] == 2 and result["gaps"][0]["sources"] == ["chrome", "youtube"]
    text = report.pains(result)
    assert "| No team or group headshots | missing | 2 | ai headshots for teams |" in text and "Trustpilot" in text


def test_tutorial_asks_only_for_leads_marked_one_click(cfg, monkeypatch):
    cfg["tutorials"]["max_leads"] = 3
    monkeypatch.setenv("GOANYAPI_API_KEY", "k")
    monkeypatch.setattr(pains.goanyapi, "serp", lambda q, gl="US": ([{"url": "https://www.youtube.com/watch?v=abcdefghijk", "title": "t", "rank": 1}], 2000))
    monkeypatch.setattr(pains.social, "youtube_comments", lambda vid: ([{"text": "Can someone make this one click? Too many steps", "likes": 12}], 1000))
    monkeypatch.setattr(pains.aisa, "chat", lambda model, system, user, max_tokens=2500: json.dumps(
        [{"gap": "Too many manual steps", "kind": "wants a tool", "items": [0], "phrase": "one click blender export"}]))
    out = pains.tutorial_asks(cfg, [{"term": "opus blender game", "one_click": "tutorial"}, {"term": "plain", "one_click": None}])
    assert [x["term"] for x in out["leads"]] == ["opus blender game"] and out["cost_micros"] == 2000 + 1000
    text = "\n".join(report._tutorial_lines(out))
    assert "Too many manual steps (wants a tool, 1) → one click blender export" in text


# --- datasrc: the paid provider layer is gone — call() is a loud stub ---------

from radar import datasrc, aisa as _aisa, webtool as _webtool  # noqa: E402
import pytest as _pytest  # noqa: E402


def test_ideas_exact_match_returns_metrics_and_unknown_returns_none(monkeypatch):
    """webtool.ideas: the expander echoes a covered seed back as a plain item —
    exact-match on normalized keyword; near-misses and unknown words are not
    silently returned as the seed's metrics."""
    def call(tool, method, path, body=None, params=None):
        if body["seeds"] == ["mystery"]:
            return {"items": [{"keyword": "mystery shopper", "cpc": 1.0}]}
        return {"items": [{"keyword": "claude code alternative", "cpc": 9.9},
                          {"keyword": "Claude  Code ", "volume": 550000, "cpc": 5.04,
                           "lowBid": 1.2, "highBid": 8.0, "competition": 0.7,
                           "competitionLevel": "MEDIUM", "kd": 55,
                           "intent": "commercial", "months": [{"ym": "2026-09", "v": 1}],
                           "trend": {"dir": "rising"}}]}
    monkeypatch.setattr(_webtool, "_call", call)
    x = _webtool.ideas("claude code")
    assert x["cpc"] == 5.04 and x["low_bid"] == 1.2 and x["kd_prescreen"] == 55
    assert x["competition_level"] == "MEDIUM" and x["intent"] == "commercial"
    assert _webtool.ideas("mystery") is None


def test_datasrc_call_is_a_loud_stub_not_a_provider(monkeypatch):
    """Keyword metrics live on webtool.* — any post_dataforseo_* call reaching
    datasrc.call (e.g. aisa.call delegating with no router creds) fails loud,
    never silently routes to a billable endpoint."""
    with _pytest.raises(_aisa.AisaError, match="free webtool lane"):
        datasrc.call([{"call_id": "c", "tool": "post_dataforseo_x", "arguments": {}}])


# --- aisa pure helpers: URL normalization, credential scoping -----------------

def test_aisa_host_rejects_lookalike_domains():
    assert _aisa._aisa_host("https://api.aisa.one/v1")
    assert _aisa._aisa_host("https://aisa.one/x")
    assert not _aisa._aisa_host("https://aisa.one.evil.tld")
    assert not _aisa._aisa_host("https://openrouter.ai/api/v1")
    assert not _aisa._aisa_host("not a url")


def test_chat_url_normalizes_openrouter_suffixes(monkeypatch):
    monkeypatch.setenv("RADAR_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    assert _aisa._chat_url() == "https://openrouter.ai/api/v1/chat/completions"
    monkeypatch.setenv("RADAR_LLM_BASE_URL", "openrouter.ai")
    assert _aisa._chat_url() == "https://openrouter.ai/api/v1/chat/completions"
    monkeypatch.delenv("RADAR_LLM_BASE_URL")
    monkeypatch.delenv("AISA_API_BASE_URL", raising=False)
    assert _aisa._chat_url() == "https://api.aisa.one/v1/chat/completions"


def test_chat_url_appends_to_generic_bases(monkeypatch):
    monkeypatch.setenv("RADAR_LLM_BASE_URL", "https://llm.example.com/v1")
    assert _aisa._chat_url() == "https://llm.example.com/v1/chat/completions"
    monkeypatch.setenv("RADAR_LLM_BASE_URL",
                       "https://llm.example.com/v1/chat/completions")
    assert _aisa._chat_url() == "https://llm.example.com/v1/chat/completions"


def test_llm_key_never_sends_aisa_credential_to_custom_endpoint(monkeypatch):
    # custom endpoint resolved → aisa creds must NOT go out; RADAR_LLM_API_KEY required
    monkeypatch.setattr(_aisa, "CHAT", "https://llm.example.com/v1/chat/completions")
    monkeypatch.setenv("AISA_API_KEY", "aisa-key-must-not-leak")
    monkeypatch.delenv("RADAR_LLM_API_KEY", raising=False)
    with _pytest.raises(_aisa.AisaError, match="RADAR_LLM_API_KEY"):
        _aisa.llm_key()
    monkeypatch.setenv("RADAR_LLM_API_KEY", "ok")
    assert _aisa.llm_key() == "ok"


def test_llm_key_on_aisa_host_prefers_aisa_credential(monkeypatch):
    monkeypatch.setattr(_aisa, "CHAT", "https://api.aisa.one/v1/chat/completions")
    monkeypatch.setenv("AISA_API_KEY", "the-aisa-key")
    monkeypatch.setenv("RADAR_LLM_API_KEY", "scoped-to-custom")
    # on aisa.one the scoped key must not leak as the chat credential either
    assert _aisa.llm_key() == "the-aisa-key"


# _post's own guards (cleartext off-loopback, explicit-key for non-aisa hosts,
# HTTPException wrapping) live inside the function conftest stubs for every
# test — they can't be exercised without defeating the "no network" fixture.
# The reachable halves are covered above: _aisa_host (what the Bearer/x-aisa-
# source decisions key on) and llm_key (which key goes where).
