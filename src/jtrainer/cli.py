"""`jtrain` command line. With no subcommand it opens the TUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from jtrainer import db
from jtrainer.clock import SystemClock
from jtrainer.config import Config


def _conn(cfg: Config):
    return db.connect(cfg.resolve(cfg.paths.db))


def cmd_play(cfg: Config, args) -> int:
    from jtrainer.llm import default_client
    from jtrainer.tui.app import JTrainApp

    conn = _conn(cfg)
    if conn.execute("SELECT count(*) FROM clues").fetchone()[0] == 0:
        print("The clue database is empty. Run `jtrain download` then `jtrain ingest` first.")
        return 1
    client = default_client(cfg.llm.effort)
    from jtrainer.llm import NoLLM

    JTrainApp(
        conn,
        cfg,
        SystemClock(),
        np.random.default_rng(),
        None if isinstance(client, NoLLM) else client,
    ).run()
    conn.close()
    return 0


def cmd_download(cfg: Config, args) -> int:
    from jtrainer.ingest.jwolle1 import MAIN_FILE, SPECIAL_FILE, download

    files = [MAIN_FILE] + ([SPECIAL_FILE] if args.special else [])
    for p in download(cfg.resolve(cfg.paths.raw_dir), files):
        print(f"{p} ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_ingest(cfg: Config, args) -> int:
    from jtrainer.ingest.jwolle1 import MAIN_FILE, SPECIAL_FILE, ingest_tsv

    conn = _conn(cfg)
    raw = cfg.resolve(cfg.paths.raw_dir)
    path = Path(args.file) if args.file else raw / MAIN_FILE
    if not path.exists():
        print(f"{path} not found; run `jtrain download` first.")
        return 1
    print(ingest_tsv(conn, path, cfg, progress=True).summary())
    if args.special:
        sp = raw / SPECIAL_FILE
        print(ingest_tsv(conn, sp, cfg, special=True).summary())
    return 0


def cmd_taxonomy(cfg: Config, args) -> int:
    from jtrainer.tagging.taxonomy import load_taxonomy, store_taxonomy

    path = (
        Path(args.path)
        if args.path
        else (cfg.resolve(cfg.paths.taxonomy_dir) / f"{cfg.taxonomy.version}.yaml")
    )
    tax = load_taxonomy(path)
    conn = _conn(cfg)
    new = store_taxonomy(conn, tax)
    print(
        f"taxonomy {tax.version}: {len(tax.topics)} topics "
        f"({'stored' if new else 'already stored'})"
    )
    if tax.version != cfg.taxonomy.version:
        print(f"note: config.toml selects taxonomy {cfg.taxonomy.version!r}")
    return 0


def _tagger(cfg: Config, conn, client):
    from jtrainer.llm import CachedLLM
    from jtrainer.tagging.tagger import Tagger
    from jtrainer.tagging.taxonomy import load_taxonomy

    tax = load_taxonomy(cfg.resolve(cfg.paths.taxonomy_dir) / f"{cfg.taxonomy.version}.yaml")
    from jtrainer.tagging.taxonomy import store_taxonomy

    store_taxonomy(conn, tax)
    return Tagger(
        conn, tax, CachedLLM(client, conn), cfg.llm.tag_model, max_tokens=cfg.llm.tag_max_tokens
    )


def _require_api(cfg: Config):
    from jtrainer.llm import AnthropicClient, default_client

    client = default_client(cfg.llm.effort)
    if not isinstance(client, AnthropicClient):
        print("No Anthropic credentials found. Set ANTHROPIC_API_KEY to use tagging.")
        sys.exit(1)
    return client


def cmd_tag(cfg: Config, args) -> int:
    conn = _conn(cfg)
    if args.tag_cmd == "status":
        from jtrainer.tagging.tagger import TagReview
        from jtrainer.tagging.taxonomy import current_tags_sql

        n = conn.execute(
            f"SELECT count(*) FROM ({current_tags_sql(cfg.taxonomy.version)})"
        ).fetchone()[0]
        fails = conn.execute(
            "SELECT reason, count(*) FROM tag_failures WHERE taxonomy_version = ? GROUP BY 1",
            [cfg.taxonomy.version],
        ).fetchall()
        batches = conn.execute(
            "SELECT batch_id, n_requests, submitted_at, collected_at "
            "FROM tag_batches ORDER BY submitted_at"
        ).fetchall()
        print(f"taxonomy {cfg.taxonomy.version}: {n:,} clues tagged; failures {dict(fails)}")
        for b in batches:
            print(f"  batch {b[0]}: {b[1]} requests, submitted {b[2]}, collected {b[3]}")
        print(f"hand review: {TagReview(conn, cfg.taxonomy.version).summary()}")
        return 0
    client = _require_api(cfg)
    tagger = _tagger(cfg, conn, client)
    if args.tag_cmd == "sample":
        ids = tagger.sample_clue_ids(args.n, np.random.default_rng(args.seed))
        print(f"tagging {len(ids)} sampled clues (seed {args.seed}) with {cfg.llm.tag_model}...")
        report = tagger.tag_clues(ids)
        print(
            f"tagged {report.tagged} ({report.from_cache} from cache); "
            f"{report.already_tagged} already tagged; failures {dict(report.failed)}"
        )
        print("Review them in the TUI: `jtrain`, then t.")
    elif args.tag_cmd == "submit":
        ids = tagger.untagged_clue_ids(limit=args.limit or cfg.llm.batch_size)
        batch_id, report = tagger.submit_batch(client, ids)
        print(
            f"{report.from_cache} stored from cache; submitted {report.submitted} "
            f"as batch {batch_id}"
            if batch_id
            else "nothing to submit"
        )
    elif args.tag_cmd == "collect":
        batches = (
            [args.batch]
            if args.batch
            else [
                r[0]
                for r in conn.execute(
                    "SELECT batch_id FROM tag_batches WHERE collected_at IS NULL"
                ).fetchall()
            ]
        )
        for bid in batches:
            status = client.batch_status(bid)
            if status != "ended":
                print(f"batch {bid}: {status} (not ready)")
                continue
            r = tagger.collect_batch(client, bid)
            print(f"batch {bid}: tagged {r.tagged}, retried {r.retried}, failed {dict(r.failed)}")
    return 0


def cmd_stats(cfg: Config, args) -> int:
    from jtrainer.model.stats import compute_stats
    from jtrainer.tui.stats import summary_text

    conn = _conn(cfg)
    st = compute_stats(conn, cfg)
    from rich.console import Console

    console = Console()
    console.print(summary_text(st))
    if st.topics:
        console.print("\n[b]Topics by priority[/b]")
        for t in st.topics[: args.top]:
            acc = "–" if t.accuracy.ipw is None else f"{t.accuracy.ipw:.0%}"
            console.print(
                f"  {t.rank:>2}. {t.label:<32} n={t.accuracy.n:<4} IPW acc {acc:>4}  "
                f"ability {t.mu:+.2f}±{t.sd:.2f}  freq {t.frequency:.1%}  "
                f"priority {t.priority:.3f}"
            )
    return 0


TERMS_NOTICE = """\
J! Archive is a volunteer-run site. Before scraping, read its terms and robots.txt.
This client obeys robots.txt (including Crawl-delay), waits at least
scrape.min_interval_s between requests, caches every page under paths.html_cache and
never re-fetches a cached page. Re-run with --confirm-terms once you have read them."""


def cmd_scrape(cfg: Config, args) -> int:
    from jtrainer.ingest.flags import Flagger
    from jtrainer.ingest.jarchive import parse_game, parse_season
    from jtrainer.ingest.scrape import PoliteClient, game_path, merge_game, season_path

    if not args.confirm_terms:
        print(TERMS_NOTICE)
        return 1
    conn = _conn(cfg)
    client = PoliteClient(cfg.scrape, cfg.resolve(cfg.paths.html_cache), conn=conn)
    _ = client.robots  # reads robots.txt first; raises the interval to its Crawl-delay
    print(f"robots.txt read; waiting {client.interval_s:.0f}s between requests")
    flagger = Flagger(cfg.ingest.tournament_patterns, cfg.ingest.media_patterns)
    page = client.get(season_path(args.season))
    if page is None:
        print(f"season {args.season} not found")
        return 1
    games = parse_season(page)
    if args.limit:
        games = games[: args.limit]
    print(f"season {args.season}: {len(games)} games")
    totals = {"matched": 0, "inserted": 0, "outcomes": 0}
    for i, g in enumerate(games, 1):
        html = client.get(game_path(g.game_id))
        if html is None:
            continue
        counts = merge_game(conn, parse_game(html, g.game_id), flagger)
        for k in totals:
            totals[k] += counts[k]
        print(f"  [{i}/{len(games)}] game {g.game_id} ({g.air_date}): {counts}")
    print(f"done: {totals}; {client.requests_made} requests made")
    return 0


def cmd_fit(cfg: Config, args) -> int:
    from jtrainer.model.irt import fit_from_db

    conn = _conn(cfg)
    fit = fit_from_db(conn, cfg, SystemClock())
    if fit is None:
        print("No contestant outcomes yet. Run `jtrain scrape` first.")
        return 1
    print(
        f"fitted {len(fit.level_mu)} level difficulties; tau={fit.tau:.2f}, "
        f"rho_know={fit.rho_k:.2f}, rho_guess={fit.rho_g:.2f}"
    )
    print("level means:", ", ".join(f"{m:+.2f}" for m in fit.level_mu))
    return 0


def cmd_golden(cfg: Config, args) -> int:
    """Append real overrides to a candidate file for review before joining the golden set."""
    conn = _conn(cfg)
    rows = conn.execute(
        "SELECT a.attempt_id, c.clue_text, c.response, a.user_response, a.correct "
        "FROM attempts a JOIN clues c USING (clue_id) "
        "WHERE (a.overridden OR a.grader = 'user') AND a.user_response IS NOT NULL "
        "ORDER BY a.attempt_id"
    ).fetchall()
    out = Path(args.out)
    existing = yaml.safe_load(out.read_text()) if out.exists() else None
    existing = existing or []
    seen = {c.get("attempt_id") for c in existing}
    added = 0
    for aid, clue, resp, user, correct in rows:
        if aid in seen:
            continue
        existing.append(
            {
                "attempt_id": aid,
                "clue_text": clue,
                "response": resp,
                "user_response": user,
                "expected": bool(correct),
                "note": "from user override; review before adding to golden set",
            }
        )
        added += 1
    out.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True))
    print(f"added {added} candidates to {out}")
    return 0


def cmd_info(cfg: Config, args) -> int:
    conn = _conn(cfg)
    print(
        json.dumps(
            {
                "db": str(cfg.resolve(cfg.paths.db)),
                "schema_version": db.schema_version(conn),
                "clues": conn.execute("SELECT count(*) FROM clues").fetchone()[0],
                "attempts": conn.execute("SELECT count(*) FROM attempts").fetchone()[0],
                "sessions": conn.execute("SELECT count(*) FROM sessions").fetchone()[0],
                "taxonomy": cfg.taxonomy.version,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jtrain", description="Jeopardy! trainer")
    p.add_argument("--config", help="path to config.toml")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("play", help="open the TUI (default)")
    d = sub.add_parser("download", help="download the jwolle1 clue dataset")
    d.add_argument("--special", action="store_true", help="also fetch extra_matches.tsv")
    i = sub.add_parser("ingest", help="load the TSV into the database")
    i.add_argument("--file")
    i.add_argument("--special", action="store_true", help="also ingest special matches")
    t = sub.add_parser("taxonomy", help="load a taxonomy YAML")
    t.add_argument("action", choices=["load"])
    t.add_argument("path", nargs="?")
    tag = sub.add_parser("tag", help="topic tagging with the Anthropic API")
    tsub = tag.add_subparsers(dest="tag_cmd", required=True)
    ts = tsub.add_parser("sample", help="tag a random sample synchronously")
    ts.add_argument("--n", type=int, default=500)
    ts.add_argument("--seed", type=int, default=0)
    tb = tsub.add_parser("submit", help="submit untagged eligible clues as a Message Batch")
    tb.add_argument("--limit", type=int)
    tc = tsub.add_parser("collect", help="collect finished batches")
    tc.add_argument("--batch")
    tsub.add_parser("status")
    s = sub.add_parser("stats", help="print stats")
    s.add_argument("--top", type=int, default=15)
    sc = sub.add_parser("scrape", help="politely scrape a J! Archive season")
    sc.add_argument("--season", type=int, required=True)
    sc.add_argument("--limit", type=int)
    sc.add_argument("--confirm-terms", action="store_true")
    sub.add_parser("fit-irt", help="fit clue difficulties from contestant outcomes")
    g = sub.add_parser("golden-candidates", help="export overrides as golden-set candidates")
    g.add_argument("--out", default="tests/fixtures/grading_candidates.yaml")
    sub.add_parser("info", help="database summary")
    return p


COMMANDS = {
    None: cmd_play,
    "play": cmd_play,
    "download": cmd_download,
    "ingest": cmd_ingest,
    "taxonomy": cmd_taxonomy,
    "tag": cmd_tag,
    "stats": cmd_stats,
    "scrape": cmd_scrape,
    "fit-irt": cmd_fit,
    "golden-candidates": cmd_golden,
    "info": cmd_info,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    return COMMANDS[args.cmd](cfg, args) or 0


if __name__ == "__main__":
    sys.exit(main())
