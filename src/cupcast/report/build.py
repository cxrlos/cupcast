from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cupcast.features.matches import build_match_table  # noqa: E402
from cupcast.report.match_predictions import (  # noqa: E402
    group_stage_predictions,
    knockout_projections,
)
from cupcast.run import AS_OF, build_fit  # noqa: E402
from cupcast.sim.monte_carlo import simulate_tournament  # noqa: E402
from cupcast.sim.worldcup2026 import ALL_TEAMS  # noqa: E402
from cupcast.validate.backtest import evaluate_folds, prob_array, rolling_folds  # noqa: E402
from cupcast.validate.club import club_backtest, club_data_available, club_report  # noqa: E402
from cupcast.validate.metrics import calibration_table, summarize  # noqa: E402
from cupcast.validate.tournaments import replay_report  # noqa: E402

OUTPUTS = Path("outputs")
FIGURES = Path("docs/tex/figures")
MARKET_CSV = Path("data/processed/market_outrights.csv")


def fmt(value: float) -> str:
    return f"{value:.3f}"


def podium_section(details) -> str:
    champion = details.match_winner[104]
    runner = details.match_loser[104]
    third = details.match_winner[103]
    triples = pd.DataFrame(
        {
            "champion": [ALL_TEAMS[i] for i in champion],
            "runner_up": [ALL_TEAMS[i] for i in runner],
            "third": [ALL_TEAMS[i] for i in third],
        }
    )
    top = (
        triples.value_counts(normalize=True)
        .head(10)
        .rename("probability")
        .reset_index()
    )
    lines = ["## Most likely podiums (joint)", ""]
    lines.append("| Champion | Runner-up | Third | Probability |")
    lines.append("|---|---|---|---|")
    for row in top.itertuples():
        lines.append(
            f"| {row.champion} | {row.runner_up} | {row.third} | {row.probability:.3%} |"
        )
    return "\n".join(lines)


def write_match_predictions(fit, details) -> None:
    groups = group_stage_predictions(fit)
    knockouts = knockout_projections(fit, details)
    groups.to_csv(OUTPUTS / "match_predictions_groups.csv", index=False)
    knockouts.to_csv(OUTPUTS / "match_predictions_knockouts.csv", index=False)

    lines = ["# Match predictions", ""]
    lines += [f"Model as of {AS_OF.date()}; probabilities home/draw/away.", ""]
    for stage, stage_frame in groups.groupby("stage", sort=False):
        lines += [f"## {stage}", ""]
        lines.append("| Match | P(H/D/A) | xG | Modal | Conf | Note |")
        lines.append("|---|---|---|---|---|---|")
        for row in stage_frame.itertuples():
            lines.append(
                f"| {row.home} v {row.away} "
                f"| {row.p_home:.2f}/{row.p_draw:.2f}/{row.p_away:.2f} "
                f"| {row.xg_home:.2f}-{row.xg_away:.2f} | {row.modal_score} "
                f"| {row.confidence} | {row.note} |"
            )
        lines.append("")
    lines += ["## Knockout rounds (modal pairings from 50k simulations)", ""]
    lines.append(
        "| Stage | Most likely pairing | P(pairing) | P(first advances) | xG | Modal | Conf |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in knockouts.itertuples():
        lines.append(
            f"| {row.stage} | {row.modal_pairing} | {row.p_this_pairing:.2f} "
            f"| {row.p_first_advances:.2f} | {row.xg_home:.2f}-{row.xg_away:.2f} "
            f"| {row.modal_score} | {row.confidence} |"
        )
    (OUTPUTS / "match_predictions.md").write_text("\n".join(lines) + "\n")


def write_top3(details) -> None:
    table = details.table
    lines = ["# Top-3 forecast", ""]
    for slot, column in (
        ("Champion", "p_champion"),
        ("Runner-up", "p_runner_up"),
        ("Third place", "p_podium_third"),
    ):
        best = table.nlargest(5, column)
        lines += [f"## {slot}", ""]
        for row in best.itertuples():
            lines.append(f"- {row.team}: {getattr(row, column):.1%}")
        lines.append("")
    lines.append(podium_section(details))
    (OUTPUTS / "top3_forecast.md").write_text("\n".join(lines) + "\n")


def write_executive_summary(details) -> None:
    table = details.table.set_index("team")
    lines = ["# Executive summary — Telegraph predictor picks", ""]
    lines.append("## Group stage")
    lines.append("")
    lines.append("| Group | Winner pick | Runner-up pick | Third (qualify odds) |")
    lines.append("|---|---|---|---|")
    for letter, group_frame in details.table.groupby("group"):
        ordered = group_frame.sort_values("p_group_win", ascending=False)
        winner = ordered.iloc[0]
        rest = group_frame[group_frame["team"] != winner["team"]]
        runner = rest.sort_values("p_group_runner_up", ascending=False).iloc[0]
        third_pool = rest[rest["team"] != runner["team"]]
        third = third_pool.sort_values("p_qualify", ascending=False).iloc[0]
        lines.append(
            f"| {letter} | {winner['team']} ({winner['p_group_win']:.0%}) "
            f"| {runner['team']} ({runner['p_group_runner_up']:.0%}) "
            f"| {third['team']} ({third['p_qualify']:.0%}) |"
        )
    lines += ["", "## Knockout picks (advance the higher-probability side)", ""]
    for stage, column in (
        ("Quarter-finalists", "p_qf"),
        ("Semi-finalists", "p_sf"),
        ("Finalists", "p_final"),
    ):
        count = {"Quarter-finalists": 8, "Semi-finalists": 4, "Finalists": 2}[stage]
        picks = table.nlargest(count, column)
        lines.append(f"- **{stage}**: " + ", ".join(picks.index))
    champion = table["p_champion"].idxmax()
    runner = table.drop(champion)["p_runner_up"].idxmax()
    third = table.drop([champion, runner])["p_podium_third"].idxmax()
    lines += [
        "",
        "## Podium",
        "",
        f"1. **{champion}** ({table.loc[champion, 'p_champion']:.1%})",
        f"2. **{runner}** ({table.loc[runner, 'p_runner_up']:.1%})",
        f"3. **{third}** ({table.loc[third, 'p_podium_third']:.1%})",
    ]
    (OUTPUTS / "executive_summary.md").write_text("\n".join(lines) + "\n")


def write_market_comparison(details) -> None:
    if not MARKET_CSV.exists():
        print("market_outrights.csv missing — skipping market comparison (run fetch.odds)")
        return
    market = pd.read_csv(MARKET_CSV)
    merged = details.table[["team", "p_champion"]].merge(market, on="team", how="left")
    merged["edge"] = merged["p_champion"] - merged["market_p_champion"]
    merged = merged.sort_values("edge", key=abs, ascending=False)
    merged.to_csv(OUTPUTS / "market_comparison.csv", index=False)
    lines = ["# Model vs market (championship probabilities)", ""]
    lines.append("| Team | Model | Market | Edge |")
    lines.append("|---|---|---|---|")
    for row in merged.head(15).itertuples():
        market_p = "-" if pd.isna(row.market_p_champion) else f"{row.market_p_champion:.1%}"
        lines.append(f"| {row.team} | {row.p_champion:.1%} | {market_p} | {row.edge:+.1%} |")
    (OUTPUTS / "market_comparison.md").write_text("\n".join(lines) + "\n")


def write_validation(table: pd.DataFrame) -> None:
    lines = ["# Validation", ""]
    replays = replay_report(table, 2.5, 1.0)
    lines += ["## Tournament replays (as-of training)", "", replays.to_markdown(index=False), ""]

    predictions = evaluate_folds(table, rolling_folds("2022-01-01", "2026-06-01", 6), 2.5, 1.0)
    probs, outcome = prob_array(predictions)
    pooled = summarize(probs, outcome)
    lines += [
        "## Rolling out-of-sample (internationals 2022-2026)",
        "",
        pd.DataFrame([pooled]).to_markdown(index=False),
        "",
    ]
    calibration = calibration_table(probs, outcome)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.scatter(calibration["mean_predicted"], calibration["observed_rate"], s=30)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability, rolling out-of-sample internationals")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.pdf")
    plt.close(fig)

    if club_data_available():
        club_predictions = club_backtest()
        club_metrics = club_report(club_predictions)
        lines += [
            "## Club tier (vs Pinnacle closing odds)",
            "",
            club_metrics.to_markdown(index=False),
            "",
        ]
    else:
        lines += [
            "## Club tier",
            "",
            "_Club data not present; run `fetch.football_data` first._",
            "",
        ]
    (OUTPUTS / "validation.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    warnings.filterwarnings("ignore")
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)  # matplotlib jitter only; simulation seeds are explicit

    fit = build_fit()
    details = simulate_tournament(fit, n_sims=50_000, seed=2026)
    details.table.round(5).to_csv(OUTPUTS / "simulation_results.csv", index=False)

    write_match_predictions(fit, details)
    write_top3(details)
    write_executive_summary(details)
    write_market_comparison(details)
    write_validation(build_match_table(since="2012-01-01"))
    print("wrote outputs/: simulation_results.csv, match_predictions(.md/.csv),")
    print("  top3_forecast.md, executive_summary.md, validation.md, market_comparison(.md)")
    print(f"figures: {FIGURES / 'calibration.pdf'}")


if __name__ == "__main__":
    main()
