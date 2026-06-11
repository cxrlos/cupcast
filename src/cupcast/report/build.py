from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cupcast.features.goalkeepers import keeper_zscores  # noqa: E402
from cupcast.features.matches import build_match_table  # noqa: E402
from cupcast.report.match_predictions import (  # noqa: E402
    group_stage_predictions,
    knockout_projections,
)
from cupcast.report.sensitivity import (  # noqa: E402
    conditional_paths,
    run_player_scenarios,
)
from cupcast.run import AS_OF, build_components, load_composites, shrunk_fit  # noqa: E402
from cupcast.sim.monte_carlo import simulate_tournament  # noqa: E402
from cupcast.sim.worldcup2026 import ALL_TEAMS  # noqa: E402
from cupcast.validate.backtest import evaluate_folds, prob_array, rolling_folds  # noqa: E402
from cupcast.validate.club import club_backtest, club_data_available, club_report  # noqa: E402
from cupcast.validate.metrics import calibration_table, summarize  # noqa: E402
from cupcast.validate.tournaments import replay_report  # noqa: E402

OUTPUTS = Path("outputs")
FIGURES = Path("docs/tex/figures")
GENERATED = Path("docs/tex/generated")
MARKET_CSV = Path("data/processed/market_outrights.csv")


def write_fragment(name: str, content: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / name).write_text(content + "\n")


def latex_table(frame: pd.DataFrame, columns: dict[str, str], digits: int = 3) -> str:
    renamed = frame[list(columns)].rename(columns=columns)
    return renamed.to_latex(index=False, float_format=f"%.{digits}f")


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

    write_fragment(
        "forecast_headline.tex",
        latex_table(
            table.head(12),
            {
                "team": "Team",
                "p_qualify": "R32",
                "p_qf": "QF",
                "p_sf": "SF",
                "p_final": "Final",
                "p_champion": "Champion",
            },
        ),
    )
    champion = details.match_winner[104]
    runner = details.match_loser[104]
    third = details.match_winner[103]
    triples = pd.DataFrame(
        {
            "Champion": [ALL_TEAMS[i] for i in champion],
            "Runner-up": [ALL_TEAMS[i] for i in runner],
            "Third": [ALL_TEAMS[i] for i in third],
        }
    )
    podium = (
        triples.value_counts(normalize=True).head(8).rename("Probability").reset_index()
    )
    write_fragment(
        "forecast_podium.tex",
        podium.to_latex(index=False, float_format="%.4f"),
    )


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
        write_fragment(
            "market_comparison.tex",
            "\\emph{Market snapshot pending: run \\texttt{cupcast.fetch.odds} "
            "and rebuild reports.}",
        )
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
    write_fragment(
        "market_comparison.tex",
        latex_table(
            merged.head(15),
            {
                "team": "Team",
                "p_champion": "Model",
                "market_p_champion": "Market",
                "edge": "Edge",
            },
        ),
    )


def write_sensitivity(fit, n_eff, elo_ratings, details, n_sims: int, seed: int) -> None:
    lines = ["# Sensitivity analysis", ""]
    lines += [
        "## Path dependence: P(champion | group-stage outcome)",
        "",
        conditional_paths(details).round(4).to_markdown(index=False),
        "",
        "Conditioning is over the same 50k simulations; finishing third forces a",
        "tougher bracket path, which is the gap between the columns.",
        "",
    ]
    scenarios, skipped = run_player_scenarios(
        fit, n_eff, elo_ratings, details, n_sims=n_sims, seed=seed
    )
    lines.append("## Key-player scenarios")
    lines.append("")
    if not scenarios.empty:
        lines += [scenarios.round(4).to_markdown(index=False), ""]
        lines += [
            "Each scenario reweights the squad's expected minutes (next player up",
            "within the position absorbs the freed minutes), rebuilds the squad",
            "composite and the shrinkage prior, and reruns the same seeded 50k",
            "simulation. Deltas are read against the baseline run.",
            "",
        ]
        write_fragment(
            "sensitivity_scenarios.tex",
            latex_table(
                scenarios,
                {
                    "scenario": "Scenario",
                    "p_champion_before": "Before",
                    "p_champion_after": "After",
                    "delta_champion": "$\\Delta$ champion",
                    "delta_sf": "$\\Delta$ SF",
                },
                digits=4,
            ),
        )
    else:
        write_fragment(
            "sensitivity_scenarios.tex",
            "\\emph{Player scenarios pending FBref data: run "
            "\\texttt{cupcast.fetch.fbref} and rebuild reports.}",
        )
    for item in skipped:
        lines.append(f"- skipped: {item}")
    (OUTPUTS / "sensitivity.md").write_text("\n".join(lines) + "\n")
    write_fragment(
        "sensitivity_conditional.tex",
        latex_table(
            conditional_paths(details),
            {
                "team": "Team",
                "p_champion": "P(champ)",
                "p_champ_if_group_win": "given 1st",
                "p_champ_if_runner_up": "given 2nd",
                "p_champ_if_third": "given 3rd q.",
            },
            digits=4,
        ),
    )


def write_validation(table: pd.DataFrame) -> None:
    lines = ["# Validation", ""]
    replays = replay_report(table, 2.5, 1.0)
    lines += ["## Tournament replays (as-of training)", "", replays.to_markdown(index=False), ""]
    write_fragment(
        "validation_replays.tex",
        latex_table(
            replays,
            {
                "tournament": "Tournament",
                "n": "n",
                "log_loss": "Log-loss",
                "brier": "Brier",
                "rps": "RPS",
                "log_loss_uniform": "Log-loss (uniform)",
            },
        ),
    )

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

    write_fragment(
        "validation_pooled.tex",
        latex_table(
            pd.DataFrame([pooled]),
            {"n": "n", "log_loss": "Log-loss", "brier": "Brier", "rps": "RPS"},
        ),
    )

    if club_data_available():
        club_predictions = club_backtest()
        club_metrics = club_report(club_predictions)
        lines += [
            "## Club tier (vs Pinnacle closing odds)",
            "",
            club_metrics.to_markdown(index=False),
            "",
        ]
        write_fragment(
            "club_tier.tex",
            latex_table(
                club_metrics,
                {
                    "forecaster": "Forecaster",
                    "n": "n",
                    "log_loss": "Log-loss",
                    "brier": "Brier",
                    "rps": "RPS",
                },
            ),
        )
    else:
        lines += [
            "## Club tier",
            "",
            "_Club data not present; run `fetch.football_data` first._",
            "",
        ]
        write_fragment(
            "club_tier.tex",
            "\\emph{Club-tier results pending: run \\texttt{cupcast.fetch.football\\_data} "
            "and rebuild reports.}",
        )
    (OUTPUTS / "validation.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    warnings.filterwarnings("ignore")
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)  # matplotlib jitter only; simulation seeds are explicit

    n_sims, seed = 50_000, 2026
    base_fit, n_eff, elo_ratings = build_components()
    fit = shrunk_fit(base_fit, n_eff, elo_ratings, load_composites())
    gk_z = keeper_zscores()
    print(f"goalkeeper shootout adjustment: {len(gk_z)}/48 teams covered")
    details = simulate_tournament(fit, n_sims=n_sims, seed=seed, gk_z=gk_z)
    details.table.round(5).to_csv(OUTPUTS / "simulation_results.csv", index=False)

    write_match_predictions(fit, details)
    write_top3(details)
    write_executive_summary(details)
    write_market_comparison(details)
    write_sensitivity(base_fit, n_eff, elo_ratings, details, n_sims, seed)
    write_validation(build_match_table(since="2012-01-01"))
    print("wrote outputs/: simulation_results.csv, match_predictions(.md/.csv),")
    print("  top3_forecast.md, executive_summary.md, sensitivity.md, validation.md,")
    print("  market_comparison(.md) [if odds snapshot present]")
    print(f"figures: {FIGURES / 'calibration.pdf'}")


if __name__ == "__main__":
    main()
