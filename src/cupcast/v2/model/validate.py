"""Model-core validation: probabilistic scoring and rolling OOS evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_loss(P: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-probability of the realized outcome.

    Parameters
    ----------
    P : ndarray, shape (n, 3)
        Predicted probability triples (home win, draw, away win). Clipped to
        ``[1e-15, 1]`` before taking the log.
    y : ndarray, shape (n,)
        Integer outcomes (0=home win, 1=draw, 2=away win).
    """
    P = np.clip(np.asarray(P, dtype=float), 1e-15, 1.0)
    y = np.asarray(y, dtype=int)
    return float(-np.sum(np.log(P[np.arange(len(y)), y])) / len(y))


def brier(P: np.ndarray, y: np.ndarray) -> float:
    """Mean Brier score summed over the three outcome classes.

    Parameters
    ----------
    P : ndarray, shape (n, 3)
        Predicted probability triples (home win, draw, away win).
    y : ndarray, shape (n,)
        Integer outcomes (0=home win, 1=draw, 2=away win).
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros_like(P)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - one_hot) ** 2, axis=1)))


def rps(P: np.ndarray, y: np.ndarray) -> float:
    """Mean ranked probability score for the ordered home/draw/away outcome.

    ``RPS_i = sum_k (cumsum(P_i)_k − cumsum(onehot_i)_k)^2 / (K−1)``
    where ``K=3``.

    Parameters
    ----------
    P : ndarray, shape (n, 3)
        Predicted probability triples (home win, draw, away win).
    y : ndarray, shape (n,)
        Integer outcomes (0=home win, 1=draw, 2=away win).
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    K = P.shape[1]
    one_hot = np.zeros_like(P)
    one_hot[np.arange(len(y)), y] = 1.0
    cum_P = np.cumsum(P, axis=1)
    cum_O = np.cumsum(one_hot, axis=1)
    return float(np.mean(np.sum((cum_P - cum_O) ** 2, axis=1) / (K - 1)))


def _outcome(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def rolling_score(
    matches: pd.DataFrame,
    fit_fn,
    fold_dates: list[str],
) -> pd.DataFrame:
    """Time-respecting OOS evaluation against a uniform (1/3, 1/3, 1/3) baseline.

    For each entry in ``fold_dates``, fit on matches strictly before that
    cutoff and predict matches in ``[cutoff, next_cutoff)``; the last fold
    predicts all remaining matches.

    Parameters
    ----------
    matches:
        DataFrame with columns ``date, home, away, home_goals, away_goals,
        host_home``.
    fit_fn:
        ``fit_fn(train: pd.DataFrame) -> predict`` where
        ``predict(home, away, host_home) -> (p_home, p_draw, p_away)``.
    fold_dates:
        ISO date strings defining fold boundaries.

    Returns
    -------
    DataFrame with index name ``forecaster`` (rows ``model``, ``uniform``)
    and columns ``n, log_loss, brier, rps``.
    """
    if not matches.empty:
        sample = matches["date"].iloc[0]
        tz = getattr(sample, "tzinfo", None)
        if tz is not None:
            dates_ts = [pd.Timestamp(d, tz="UTC") for d in fold_dates]
        else:
            dates_ts = [pd.Timestamp(d) for d in fold_dates]
    else:
        dates_ts = [pd.Timestamp(d) for d in fold_dates]

    model_Ps: list[list[float]] = []
    ys: list[int] = []

    for i, cutoff in enumerate(dates_ts):
        next_cutoff = dates_ts[i + 1] if i + 1 < len(dates_ts) else None

        train = matches[matches["date"] < cutoff]
        if next_cutoff is not None:
            test = matches[(matches["date"] >= cutoff) & (matches["date"] < next_cutoff)]
        else:
            test = matches[matches["date"] >= cutoff]

        if train.empty or test.empty:
            continue

        predict = fit_fn(train)

        for _, row in test.iterrows():
            outcome = _outcome(int(row["home_goals"]), int(row["away_goals"]))
            ph, pd_, pa = predict(row["home"], row["away"], bool(row.get("host_home", True)))
            model_Ps.append([float(ph), float(pd_), float(pa)])
            ys.append(outcome)

    _empty = pd.DataFrame(
        columns=["n", "log_loss", "brier", "rps"],
        index=pd.Index(["model", "uniform"], name="forecaster"),
    )
    if not model_Ps:
        return _empty

    P_model = np.array(model_Ps)
    y_arr = np.array(ys)
    P_uniform = np.full_like(P_model, 1.0 / 3.0)

    rows = [
        {
            "forecaster": name,
            "n": len(y_arr),
            "log_loss": log_loss(P, y_arr),
            "brier": brier(P, y_arr),
            "rps": rps(P, y_arr),
        }
        for name, P in [("model", P_model), ("uniform", P_uniform)]
    ]
    return pd.DataFrame(rows).set_index("forecaster")


def main() -> None:
    """Real-data smoke: fit dynamic DC + clubform prior, print OOS scores vs baselines."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from dotenv import load_dotenv

    load_dotenv()

    import jax.numpy as jnp

    from cupcast.v2.clubform.composite import build_clubform
    from cupcast.v2.clubform.league_strength import fetch_clubelo, league_strength_index
    from cupcast.v2.clubform.report import _build_club_to_league
    from cupcast.v2.fetch import catalog as catalog_mod
    from cupcast.v2.fetch.client import ApiFootballClient
    from cupcast.v2.fetch.endpoints import fetch_players
    from cupcast.v2.model.baseline import fit_penaltyblog_dc, outcome_probs_pb
    from cupcast.v2.model.dixon_coles import team_index
    from cupcast.v2.model.fit import fit_svi
    from cupcast.v2.model.matches import assemble_internationals
    from cupcast.v2.model.predict import outcome_probs
    from cupcast.v2.model.prior import clubform_prior_locs, dynamic_dc_with_prior

    CUTOFF = "2026-06-11"
    FOLD_DATES = ["2024-07-01", "2025-01-01", "2025-07-01"]
    SEED = 2026
    SVI_STEPS = 3000

    client = ApiFootballClient()
    cat = client.get_response("leagues")
    squads = pd.read_csv("data/processed/squads.csv")

    club_entries = catalog_mod.club_leagues_from_squads(cat, squads)
    intl_entries = catalog_mod.resolve_competitions(cat, catalog_mod.INTERNATIONAL_COMPETITIONS)

    club_league_seasons: list[tuple[int, int]] = [
        (e["league"]["id"], season)
        for e in club_entries
        for season in (2024, 2025)
    ]
    intl_comp_seasons: list[tuple[int, int]] = [
        (e["league"]["id"], season)
        for e in intl_entries
        for season in catalog_mod.seasons_between(e, 2018, 2026)
    ]

    matches = assemble_internationals(client, intl_comp_seasons, cutoff=CUTOFF)
    print(f"Matches assembled (pre-cutoff): {len(matches)}")

    player_responses_by_league: dict[int, list[dict]] = {}
    for league_id, season in club_league_seasons:
        responses = fetch_players(client, league_id, season)
        player_responses_by_league.setdefault(league_id, []).extend(responses)

    club_to_league = _build_club_to_league(player_responses_by_league)
    clubelo = fetch_clubelo()
    strength = league_strength_index(clubelo, club_to_league)
    clubform = build_clubform(client, club_league_seasons, intl_comp_seasons, strength)

    teams_all, _, _ = team_index(matches)
    team_to_idx = {t: i for i, t in enumerate(teams_all)}
    att_prior, def_prior = clubform_prior_locs(clubform, teams_all)
    n_teams = len(teams_all)
    n_periods = int(matches["period"].max()) + 1

    def _model_fit_fn(train: pd.DataFrame):
        home_i = [team_to_idx[h] for h in train["home"]]
        away_i = [team_to_idx[a] for a in train["away"]]
        model_args = (
            jnp.array(home_i),
            jnp.array(away_i),
            jnp.array(train["period"].to_numpy(dtype=int)),
            jnp.array(train["host_home"].to_numpy(dtype=float)),
            jnp.array(train["home_goals"].to_numpy(dtype=int)),
            jnp.array(train["away_goals"].to_numpy(dtype=int)),
            n_teams,
            n_periods,
            jnp.array(att_prior),
            jnp.array(def_prior),
        )
        post = fit_svi(dynamic_dc_with_prior, model_args, teams_all, seed=SEED, steps=SVI_STEPS)

        def predict(home: str, away: str, host_home: bool) -> tuple[float, float, float]:
            if home not in post.index or away not in post.index:
                return (1.0 / 3, 1.0 / 3, 1.0 / 3)
            return outcome_probs(post, home, away, host_home)

        return predict

    def _pb_fit_fn(train: pd.DataFrame):
        pb = fit_penaltyblog_dc(train)
        pb_teams = set(train["home"]) | set(train["away"])

        def predict(home: str, away: str, host_home: bool) -> tuple[float, float, float]:
            if home not in pb_teams or away not in pb_teams:
                return (1.0 / 3, 1.0 / 3, 1.0 / 3)
            try:
                return outcome_probs_pb(pb, home, away)
            except Exception:
                return (1.0 / 3, 1.0 / 3, 1.0 / 3)

        return predict

    print("\nFitting dynamic-DC model (SVI) per fold — takes a few minutes...")
    model_scores = rolling_score(matches, _model_fit_fn, FOLD_DATES)

    print("Fitting penaltyblog DC model per fold...")
    pb_scores = rolling_score(matches, _pb_fit_fn, FOLD_DATES)
    pb_scores.index = pd.Index(["penaltyblog", "uniform_pb"], name="forecaster")

    combined = pd.concat([model_scores, pb_scores.loc[["penaltyblog"]]])

    print("\nOOS scores (3-fold rolling origin):")
    print(combined.to_string(float_format="{:.4f}".format))

    # Full fit for ratings sanity check.
    print("\nFitting model on all pre-cutoff data for ratings table...")
    home_i_full = [team_to_idx[h] for h in matches["home"]]
    away_i_full = [team_to_idx[a] for a in matches["away"]]
    model_args_full = (
        jnp.array(home_i_full),
        jnp.array(away_i_full),
        jnp.array(matches["period"].to_numpy(dtype=int)),
        jnp.array(matches["host_home"].to_numpy(dtype=float)),
        jnp.array(matches["home_goals"].to_numpy(dtype=int)),
        jnp.array(matches["away_goals"].to_numpy(dtype=int)),
        n_teams,
        n_periods,
        jnp.array(att_prior),
        jnp.array(def_prior),
    )
    full_post = fit_svi(
        dynamic_dc_with_prior, model_args_full, teams_all, seed=SEED, steps=SVI_STEPS
    )
    ratings = (
        pd.DataFrame(
            {
                "team": teams_all,
                "attack": full_post.attack,
                "defense": full_post.defense,
                "rating": full_post.attack - full_post.defense,
            }
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    print("\nTop-20 team ratings (attack − defense):")
    print(ratings.head(20).to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
