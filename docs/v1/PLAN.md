# Plan

My working plan for the 2026 World Cup forecast. Decisions here are locked; details live in the methodology paper as they're implemented.

## Goal

A reproducible pipeline — fetch → features → fit → validate → simulate → report — that produces W/D/L probabilities, expected goals, and a modal scoreline for every match of the 2026 World Cup, plus championship and podium probabilities from 50,000 Monte Carlo tournament simulations, backtested and compared against bookmaker odds.

## Model decisions

- **Goal model**: Dixon-Coles MLE — Poisson attack/defense with the low-score dependence correction and exponential time decay, friendlies down-weighted. Host advantage applies only to Mexico, USA, and Canada in their home venues; all other matches are neutral.
- **Rating input**: my own implementation of the World Football Elo formula over the full match history, giving as-of-date ratings for backtesting. Spot-checked against eloratings.net.
- **Squad quality**: FBref per-90 performance metrics for 2025-26 club seasons (no market values), aggregated to squad level weighted by expected tournament minutes. Tiered fallback to Elo-implied strength for squads in uncovered leagues.
- **Blend**: shrinkage prior — fitted ratings shrink toward the squad-composite prior with weight inversely proportional to decay-weighted match count, so data-poor teams lean on squad quality.
- **Knockouts**: extra time as rescaled scoring rates, penalty shootouts near 50/50 with a small goalkeeper-quality adjustment.
- **Uncertainty**: parameter draws from the MLE's asymptotic distribution across simulation batches; scenario-based sensitivity for key player availability.

## Data decisions

- **Match history**: API-Football, pulled by competition-season (qualifiers, Nations Leagues, continental championships, friendlies; 2015→2026) through a cache-first client with a persistent request ledger to stay inside the free tier.
- **Club validation data**: football-data.co.uk CSV archive (results + closing odds).
- **Odds**: The Odds API, snapshotted with timestamps.
- **Squads**: confirmed 26-man lists (post June 2, 2026 registration deadline), compiled with citations.

## Validation

Two tiers. Tier 1: match-level log-loss, Brier score, and calibration on club data against bookmaker closing odds. Tier 2: full as-of replays of Euro 2024 and Copa América 2024. Isotonic recalibration if the reliability curve warrants it.

## Phases

1. Scaffold — repo, uv, docs skeleton ✓
2. Data layer — cached clients, rationed pulls, squad research
3. Elo engine
4. Dixon-Coles + shrinkage
5. Ugly end-to-end tournament sim, then refine
6. Two-tier validation, recalibration if needed
7. Final 50k simulation + sensitivity
8. Papers, outputs, publish

Phases 2–4 interleave: the Elo engine and goal model are developed and tested on partial data while rationed API pulls complete.
