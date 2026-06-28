# Cupcast II — Knockout Bracket Forecast (conditioned on the actual draw)

Championship probabilities from **50,000 Monte Carlo simulations seeded with the
real Round-of-32 bracket** (group stage complete, 72/72 played). Unlike the
pre-tournament forecast — which also simulated the groups — this conditions on
the matchups that actually materialised, so each team's odds reflect its true
bracket path. The held-out model (dynamic state-space Dixon–Coles with clubform
priors, default configuration) and the penalty-specific shootout edge are both
included. Seed `2026`.

**Bracket integrity:** the 16 R32 slots resolve to 32 distinct teams and every
slot matches an actual fixture exactly (winners/runners from the official final
standings; third-place pairings taken from the real draw rather than the
`allocate_thirds` heuristic, which mis-assigns 3 of the 16 thirds for this
combination).

## Championship probabilities

| Team | Champion | Final | Semi | QF |
|---|---:|---:|---:|---:|
| Brazil | 19.4% | 28.4% | 43.1% | 65.5% |
| Argentina | 15.2% | 25.7% | 49.8% | 74.6% |
| England | 13.0% | 21.9% | 36.8% | 68.0% |
| Spain | 12.1% | 22.3% | 33.0% | 45.9% |
| Portugal | 6.8% | 14.3% | 23.3% | 35.1% |
| France | 6.2% | 14.5% | 29.0% | 47.9% |
| Netherlands | 6.1% | 14.3% | 28.5% | 49.4% |
| Colombia | 5.4% | 10.9% | 26.3% | 53.6% |
| Belgium | 4.1% | 10.0% | 19.0% | 44.5% |
| Germany | 2.6% | 7.8% | 19.0% | 36.1% |
| Switzerland | 1.6% | 4.0% | 12.2% | 31.0% |
| Morocco | 1.3% | 3.9% | 10.4% | 23.9% |
| USA | 1.1% | 3.7% | 9.3% | 30.8% |
| Norway | 0.8% | 2.3% | 6.2% | 16.8% |
| Austria | 0.7% | 2.2% | 5.1% | 10.4% |
| Senegal | 0.6% | 2.1% | 5.5% | 18.9% |

Full table for all 32 teams in `knockout_bracket_forecast.csv`.

## How the actual draw moved the forecast

- **Argentina: 10.3% → 15.2%.** The softest quadrant of the bracket (Cape Verde
  in the R32) gives it the tournament's highest semifinal probability (49.8%).
- **England: 11.9% → 13.0%** on a favourable path.
- **Brazil** remains the nominal favourite (19.4%) but Argentina is now within a
  few points.
- **Uruguay** was eliminated in the group stage and leaves the title picture
  entirely (pre-tournament 1.5%).
