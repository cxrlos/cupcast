# Validation

## Held-out 2026 World Cup (the real test)

Both models trained strictly to the 11 June 2026 cutoff, then scored on the 72 completed World Cup matches that both models can name. Lower is better; the uniform forecast scores log-loss 1.099.

| forecaster   |   n |   log_loss |   brier |    rps |
|:-------------|----:|-----------:|--------:|-------:|
| v2           |  72 |     0.8279 |  0.4958 | 0.1489 |
| v1           |  72 |     0.8457 |  0.5043 | 0.1513 |
| uniform      |  72 |     1.0986 |  0.6667 | 0.2315 |

## Rolling out-of-sample (internationals, 3-fold rolling origin)

Time-respecting cross-validation on 3973 pre-cutoff international matches: fit before each fold boundary, score the next block. The dynamic model is compared to a static weighted Dixon–Coles baseline (penaltyblog) and the uniform forecast.

| forecaster   |   n |   log_loss |   brier |    rps |
|:-------------|----:|-----------:|--------:|-------:|
| model        | 772 |     0.9096 |  0.5378 | 0.1746 |
| uniform      | 772 |     1.0986 |  0.6667 | 0.2342 |
| penaltyblog  | 772 |     0.9119 |  0.5373 | 0.1748 |

