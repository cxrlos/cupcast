# Validation

## Tournament replays (as-of training)

| tournament   |   n |   log_loss |    brier |      rps |   log_loss_uniform |   rps_uniform |
|:-------------|----:|-----------:|---------:|---------:|-------------------:|--------------:|
| euro2024     |  51 |   0.990105 | 0.592759 | 0.18709  |            1.09861 |      0.222222 |
| copa2024     |  32 |   0.88697  | 0.522047 | 0.162184 |            1.09861 |      0.230903 |

## Rolling out-of-sample (internationals 2022-2026)

|    n |   log_loss |    brier |      rps |
|-----:|-----------:|---------:|---------:|
| 4167 |   0.862224 | 0.506782 | 0.168272 |

## Club tier (vs Pinnacle closing odds)

| forecaster       |    n |   log_loss |    brier |      rps |
|:-----------------|-----:|-----------:|---------:|---------:|
| Dixon-Coles      | 9076 |   0.991459 | 0.589838 | 0.202655 |
| Pinnacle closing | 9076 |   0.965148 | 0.573281 | 0.194839 |

