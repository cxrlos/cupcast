# Sensitivity analysis

## Path dependence: P(champion | group-stage outcome)

| team          |   p_champion |   p_champ_if_group_win |   p_champ_if_runner_up |   p_champ_if_third |
|:--------------|-------------:|-----------------------:|-----------------------:|-------------------:|
| Spain         |       0.15   |                 0.1571 |                 0.1329 |             0.1298 |
| Argentina     |       0.1629 |                 0.172  |                 0.1452 |             0.1684 |
| England       |       0.084  |                 0.0923 |                 0.071  |             0.0827 |
| Japan         |       0.0183 |                 0.0252 |                 0.0186 |             0.0223 |
| France        |       0.0709 |                 0.0826 |                 0.0621 |             0.073  |
| Brazil        |       0.0926 |                 0.0925 |                 0.0993 |             0.0906 |
| United States |       0.0037 |                 0.0061 |                 0.0046 |             0.0047 |
| Mexico        |       0.0082 |                 0.0094 |                 0.0085 |             0.0082 |

Conditioning is over the same 50k simulations; finishing third forces a
tougher bracket path, which is the gap between the columns.

## Key-player scenarios

| scenario                     | team      |   p_champion_before |   p_champion_after |   delta_champion |   delta_sf |   delta_final |
|:-----------------------------|:----------|--------------------:|-------------------:|-----------------:|-----------:|--------------:|
| Rodri out                    | Spain     |              0.15   |             0.1495 |          -0.0005 |     0.0031 |        0.0003 |
| Lamine Yamal out             | Spain     |              0.15   |             0.1473 |          -0.0027 |    -0.0023 |       -0.0025 |
| Bellingham at 70 percent     | England   |              0.084  |             0.0829 |          -0.001  |    -0.0044 |       -0.0016 |
| Messi limited to ~60 minutes | Argentina |              0.1629 |             0.1627 |          -0.0002 |     0.0002 |        0.0018 |
| Mbappé out                   | France    |              0.0709 |             0.0707 |          -0.0002 |     0.0008 |        0.0007 |
| Vinícius Júnior out          | Brazil    |              0.0926 |             0.0935 |           0.0009 |     0.0014 |        0.0019 |

Each scenario reweights the squad's expected minutes (next player up
within the position absorbs the freed minutes), rebuilds the squad
composite and the shrinkage prior, and reruns the same seeded 50k
simulation. Deltas are read against the baseline run.

