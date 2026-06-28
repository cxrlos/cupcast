# Sensitivity analysis

## Path dependence: P(champion | group-stage outcome)

| team        |   p_champion |   p_champ_if_group_win |   p_champ_if_runner_up |   p_champ_if_third |
|:------------|-------------:|-----------------------:|-----------------------:|-------------------:|
| Brazil      |       0.2035 |                 0.203  |                 0.2133 |             0.197  |
| Spain       |       0.1305 |                 0.1362 |                 0.1179 |             0.1278 |
| England     |       0.1189 |                 0.1283 |                 0.1014 |             0.0803 |
| Argentina   |       0.1034 |                 0.1086 |                 0.0939 |             0.1116 |
| Portugal    |       0.0842 |                 0.0929 |                 0.0783 |             0.0624 |
| France      |       0.0572 |                 0.0708 |                 0.0434 |             0.0522 |
| Netherlands |       0.0554 |                 0.0679 |                 0.0397 |             0.0561 |
| Belgium     |       0.0463 |                 0.0536 |                 0.0379 |             0.0384 |

Conditioning is over the same 50k simulations; finishing third forces a
tougher bracket path, which is the gap between the columns.

