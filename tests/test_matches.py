import pandas as pd

from cupcast.features.matches import k_factor, restrict_to_fifa_pool
from cupcast.ratings.elo import (
    K_CONTINENTAL,
    K_FRIENDLY,
    K_OTHER_TOURNAMENT,
    K_QUALIFIER,
    K_WORLD_CUP,
)


def test_k_factor_mapping():
    assert k_factor("FIFA World Cup") == K_WORLD_CUP
    assert k_factor("Copa América") == K_CONTINENTAL
    assert k_factor("UEFA Euro") == K_CONTINENTAL
    assert k_factor("FIFA World Cup qualification") == K_QUALIFIER
    assert k_factor("UEFA Nations League") == K_QUALIFIER
    assert k_factor("Friendly") == K_FRIENDLY
    assert k_factor("Baltic Cup") == K_OTHER_TOURNAMENT


def test_restrict_to_fifa_pool_drops_unanchored_teams():
    matches = pd.DataFrame(
        {
            "home": ["Spain", "Maule Sur", "Spain"],
            "away": ["France", "Aymara", "France"],
            "k": [K_QUALIFIER, K_OTHER_TOURNAMENT, K_FRIENDLY],
        }
    )
    kept = restrict_to_fifa_pool(matches)
    assert len(kept) == 2
    assert set(kept["home"]) == {"Spain"}
