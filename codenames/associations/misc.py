from typing import List, Tuple

import codenames.config.associations as associations_config


def prepare_rival_words_with_coefficients(
        assassins: List[str],
        opponent_agents: List[str],
        bystanders: List[str]) -> List[Tuple[str, float]]:
    return \
        list(zip(opponent_agents, [associations_config.OPPONENTS_COEFFICIENT] * len(opponent_agents))) + \
        list(zip(assassins, [associations_config.ASSASSINS_COEFFICIENT] * len(assassins))) + \
        list(zip(bystanders, [associations_config.BYSTANDER_COEFFICIENT] * len(bystanders)))
