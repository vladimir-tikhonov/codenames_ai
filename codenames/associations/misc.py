from typing import List, Tuple

from frozendict import frozendict


def prepare_rival_words_with_coefficients(
        assassins: List[str],
        opponent_agents: List[str],
        bystanders: List[str],
        config: frozendict) -> List[Tuple[str, float]]:
    return \
        list(zip(opponent_agents, [config['opponentsCoefficient']] * len(opponent_agents))) + \
        list(zip(assassins, [config['assassinsCoefficient']] * len(assassins))) + \
        list(zip(bystanders, [config['bystanderCoefficient']] * len(bystanders)))
