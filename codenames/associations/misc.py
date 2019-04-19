from typing import List, Tuple
from configparser import SectionProxy


def prepare_rival_words_with_coefficients(
        assassins: List[str],
        opponent_agents: List[str],
        bystanders: List[str],
        config: SectionProxy) -> List[Tuple[str, float]]:
    return \
        list(zip(opponent_agents, [float(config['OpponentsCoefficient'])] * len(opponent_agents))) + \
        list(zip(assassins, [float(config['AssassinsCoefficient'])] * len(assassins))) + \
        list(zip(bystanders, [float(config['BystanderCoefficient'])] * len(bystanders)))
