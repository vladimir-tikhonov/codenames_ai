import argparse
from codenames.config import read_app_config
from codenames.models import get_w2v_models
from codenames.associations import build_associations, get_score

parser = argparse.ArgumentParser(description='Codenames AI')
parser.add_argument('my_agents')
parser.add_argument('--opponent-agents')
parser.add_argument('--assassins')
parser.add_argument('--bystanders')
parser.add_argument('--lang', choices=['en', 'ru'], default='ru')

args = vars(parser.parse_args())
my_agents = args['my_agents'].split(',')
opponent_agents = args['opponent_agents'].split(',') if args['opponent_agents'] else []
assassins = args['assassins'].split(',') if args['assassins'] else []
bystanders = args['bystanders'].split(',') if args['bystanders'] else []
lang = args['lang']

app_config = read_app_config()
w2v_models = get_w2v_models(app_config['models'])

associations_config = app_config['associations']
rival_words_with_coefficients = \
    list(zip(opponent_agents, [float(associations_config['OpponentsCoefficient'])] * len(opponent_agents))) + \
    list(zip(assassins, [float(associations_config['AssassinsCoefficient'])] * len(assassins))) + \
    list(zip(bystanders, [float(associations_config['BystanderCoefficient'])] * len(bystanders)))

associations = build_associations(
    my_agents,
    rival_words_with_coefficients,
    w2v_models[lang],
    associations_config
)

for association in sorted(associations, key=lambda a: get_score(a), reverse=True):
    print(association)
