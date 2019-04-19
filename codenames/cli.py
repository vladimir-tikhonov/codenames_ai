import argparse
from codenames.config import read_app_config
from codenames.models import get_w2v_models
from codenames.associations import build_associations, get_score, prepare_rival_words_with_coefficients


def cli() -> None:
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
    models_config = app_config['models']
    associations_config = app_config['associations']

    w2v_models = get_w2v_models(models_config)

    rival_words_with_coefficients = prepare_rival_words_with_coefficients(
        assassins, opponent_agents, bystanders, associations_config
    )

    associations = build_associations(
        my_agents,
        rival_words_with_coefficients,
        w2v_models[lang],
        associations_config
    )

    if not associations:
        print('No associations found :(')
        return

    for association in sorted(associations, key=get_score, reverse=True):
        print(str(association))


cli()
