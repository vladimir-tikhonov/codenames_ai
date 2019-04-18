from codenames.config import read_app_config
from codenames.models import get_w2v_models
from codenames.associations import build_associations, get_score

app_config = read_app_config()
w2v_models = get_w2v_models(app_config['models'])
associations = build_associations(
    [
        'пол',
        'ударник',
        'поток',
        'орган',
        'секция',
        'титан',
        'греция',
        'лад',
        'норка',
    ],
    [('птица', 1), ('музыкант', 1)],
    w2v_models['ru'],
    app_config['associations']
)

print(len(associations))
for association in sorted(associations, key=lambda a: get_score(a), reverse=True):
    print(association)
