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
    w2v_models['ru'],
    app_config['associations']
)
print(sorted(associations, key=lambda association: get_score(association), reverse=True)[:20])
print(len(associations))
