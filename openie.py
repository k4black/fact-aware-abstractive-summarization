import itertools
import json
import hashlib
import os.path
import pickle

from stanfordcorenlp import StanfordCoreNLP

print('Connecting to corenlp...')
# nlp = StanfordCoreNLP('http://localhost', port=8811)


if not os.path.exists('triplets'):
    os.makedirs('triplets')


def get_triplets(text, use_cache=True):
    hash = None
    if use_cache:
        hash = hashlib.md5(text.encode()).hexdigest()
        if os.path.isfile(f'triplets/{hash}.pkl'):
            with open(f'triplets/{hash}.pkl', 'rb') as f:
                return pickle.load(f)

        # if os.path.isfile(f'triplets/{hash}.tsv'):
        #     with open(f'triplets/{hash}.tsv', 'r') as f:
        #         return [tuple(i.split('\t')) for i in f.read().split('\n')]

    # text = text.replace('%', '%25')
    text = text.replace('%', ' percents')
    props = {'annotators': 'openie,coref', 'pipelineLanguage': 'en', 'outputFormat': 'json', "triple.strict": "true",
             "resolve_coref": "true"}
    props = {'annotators': 'openie', 'pipelineLanguage': 'en', 'outputFormat': 'json', "triple.strict": "true",
             "resolve_coref": "false"}
    props = {'annotators': 'openie', 'pipelineLanguage': 'en', 'outputFormat': 'json', "triple.strict": "false", "resolve_coref": "false"}
    res = nlp.annotate(text, properties=props)

    try:
        res_parsed = [map(lambda j: (j['subject'], j['relation'], j['object']), i['openie']) for i in json.loads(res)['sentences']]
    except json.JSONDecodeError as e:
        print('text', text)
        raise e

    res_list = list(itertools.chain(*res_parsed))

    if use_cache:
        with open(f'triplets/{hash}.pkl', 'wb') as f:
            pickle.dump(res_list, f)

        # with open(f'triplets/{hash}.tsv', 'w') as f:
        #     for i in res_list:
        #         f.write(f'{i[0]}\t{i[1]}\t{i[2]}\n')

    return res_list
