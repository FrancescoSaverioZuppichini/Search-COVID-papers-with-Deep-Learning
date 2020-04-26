from art import text2art
from embed import Embedder
from es import ElasticSearcher

print(text2art('COVID-19 Browser'))
print('Loading model...')
embedder = Embedder()
es_searcher = ElasticSearcher()
print('Done!')

while True:
    query = input('Type your question:')
    query_emb = embedder([query])[0].tolist()
    res = es_searcher(query_emb)
    print(res)
