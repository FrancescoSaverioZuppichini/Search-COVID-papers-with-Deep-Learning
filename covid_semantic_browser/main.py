from art import text2art
from embed import Embedder
from es import ElasticSearcher
from elasticsearch import Elasticsearch

print(text2art('COVID-19 Browser'))
embedder = Embedder()
es_searcher = ElasticSearcher()

while True:
    query = input('Type your question:')
    query_emb = embedder([query])[0].tolist()
    res = es_searcher(query_emb)
    print(res)
