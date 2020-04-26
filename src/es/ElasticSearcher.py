import textwrap
from elasticsearch import Elasticsearch
from dataclasses import dataclass
from prettytable import PrettyTable

@dataclass
class ElasticSearcherResponse:
    """This class represents a rensponse from the ElasticSearcher.
    It parse the raw response and create a nice looking talbe.
    """
    res: dict
    
    def __post_init__(self):
        self.table = PrettyTable()
        self.table.field_names = ["Score", "Paper"]

        for hit in self.res['hits']['hits']:
            paper = hit['_source']
            title_wrap = '\033[1m' + textwrap.fill(paper['title'], width=75) + '\033[0m'
            abstract_wrap = textwrap.fill(paper['abstract'], width=75)
            url = '\033[4m' + paper['url'] + '\033[0m'
            text = f"{title_wrap}\n{abstract_wrap}\n{url}\n\n"
            self.table.add_row([hit['_score'], text])

    def __str__(self):
        return str(self.table)


@dataclass
class ElasticSearcher:
    """
    This class implements the logic behind searching for a vector in elastic search.
    """
    client: Elasticsearch = Elasticsearch()
    index_name: str = 'covid'

    def __call__(self, vector: list):
        script_query = {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['embed'])",
                    "params": {
                        "query_vector": vector
                    }
                }
            }
        }

        res = self.client.search(
            index= self.index_name,
            body={
                "size": 25,
                "query": script_query,
                "_source": {
                    "includes": ["title", "abstract", 'url']
                }
            })

        return ElasticSearcherResponse(res)