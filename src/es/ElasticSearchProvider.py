import json
from pathlib import Path
from elasticsearch import Elasticsearch
from tqdm.autonotebook import tqdm
from elasticsearch.helpers import bulk
from dataclasses import dataclass


@dataclass
class ElasticSearchProvider:
    """
    This class provides the Elastic Search functionalies. 
    It allows to store a list of 'entries' on the database as well as an array of utilities function to create and drop one index.
    """
    index_file: dict
    client: Elasticsearch = Elasticsearch()
    index_name: str = 'covid'

    def drop(self):
        self.client.indices.delete(index=self.index_name, ignore=[404])
        return self

    def create_index(self):
        self.client.indices.create(index=self.index_name, body=self.index_file)
        return self

    def create_and_bulk_documents(self, entries: list):
        entries_elastic = []
        for entry in entries:
            entry_elastic = {
                **entry,
                **{
                    '_op_type': 'index',
                    '_index': self.index_name
                }
            }

            entries_elastic.append(entry_elastic)

        bulk(self.client, entries_elastic)

    def __call__(self, entries: list):
        self.create_and_bulk_documents(entries)

        return self
