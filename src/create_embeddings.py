import json
from torch.utils.data import DataLoader
from embed import CovidPapersEmbeddedAdapter, Embedder
from es import ElasticSearchProvider
from data import CovidPapersDataset
from Project import Project

pr = Project()
# prepare the data
ds = CovidPapersDataset.from_path(pr.data_dir / 'metadata.csv')
dl = DataLoader(ds, batch_size=128, num_workers=4, collate_fn=lambda x: x)

with open(pr.base_dir / 'es_index.json', 'r') as f:
    index_file = json.load(f)
    es_provider = ElasticSearchProvider(index_file)

# create the adpater for the data
es_adapter = CovidPapersEmbeddedAdapter()
# drop the dataset
es_provider.drop()
# create a new one
es_provider.create_index()

embedder = Embedder()

for batch in tqdm(dl):
    x = [b['title_abstract'] for b in batch]
    embs = embedder(x)
    es_provider(es_adapter(batch, embs))