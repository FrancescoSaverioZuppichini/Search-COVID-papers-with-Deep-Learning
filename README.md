

```python
%load_ext autoreload
%autoreload 2
```

# Search COVID papers with Deep Learning
*Transformers + Elastic Search = ❤️*

![alt](https://github.com/FrancescoSaverioZuppichini/Search-COVID-papers-with-Deep-Learning/blob/develop/images/cl.gif?raw=true)

Good news everyone, in this article we are not going to fit a linear regression model on the COVID-19 data! But, instead we are going to build a sematic browser using deep learning to search in more than 50k papers about the recent COVID-19 disease.  

The key idea is to encode each paper in a vector representing its semantic content and then search using cosine similary between a query and all the encoded documents. This is the same process used by image browsers (e.g. Google Images) to search for similar images. 

So, so our puzzle is composed by three pieces: data, a mapping from papers to vectors and a way to search.

Most of the work is based on [this project](https://github.com/gsarti/covid-papers-browser) in which I am working with students from the Universita of Trieste (Italy). A live demo is available [here](http://covidbrowser.areasciencepark.it/).


Let's get started!

## Data

Everything starts with the data. We will use this [dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) from Kaggle. A list of of over 57,000 scholarly articles prepared by the White House and a coalition of leading research groups. Actually, the only file we need is `metadata.csv` that contains information about the papers and the full text of the abstract. You need to store the file inside `./dataset`.

Let's take a look


```python
import pandas as pd
from covid_semantic_browser.Project import Project
# Project holds all the paths
pr = Project()

df = pd.read_csv(pr.data_dir / 'metadata.csv')

df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cord_uid</th>
      <th>sha</th>
      <th>source_x</th>
      <th>title</th>
      <th>doi</th>
      <th>pmcid</th>
      <th>pubmed_id</th>
      <th>license</th>
      <th>abstract</th>
      <th>publish_time</th>
      <th>authors</th>
      <th>journal</th>
      <th>Microsoft Academic Paper ID</th>
      <th>WHO #Covidence</th>
      <th>has_pdf_parse</th>
      <th>has_pmc_xml_parse</th>
      <th>full_text_file</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>zjufx4fo</td>
      <td>b2897e1277f56641193a6db73825f707eed3e4c9</td>
      <td>PMC</td>
      <td>Sequence requirements for RNA strand transfer ...</td>
      <td>10.1093/emboj/20.24.7220</td>
      <td>PMC125340</td>
      <td>11742998.0</td>
      <td>unk</td>
      <td>Nidovirus subgenomic mRNAs contain a leader se...</td>
      <td>2001-12-17</td>
      <td>Pasternak, Alexander O.; van den Born, Erwin; ...</td>
      <td>The EMBO Journal</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>custom_license</td>
      <td>http://europepmc.org/articles/pmc125340?pdf=re...</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, we have a lot of information. We are obviously interested in the text columns. Working with pandas is not ideal, so let's create a `Dataset`. This will allow us to later create a `DataLoader` to perform batch-wise encoding. If you are not familiar with the Pytorch data loading ecosystem you can read more about [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)


```python
from torch.utils.data import Dataset, DataLoader

class CovidPapersDataset(Dataset):
    FILTER_TITLES = ['Index', 'Subject Index', 'Subject index', 'Author index', 'Contents', 
    'Articles of Significant Interest Selected from This Issue by the Editors',
    'Information for Authors', 'Graphical contents list', 'Table of Contents',
    'In brief', 'Preface', 'Editorial Board', 'Author Index', 'Volume Contents',
    'Research brief', 'Abstracts', 'Keyword index', 'In This Issue', 'Department of Error',
    'Contents list', 'Highlights of this issue', 'Abbreviations', 'Introduction',
    'Cumulative Index', 'Positions available', 'Index of Authors', 'Editorial',
    'Journal Watch', 'QUIZ CORNER', 'Foreword', 'Table of contents', 'Quiz Corner',
    'INDEX', 'Bibliography of the current world literature', 'Index of Subjects',
    '60 Seconds', 'Contributors', 'Public Health Watch', 'Commentary',
    'Chapter 1 Introduction', 'Facts and ideas from anywhere', 'Erratum',
    'Contents of Volume', 'Patent reports', 'Oral presentations', 'Abkürzungen',
    'Abstracts cont.', 'Related elsevier virology titles contents alert', 'Keyword Index',
    'Volume contents', 'Articles of Significant Interest in This Issue', 'Appendix', 
    'Abkürzungsverzeichnis', 'List of Abbreviations', 'Editorial Board and Contents',
    'Instructions for Authors', 'Corrections', 'II. Sachverzeichnis', '1 Introduction',
    'List of abbreviations', 'Response', 'Feedback', 'Poster Sessions', 'News Briefs',
    'Commentary on the Feature Article', 'Papers to Appear in Forthcoming Issues', 'TOC',
    'Glossary', 'Letter from the editor', 'Croup', 'Acronyms and Abbreviations',
    'Highlights', 'Forthcoming papers', 'Poster presentations', 'Authors',
    'Journal Roundup', 'Index of authors', 'Table des mots-clés', 'Posters',
    'Cumulative Index 2004', 'A Message from the Editor', 'Contents and Editorial Board',
    'SUBJECT INDEX', 'Contents page 1']
    # Abstracts that should be treated as missing abstract
    FILTER_ABSTRACTS = ['Unknown', '[Image: see text]']

    def __init__(self, df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.df = self.df[['title', 'authors', 'abstract', 'url', 'pubmed_id']]
        self.df.loc[:,'title'].fillna('', inplace = True)
        self.df.loc[:,'title'] = df.title.apply( lambda x: '' if x in self.FILTER_TITLES else x)
        self.df.loc[:,'abstract'] = df.abstract.apply( lambda x: '' if x in self.FILTER_ABSTRACTS else x)
        self.df = self.df[self.df['abstract'].notna()]
        self.df = self.df[self.df.abstract != '']
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.fillna(0)
        
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        self.df.loc[idx:, 'title_abstract'] = f"{row['title']} {row['abstract']}"
        return  self.df.loc[idx].to_dict()

    def __len__(self):
        return self.df.shape[0]
    
    @classmethod
    def from_path(cls, path, *args, **kwargs):
        df = pd.read_csv(path)
        return cls(df=df, *args, **kwargs)
```

In order, I have subclassed `torch.utils.data.Dataset` to create a custom dataset. The dataset is expecting a dataframe as input from which we kept only the interesting columns. Then, we removed some of the rows where the `abstract` and `title` columns matched one of the "junk" word in `FILTER_TITLE` and `FILTER_ABSTRACT` respectively. This is done because articles were scrapted in an automatic fashion, and many have irrelevant entries instead of title/abstract information.

The dataset returns a dictionary since `pd.DataFrame` is not a supported type in pytorch. To give our search engine more context, we merge the `title` and the `abstract` together, the result is stored in the `title_abstract` key.

We can now call the dataset and see if everything is correct


```python
ds = CovidPapersDataset.from_path(pr.data_dir / 'metadata.csv')

ds[0]['title']
```

    /home/francesco/anaconda3/envs/dl/lib/python3.7/site-packages/pandas/core/generic.py:6287: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._update_inplace(new_data)
    /home/francesco/anaconda3/envs/dl/lib/python3.7/site-packages/pandas/core/indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s





    'Sequence requirements for RNA strand transfer during nidovirus discontinuous subgenomic RNA synthesis'



## Embed

Now, we need a way to create a vector (*embedding*) from the data. We define a class `Embedder` that loads automatically a model from [HuggingFace's `transformers`](https://github.com/huggingface/transformers) using the [sentence_transformers](https://github.com/UKPLab/sentence-transformers) library.

The model of choice is [gsarti/biobert-nli](https://huggingface.co/gsarti/biobert-nli) a [BioBERT](https://github.com/dmis-lab/biobert) model fine-tuned on the [SNLI](https://nlp.stanford.edu/projects/snli/) and the [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) to produce [universal sentence embeddings](https://www.aclweb.org/anthology/D17-1070/). The finetuning was made by [Gabriele Sarti](https://www.gsarti.com), the code to reproduce it is available [here](https://github.com/gsarti/covid-papers-browser/blob/master/scripts/finetune_nli.py).

BioBERT is especially fit for our dataset since it was originally trained on biomedical scientific publications. So, it should create better context-aware embeddings given the similarity with our data.

Under the hood, the model first tokenizes the input string in tokens,  then it creates one vector for each one of them. So, if we have `N` tokens in one paper we will get a `[N, 768]` vector (note that a token often corresponds to a word piece, read more about tokenization strategies [here](https://www.thoughtvector.io/blog/subword-tokenization/). Thus, if two papers have a different word size, we will have two vectors with two different first dimensions. This is a problem since we need to compare them to search.

To get a fixed embed for each paper, we apply average pooling. This methodology  computes the average of each word and outputs a fixed-size vector of dims `[1, 768]`

So, let's code an `Embedder` class


```python
from dataclasses import dataclass
from sentence_transformers import models, SentenceTransformer

@dataclass
class Embedder:
    name: str = 'gsarti/scibert-nli'
    max_seq_length: int  = 128
    do_lower_case: bool  = True
    
    def __post_init__(self):
        word_embedding_model = models.BERT(
            'gsarti/biobert-nli',
            max_seq_length=128,
            do_lower_case=True
        )
        # apply pooling to get one fixed vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False
            )
    
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
    def __call__(self, text):
        return self.model.encode(text) 
```

We can try our embedder on a data point


```python
embedder = Embedder()

emb = embedder(ds[0]['title_abstract'])

emb[0].shape
```




    (768,)



Et voilà!

## Search

Okay, we now have a way to embed each paper, but how can we search in the data using a query? Assuming we have embedded **all** the papers we could also **embed the query** and compute the cosine similarity between the query and all the embeddings. Then, we can show the results sorted by the distance (score). Intuitively, the closer they are in the embedding space to the query the more context similarity they share. 

But, how? First, we need a proper way to manage the data and to run the cosine similarity fast enough. Fortunately, Elastic Search comes to the rescue!

### Elastic Search

[Elastic Search](https://www.elastic.co/) is a database with one goal, yes you guessed right: search. We will first store all the embedding in elastic and then use its API to perform the searching. If you are lazy like me you can [install elastic search with docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)

```
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.6.2
docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.6.2

```

Perfect. The next step is to store the embeddings and the papers' information on elastic search. It is a very straightforward process. We have to need to create an `index` (a new database) and then build one entry for each paper.

To create an `index` we need to describe for elastic what we wish to store. In our case:


```
{
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1
    },
    "mappings": {
        "dynamic": "true",
        "_source": {
            "enabled": "true"
        },
        "properties": {
            "title": {
                "type": "text"
            },
            ... all other properties (columns of the datafarme)
            "embed": {
                "type": "dense_vector",
                "dims": 768
            }
        }
    }
}

```


You can read more about the index creation on the elastic search [doc](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-create-index.html). The last entry defines the `embed` field as a dense vector with `768`. This is indeed our embedding. 
For convenience, I have stored the configuration in a `.json` file and created a class named `ElasticSearchProvided` to handle the storing process.



```python
import json
from pathlib import Path
from elasticsearch import Elasticsearch
from tqdm.autonotebook import tqdm
from elasticsearch.helpers import bulk

@dataclass
class ElasticSearchProvider:
    index_file: dict
    client: Elasticsearch = Elasticsearch()
    index_name: str = 'covid'

    def drop(self):
        self.client.indices.delete(index=self.index_name, ignore=[404])
        return self

    def create_index(self):
        self.client.indices.create(index=self.index_name, body=self.index_file)
        return self

    def create_and_bulk_documents(self, entries:list):
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
```

Most of the work is done in `create_and_bulk_documents` where we just deconstruct one entry at the time and add two elastic search parameters.

Unfortunately, Elastic Search won't be able to serialize the `numpy` arrays. So we need to create an adapter for our data. This class takes as input the paper data and the embedding and "adapt" them to work in our `ElasticSearchProvider`.


```python
class CovidPapersEmbeddedAdapter:
        
    def __call__(self, x, embs):
        for el, emb in zip(x, embs):
            el['embed'] = np.array(emb).tolist()

        return x
```

Okay, we have everything in place. A way to represent the data, one to encode it in a vector and a method to store the result. Let's wrap everything up and encode all the papers.


```python
dl = DataLoader(ds, batch_size=128, num_workers=4, collate_fn=lambda x: x)
es_adapter = CovidPapersEmbeddedAdapter()

import numpy as np

with open(pr.base_dir / 'es_index.json', 'r') as f:
    index_file = json.load(f)
    es_provider = ElasticSearchProvider(index_file)
    
# drop the dataset
es_provider.drop()
# create a new one
es_provider.create_index()

for batch in tqdm(dl):
    x = [b['title_abstract'] for b in batch]
    embs = embedder(x)
    es_provider(es_adapter(batch, embs))
```

There are two tricks here, first, we use `torch.utils.data.DataLoader` to create a batch-wise iterator. In general, feeding data to the model in batch rather than as a single point boost the performance (in my case x100). Second, we replace the `collate_fn` parameter in the `DataLoader` constructor. This is because, by default, Pytorch will try to cast all our data into a `torch.Tensor` but it will fail to convert strings. By doing so, we just return an array of dictionaries, the output from `CovidPapersDataset`. So, a `batch` is a list of dictionaries with length `batch_size`. After we finished (~7m on a 1080ti), we can take a look at `http://localhost:9200/covid/_search?pretty=true&q=*:*`.

If everything works correctly, you should see our data displayed by elastic search

![alt](https://github.com/FrancescoSaverioZuppichini/Search-COVID-papers-with-Deep-Learning/blob/develop/images/es_stored.jpg?raw=true)

### Make a query

We are almost done. The last piece of the puzzle is a way to search in the database. Elastic search can perform cosine similarity between one input vector and a target vector field in all the documents. The syntax is very straightforward:

```
 {
    "query": {
        "match_all": {}
    },
    "script": {
        "source":
        "cosineSimilarity(params.query_vector, doc['embed']) + 1.0",
        "params": {
            "query_vector": vector
        }
    }
}

```

Where `vector` is our input. So, we created a class to do exactly that, take a vector as an input an show all the results from the query



```python
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
                    "includes": ["title", "abstract"]
                }
            })

        return res
```

Let's see the first result (I have copy and pasted the first matching paper's abstract)


```python
es_search = ElasticSearcher()
es_search(embedder(['Effect of the virus on pregnant women'])[0].tolist())
```


*As public health professionals respond to emerging infections, particular attention needs to be paid to **pregnant women** and their offspring. Pregnant women might be more susceptible to, or more severely affected by, emerging infections. The effects of a new maternal infection on the embryo or fetus are difficult to predict. Some medications recommended for prophylaxis or treatment could harm the embryo or fetus. We discuss the challenges of responding to emerging infections among pregnant women, and we propose strategies for overcoming these challenges.*


It worked! We can now increase the readability of the output and add an input to type the query and the final result is:

![alt](https://github.com/FrancescoSaverioZuppichini/Search-COVID-papers-with-Deep-Learning/blob/develop/images/cl.png?raw=true)

We can work a little bit to increase the readability of the output and add an input to type the query and the final result is:

![alt](https://github.com/FrancescoSaverioZuppichini/Search-COVID-papers-with-Deep-Learning/blob/develop/images/cl.png?raw=true)

We can try a few queries 

## Conclusions

In this project we build a semantic browser to search on more than 50k COVID-19 papers. The original project from in which I have been working with students from the Universita of Triste is [here](https://github.com/gsarti/covid-papers-browser). A live demo is available [here](http://covidbrowser.areasciencepark.it/)

You can also play around with the command line app, you need to follow the instruction from here.
