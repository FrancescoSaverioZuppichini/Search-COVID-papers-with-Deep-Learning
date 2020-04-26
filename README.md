

```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


# Search COVID papers with Deep Learning
*Transformers + Elastic Search = ❤️*

Good news everyone, in this article we are not going to fit a linear regression model on the COVID-19 data! But, instead we are going to build a sematic browser using deep learning to search in more than 50k papers about the recent COVID-19 disease.  

The key idea is to encode each paper in a vector and then search using cosine similary between a query and all the encoded documents. This is the same process used to search for similar images. 

So, so our puzzle is composed by three pieces: data, a mapping from papers to vectors and a way to search.

Most of the work is based on [this project](https://github.com/gsarti/covid-papers-browser) in which I am working with students from the Universita of Trieste (Italy). A live demo is available [here](http://covidbrowser.areasciencepark.it/).


Let's get started

## Data

Everything starts with the data. We will use this [dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) from Kaggle. A list of of over 57,000 scholarly articles prepared by the White House and a coalition of leading research groups. Actually, the only file we need is `metadata.csv` that contains information about the papers and the full text of the abstract. You need to store the file inside `./dataset`.

Let's take a look


```python
import pandas as pd
from Project import Project
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



As you can see we have a bounce of information. We are obliously interested in the text columns. Working with pandas is not ideal, so let's create a `Dataset`. This will allow us to later create a `DataLoader` to perform batch-wise encoding. If you are not familiar with the Pytorch data loading ecosystem you can read more about [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)


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

In order, I have subclassed `torch.utils.data.Dataset` to create a custom dataset. The dataset is expecting a dataframe as input from which we kept only the interesting columns. Then, we removed some of the rows where the `abstract` and `title` columns matched one of the "junk" word in `FILTER_TITLE` and `FILTER_ABSTRACT` respectively. This is done because some of the articles haven't a correct title or abstract.

The dataset returns a dictionary since `pd.DataFrame` is not a support type in pytorch. To give our search engine more context, we merge the `title` and the `abstract` together, the result is stored in the `title_abstract` key.

We can now call the dataset and see if everything is correct


```python
ds = CovidPapersDataset.from_path(pr.data_dir / 'metadata.csv')

ds[0]['title']
```




    'Sequence requirements for RNA strand transfer during nidovirus discontinuous subgenomic RNA synthesis'



## Embed

Now, we need a way to create a vector (*embedding*) from the data. We define a class `Embedder` that loads automatically a model from `hugging_faces` using the [sentence_transformers](https://github.com/UKPLab/sentence-transformers) library.

The model of choice is [gsarti/biobert-nli](https://huggingface.co/gsarti/biobert-nli) a [BioBERT](https://github.com/dmis-lab/biobert) model fine-tuned on the [SNLI](https://nlp.stanford.edu/projects/snli/) and the [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) to produce [universal sentence embeddings](https://www.aclweb.org/anthology/D17-1070/). The finetuning was made by [Gabriele Sarti](https://www.linkedin.com/in/gabrielesarti/), the code is available [here](https://github.com/gsarti/covid-papers-browser/blob/master/scripts/finetune_nli.py).

BioBERT is especially fit for our dataset since it was originally trained on scientific papers. So, it should create better context-aware embeddings given the similarity with our data.

Under the hood, the model first tokenizes each word,  then it creates one vector for each one of them. So, if we have `N` words in one paper we will get a `[N, 768]` vector. Thus, if two papers have a different word size, we will have two vectors with two different first dimensions. This is a problem since we need to compare them to search.

To get a fixed embed for each paper, we apply average polling. This methodology  computes the average of each word and outputs a fixed-size vector of dims `[1, 768]`

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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-13-8a84c6b7fabd> in <module>
    ----> 1 dl = DataLoader(ds, batch_size=128, num_workers=4, collate_fn=lambda x: x)
          2 es_adapter = CovidPapersEmbeddedAdapter()
          3 
          4 import numpy as np
          5 


    NameError: name 'DataLoader' is not defined


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
class Elasticsearcher:
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


```python
es_search = Elasticsearcher()
es_search(embedder(['Effect of the virus on pregnant women'])[0].tolist())
```




    {'took': 33,
     'timed_out': False,
     '_shards': {'total': 2, 'successful': 2, 'skipped': 0, 'failed': 0},
     'hits': {'total': {'value': 10000, 'relation': 'gte'},
      'max_score': 0.7962167,
      'hits': [{'_index': 'covid',
        '_type': '_doc',
        '_id': 'wxs0snEBuCitufzFW7q1',
        '_score': 0.7962167,
        '_source': {'abstract': 'As public health professionals respond to emerging infections, particular attention needs to be paid to pregnant women and their offspring. Pregnant women might be more susceptible to, or more severely affected by, emerging infections. The effects of a new maternal infection on the embryo or fetus are difficult to predict. Some medications recommended for prophylaxis or treatment could harm the embryo or fetus. We discuss the challenges of responding to emerging infections among pregnant women, and we propose strategies for overcoming these challenges.',
         'title': 'Public Health Approach to Emerging Infections Among Pregnant Women'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': '1RsxsnEBuCitufzFUXGQ',
        '_score': 0.7454823,
        '_source': {'abstract': "A key component of the response to emerging infections is consideration of special populations, including pregnant women. Successful pregnancy depends on adaptation of the woman's immune system to tolerate a genetically foreign fetus. Although the immune system changes are not well understood, a shift from cell-mediated immunity toward humoral immunity is believed to occur. These immunologic changes may alter susceptibility to and severity of infectious diseases in pregnant women. For example, pregnancy may increase susceptibility to toxoplasmosis and listeriosis and may increase severity of illness and increase mortality rates from influenza and varicella. Compared with information about more conventional disease threats, information about emerging infectious diseases is quite limited. Pregnant women's altered response to infectious diseases should be considered when planning a response to emerging infectious disease threats.",
         'title': 'Emerging Infections and Pregnancy'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'mxsxsnEBuCitufzFA2pY',
        '_score': 0.74368256,
        '_source': {'abstract': 'Planning for a future influenza pandemic should include considerations specific to pregnant women. First, pregnant women are at increased risk for influenza-associated illness and death. The effects on the fetus of maternal influenza infection, associated fever, and agents used for prophylaxis and treatment should be taken into account. Pregnant women might be reluctant to comply with public health recommendations during a pandemic because of concerns regarding effects of vaccines or medications on the fetus. Guidelines regarding nonpharmaceutical interventions (e.g., voluntary quarantine) also might present special challenges because of conflicting recommendations about routine prenatal care and delivery. Finally, healthcare facilities need to develop plans to minimize exposure of pregnant women to ill persons, while ensuring that women receive necessary care.',
         'title': 'Pandemic Influenza and Pregnant Women'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'lBs0snEBuCitufzFFbTK',
        '_score': 0.7235798,
        '_source': {'abstract': 'Specific immunoglobulin G antibody for severe acute respiratory syndrome (SARS) coronavirus was detected in maternal blood, umbilical blood, and amniotic fluid from a pregnant SARS patient. Potential protection of fetus from infection was suggested.',
         'title': 'Specific Immunoglobulin G Antibody Detected in Umbilical Blood and Amniotic Fluid from a Pregnant Woman Infected by the Coronavirus Associated with Severe Acute Respiratory Syndrome'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'YRszsnEBuCitufzFS6FH',
        '_score': 0.71303326,
        '_source': {'abstract': 'The placenta is a highly specialized organ that is formed during human gestation for conferring protection and generating an optimal microenvironment to maintain the equilibrium between immunological and biochemical factors for fetal development. Diverse pathogens, including viruses, can infect several cellular components of the placenta, such as trophoblasts, syncytiotrophoblasts and other hematopoietic cells. Viral infections during pregnancy have been associated with fetal malformation and pregnancy complications such as preterm labor. In this minireview, we describe the most recent findings regarding virus–host interactions at the placental interface and investigate the mechanisms through which viruses may access trophoblasts and the pathogenic processes involved in viral dissemination at the maternal–fetal interface.',
         'title': 'Cellular and molecular mechanisms of viral infection in the human placenta'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'rhs1snEBuCitufzFe9co',
        '_score': 0.70733577,
        '_source': {'abstract': 'Abstract Exanthematous diseases are frequently of infectious origin, posing risks, especially for pregnant healthcare workers (HCWs) who treat them. The shift from cell-mediated (Th1 cytokine profile) to humoral (Th2 cytokine profile) immunity during pregnancy can influence the mother’s susceptibility to infection and lead to complications for both mother and fetus. The potential for vertical transmission must be considered when evaluating the risks for pregnant HCWs treating infected patients, as fetal infection can often have devastating consequences. Given the high proportion of women of childbearing age among HCWs, the pregnancy-related risks of infectious exposure are an important topic in both patient care and occupational health. Contagious patients with cutaneous manifestations often present to dermatology or pediatric clinics, where female providers are particularly prevalent, as a growing number of these physicians are female. Unfortunately, the risks of infection for pregnant HCWs are not well defined. To our knowledge, there is limited guidance on safe practices for pregnant HCWs who encounter infectious dermatologic diseases. In this article, we review several infectious exanthems, their transmissibility to pregnant women, the likelihood of vertical transmission, and the potential consequences of infection for the mother and the fetus. Additionally, we discuss recommendations with respect to avoidance, contact and respiratory precautions, and the need for treatment following exposure.',
         'title': 'Management guidelines for pregnant healthcare workers exposed to infectious dermatoses'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'jBsysnEBuCitufzFfI0M',
        '_score': 0.6966655,
        '_source': {'abstract': 'Zika virus (ZIKV) infection in humans has been associated with congenital malformations and other neurological disorders, such as Guillain-Barré syndrome. The mechanism(s) of ZIKV intrauterine transmission, the cell types involved, the most vulnerable period of pregnancy for severe outcomes from infection and other physiopathological aspects are not completely elucidated. In this study, we analyzed placental samples obtained at the time of delivery from a group of 24 women diagnosed with ZIKV infection during the first, second or third trimesters of pregnancy. Villous immaturity was the main histological finding in the placental tissues, although placentas without alterations were also frequently observed. Significant enhancement of the number of syncytial sprouts was observed in the placentas of women infected during the third trimester, indicating the development of placental abnormalities after ZIKV infection. Hyperplasia of Hofbauer cells (HCs) was also observed in these third-trimester placental tissues, and remarkably, HCs were the only ZIKV-positive fetal cells found in the placentas studied that persisted until birth, as revealed by immunohistochemical (IHC) analysis. Thirty-three percent of women infected during pregnancy delivered infants with congenital abnormalities, although no pattern correlating the gestational stage at infection, the IHC positivity of HCs in placental tissues and the presence of congenital malformations at birth was observed. Placental tissue analysis enabled us to confirm maternal ZIKV infection in cases where serum from the acute infection phase was not available, which reinforces the importance of this technique in identifying possible causal factors of birth defects. The results we observed in the samples from naturally infected pregnant women may contribute to the understanding of some aspects of the pathophysiology of ZIKV.',
         'title': 'Zika Virus Infection at Different Pregnancy Stages: Anatomopathological Findings, Target Cells and Viral Persistence in Placental Tissues'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': '3BsysnEBuCitufzF5JYQ',
        '_score': 0.6910397,
        '_source': {'abstract': 'Asthma in pregnancy is a health issue of great concern. Physiological changes and drug compliance during pregnancy can affect asthma control in varying degrees, and the control level of asthma and the side effects of asthma medications are closely related to the adverse perinatal outcomes of mother and fetus. This article provides an update on the available literature regarding the alleviating or aggravating mechanism of asthma in pregnancy, diagnosis, disease assessment, and systematic management, to provide a new guidance for physician, obstetric joint doctor, and health care practitioner.',
         'title': 'Asthma in Pregnancy: Pathophysiology, Diagnosis, Whole-Course Management, and Medication Safety'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'ThszsnEBuCitufzF6rAL',
        '_score': 0.68881047,
        '_source': {'abstract': 'Large-scale infectious epidemics present the medical community with numerous medical and ethical challenges. Recent attention has focused on the likelihood of an impending influenza pandemic caused by the H5N1 virus. Pregnant women in particular present policymakers with great challenges to planning for such a public health emergency. By recognizing the specific considerations needed for this population, we can preemptively address the issues presented by infectious disease outbreaks. We reviewed the important ethical challenges presented by pregnant women and highlighted the considerations for all vulnerable groups when planning for a pandemic at both the local and the national level.',
         'title': 'Pandemic Influenza and Pregnancy: An Opportunity to Reassess Maternal Bioethics'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'XxszsnEBuCitufzFQKDg',
        '_score': 0.68595654,
        '_source': {'abstract': 'BACKGROUND: Other than influenza, little is known about the consequences of viral acute respiratory illness (ARI) on pregnant women and fetuses. Our objectives were to determine the frequency of ARI due to respiratory viruses and the associated clinical outcomes during pregnancy. METHODS: Pregnant women in their second or third trimester were enrolled if they reported having symptoms of ARI or were healthy within the preceding 2 weeks. Nasopharyngeal secretions were evaluated for respiratory viruses by molecular diagnostic assays. Clinical outcomes were evaluated at enrollment and via a follow-up telephone-based questionnaire 2 weeks later. RESULTS: There were 155 pregnant participants, with 81 ARI cases and 91 healthy controls. Acute lower respiratory tract illness (ALRTI) was identified in 29 cases (36%). Human rhinovirus (HRV), respiratory syncytial virus (RSV), and influenza virus accounted for 75% of virus-positive cases of ALRTI. Cases with ALRTI often reported a longer duration of illness, history of allergies, symptoms of wheezing, shortness of breath, or chest pain, and use of prescription medication. Two cases with ALRTI reported decreased fetal movement; a third case with ALRTI was hospitalized. CONCLUSIONS: In over one third of ARI cases, participants had symptoms consistent with ALRTI. Infection with HRV, RSV, or influenza virus was commonly detected in patients with ALRTI. Viral ALRTI during pregnancy appears to be common and is associated with significant morbidity.',
         'title': 'A Cross-sectional Surveillance Study of the Frequency and Etiology of Acute Respiratory Illness Among Pregnant Women'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'NRw3snEBuCitufzFLwJR',
        '_score': 0.6844306,
        '_source': {'abstract': 'Abstract Resistance to infection is the ability of the host to evoke a strong immune response sufficient to eliminate the infectious agent. In contrast, maternal tolerance to the fetus necessitates careful regulation of immune responses. Successful pregnancy requires the maternal host to effectively balance the opposing processes of maternal immune reactivity and tolerance to the fetus. However, this balance can be perturbed by infections which are recognized as the major cause of adverse pregnancy outcome including pre-term labor. Select pathogens also pose a serious threat of severe maternal illness. These include intracellular and chronic pathogens that have evolved immune evasive strategies. Murine models of intracellular bacteria and parasites that mimic pathogenesis of infection in humans have been developed. While human epidemiological studies provide insight into maternal immunity to infection, experimental infection in pregnant mice is a vital tool to unravel the complex molecular mechanisms of placental infection, congenital transmission and maternal illness. We will provide a comprehensive review of the pathogenesis of several infection models in pregnant mice and their clinical relevance. These models have revealed the immunological function of the placenta in responding to, and resisting infection. Murine feto-placental infection provides an effective way to evaluate new intervention strategies for managing infections during pregnancy, adverse fetal outcome and long-term effects on the offspring and mother.',
         'title': 'From mice to women: the conundrum of immunity to infection during pregnancy'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'Zhs0snEBuCitufzF1MbH',
        '_score': 0.68372285,
        '_source': {'abstract': 'The human fetus is protected by the mother’s antibodies. At the end of the pregnancy, the concentration of maternal antibodies is higher in the cord blood, than in the maternal circulation. Simultaneously, the immune system of the fetus begins to work and from the second trimester, fetal IgM is produced by the fetal immune system specific to microorganisms and antigens passing the maternal-fetal barrier. The same time the fetal immune system has to cope and develop tolerance and T(REG) cells to the maternal microchimeric cells, latent virus-carrier maternal cells and microorganisms transported through the maternal-fetal barrier. The maternal phenotypic inheritance may hide risks for the newborn, too. Antibody mediated enhancement results in dengue shock syndrome in the first 8 month of age of the baby. A series of pathologic maternal antibodies may elicit neonatal illnesses upon birth usually recovering during the first months of the life of the offspring. Certain antibodies, however, may impair the fetal or neonatal tissues or organs resulting prolonged recovery or initiating prolonged pathological processes of the children. The importance of maternal anti-idiotypic antibodies are believed to prime the fetal immune system with epitopes of etiologic agents infected the mother during her whole life before pregnancy and delivery. The chemotherapeutical and biological substances used for the therapy of the mother will be transcytosed into the fetal body during the last two trimesters of pregnancy. The long series of the therapeutic monoclonal antibodies and conjugates has not been tested systematically yet. The available data are summarised in this chapter. The innate immunity plays an important role in fetal defence. The concentration of interferon is relative high in the placenta. This is probably one reason, why the therapeutic interferon treatment of the mother does not impair the fetal development.',
         'title': 'Fetal and Neonatal Illnesses Caused or Influenced by Maternal Transplacental IgG and/or Therapeutic Antibodies Applied During Pregnancy'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'SRs1snEBuCitufzFm9oh',
        '_score': 0.6795876,
        '_source': {'abstract': 'BACKGROUND Person to person spread of COIVD-19 in the UK has now been confirmed. There are limited case series reporting the impact on women affected by coronaviruses (CoV) during pregnancy. In women affected by SARS and MERS, the case fatality rate appeared higher in women affected in pregnancy compared with non-pregnant women. We conducted a rapid, review to guide management of women affected by COVID -19 during pregnancy and developed interim practice guidance with the RCOG and RCPCH to inform maternity and neonatal service planning METHODS Searches were conducted in PubMed and MedRxiv to identify primary case reports, case series, observational studies or randomised-controlled trial describing women affected by coronavirus in pregnancy and on neonates. Data was extracted from relevant papers and the review was drafted with representatives of the RCPCH and RCOG who also provided expert consensus on areas where data were lacking RESULTS From 9964 results on PubMed and 600 on MedRxiv, 18 relevant studies (case reports and case series) were identified. There was inconsistent reporting of maternal, perinatal and neonatal outcomes across case reports and series concerning COVID-19, SARS, MERS and other coronaviruses. From reports of 19 women to date affected by COVID-19 in pregnancy, delivering 20 babies, 3 (16%) were asymptomatic, 1 (5%) was admitted to ICU and no maternal deaths have been reported. Deliveries were 17 by caesarean section, 2 by vaginal delivery, 8 (42%) delivered pre-term. There was one neonatal death, in 15 babies who were tested there was no evidence of vertical transmission. CONCLUSIONS Morbidity and mortality from COVID-19 appears less marked than for SARS and MERS, acknowledging the limited number of cases reported to date. Pre-term delivery affected 42% of women hospitalised with COVID-19, which may put considerable pressure on neonatal services if the UK reasonable worse-case scenario of 80% of the population affected is realised. There has been no evidence of vertical transmission to date. The RCOG and RCPCH have provided interim guidance to help maternity and neonatal services plan their response to COVID-19.',
         'title': 'CORONAVIRUS IN PREGNANCY AND DELIVERY: RAPID REVIEW AND EXPERT CONSENSUS'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'Lxs2snEBuCitufzFn_S_',
        '_score': 0.67013705,
        '_source': {'abstract': 'Abstract Infections during pregnancy may affect a developing fetus. If left untreated, these infections can lead to the death of the mother, fetus, or neonate and other adverse sequelae. There are many factors that impact infection during pregnancy, such as the immune system changes during pregnancy, hormonal flux, stress, and the microbiome. We review some of the outcomes of infection during pregnancy, such as preterm birth, chorioamnionitis, meningitis, hydrocephaly, developmental delays, microcephaly, and sepsis. Transmission routes are discussed regarding how a pregnant woman may pass her infection to her fetus. This is followed by examples of infection during pregnancy: bacterial, viral, parasitic, and fungal infections. There are many known organisms that are capable of producing similar congenital defects during pregnancy; however, whether these infections share common mechanisms of action is yet to be determined. To protect the health of pregnant women and their offspring, additional research is needed to understand how these intrauterine infections adversely affect pregnancies and/or neonates in order to develop prevention strategies and treatments.',
         'title': '5.16 Infections in Pregnancy☆'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'ahsxsnEBuCitufzFt3tA',
        '_score': 0.6663562,
        '_source': {'abstract': 'BACKGROUND: Pregnancy increases susceptibility to influenza. The placenta releases an immunosuppressive endogenous retroviral protein syncytin-1. We hypothesised that exposure of peripheral monocytes (PBMCs) to syncytin-1 would impair responses to H1N1pdm09 influenza. METHODS AND FINDINGS: Recombinant syncytin-1 was produced. PBMCs from non-pregnant women (n=10) were exposed to H1N1pdm09 in the presence and absence of syncytin-1 and compared to responses of PBMCs from pregnant women (n=12). PBMCs were characterised using flow cytometry, release of interferon (IFN)-α, IFN-λ, IFN-γ, IL-10, IL-2, IL-6 and IL-1β were measured by cytometric bead array or ELISA. Exposure of PBMCs to H1N1pdm09 resulted in the release of IFN-α, (14,787 pg/mL, 95% CI 7311-22,264 pg/mL) IFN-λ (1486 pg/mL, 95% CI 756-2216 pg/mL) and IFN-γ (852 pg/mL, 95% CI 193-1511 pg/mL) after 48 hours. This was significantly impaired in pregnant women (IFN-α; p<0.0001 and IFN-λ; p<0.001). Furthermore, in the presence of syncytin-1, PBMCs demonstrated marked reductions in IFN-α and IFN-λ, while enhanced release of IL-10 as well as IL-6 and IL-1β. CONCLUSIONS: Our data indicates that a placental derived protein, syncytin-1 may be responsible for the heightened vulnerability of pregnant women to influenza.',
         'title': 'The Placental Protein Syncytin-1 Impairs Antiviral Responses and Exaggerates Inflammatory Responses to Influenza'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': '1hsxsnEBuCitufzFUXGQ',
        '_score': 0.6658698,
        '_source': {'abstract': 'Emerging infectious disease outbreaks and bioterrorism attacks warrant urgent public health and medical responses. Response plans for these events may include use of medications and vaccines for which the effects on pregnant women and fetuses are unknown. Healthcare providers must be able to discuss the benefits and risks of these interventions with their pregnant patients. Recent experiences with outbreaks of severe acute respiratory syndrome, monkeypox, and anthrax, as well as response planning for bioterrorism and pandemic influenza, illustrate the challenges of making recommendations about treatment and prophylaxis for pregnant women. Understanding the physiology of pregnancy, the factors that influence the teratogenic potential of medications and vaccines, and the infection control measures that may stop an outbreak will aid planners in making recommendations for care of pregnant women during large-scale infectious disease emergencies.',
         'title': 'Prophylaxis and Treatment of Pregnant Women for Emerging Infections and Bioterrorism Emergencies'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'qBsysnEBuCitufzFyZSE',
        '_score': 0.66248184,
        '_source': {'abstract': 'Extravillous trophoblasts (EVT) migration into the decidua is critical for establishing placental perfusion and when dysregulated, may lead to pre-eclampsia (PE) and intrauterine growth restriction (IUGR). The breast cancer resistance protein (BCRP; encoded by ABCG2) regulates the fusion of cytotrophoblasts into syncytiotrophoblasts and protects the fetus from maternally derived xenobiotics. Information about BCRP function in EVTs is limited, however placental exposure to bacterial/viral infection leads to BCRP downregulation in syncitiotrophoblasts. We hypothesized that BCRP is involved in the regulation of EVT function and is modulated by infection/inflammation. We report that besides syncitiotrophoblasts and cytotrophoblasts, BCRP is also expressed in EVTs. BCRP inhibits EVT cell migration in HTR8/SVneo (human EVT-like) cells and in human EVT explant cultures, while not affecting cell proliferation. We have also shown that bacterial—lipopolysaccharide (LPS)—and viral antigens—single stranded RNA (ssRNA)—have a profound effect in downregulating ABCG2 and BCRP levels, whilst simultaneously increasing the migration potential of EVT-like cells. Our study reports a novel function of BCRP in early placentation and suggests that exposure of EVTs to maternal infection/inflammation could disrupt their migration potential via the downregulation of BCRP. This could negatively influence placental development/function, contribute to existing obstetric pathologies, and negatively impact pregnancy outcomes and maternal/neonatal health.',
         'title': 'Breast Cancer Resistance Protein (BCRP/ABCG2) Inhibits Extra Villous Trophoblast Migration: The Impact of Bacterial and Viral Infection'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': '8hsysnEBuCitufzFgY3P',
        '_score': 0.6619046,
        '_source': {'abstract': 'In 2009, the H1N1 swine flu pandemic highlighted the vulnerability of pregnant women to influenza viral infection. Pregnant women infected with influenza A virus were at increased risk of hospitalization and severe acute respiratory distress syndrome (ARDS), which is associated with high mortality, while their newborns had an increased risk of pre-term birth or low birth weight. Pregnant women have a unique immunological profile modulated by the sex hormones required to maintain pregnancy, namely progesterone and estrogens. The role of these hormones in coordinating maternal immunotolerance in uterine tissue and cellular subsets has been well researched; however, these hormones have wide-ranging effects outside the uterus in modulating the immune response to disease. In this review, we compile research findings in the clinic and in animal models that elaborate on the unique features of H1N1 influenza A viral pathogenesis during pregnancy, the crosstalk between innate immune signaling and hormonal regulation during pregnancy, and the role of pregnancy hormones in modulating cellular responses to influenza A viral infection at mid-gestation. We highlight the ways in which lung architecture and function is stressed by pregnancy, increasing baseline inflammation prior to infection. We demonstrate that infection disrupts progesterone production and upregulates inflammatory mediators, such as cyclooxygenase-2 (COX-2) and prostaglandins, resulting in pre-term labor and spontaneous abortions. Lastly, we profile the ways in which pregnancy alters innate and adaptive cellular immune responses to H1N1 influenza viral infection, and the ways in which these protect fetal development at the expense of effective long-term immune memory. Thus, we highlight advancements in the field of reproductive immunology in response to viral infection and illustrate how that knowledge might be used to develop more effective post-infection therapies and vaccination strategies.',
         'title': 'Hormonal Regulation of Physiology, Innate Immunity and Antibody Response to H1N1 Influenza Virus Infection During Pregnancy'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'Qxs2snEBuCitufzFzvh8',
        '_score': 0.6589718,
        '_source': {'abstract': 'Background Respiratory viral infections are common in pregnancy, but their health impact, especially in asthma, is unknown. The objective of this study was to assess the frequency, severity, and consequences of respiratory viral infection during pregnancy in women with and without asthma. Methods In this prospective cohort study, common cold symptoms were assessed during pregnancy in 168 women with asthma and 117 women without asthma using the common cold questionnaire and by self-report. Nasal and throat swabs were collected for suspected infections and tested by polymerase chain reaction for respiratory viruses. Pregnancy and asthma outcomes were recorded. Results Pregnant women with asthma had more prospective self-reported and questionnaire-detected common colds than pregnant women without asthma (incidence rate ratio, 1.77; 95% CI, 1.30-2.42; P < .0001). Retrospectively reported common colds in early pregnancy and post partum were increased in women with asthma compared with women without asthma. The severity of cold symptoms was also increased in women with asthma (total cold score median, 8; interquartile range [5, 10] in women with asthma vs 6 [5, 8] in control subjects; P = .031). Among women with asthma, having a laboratory-confirmed viral infection was associated with poorer maternal health, with 60% of infections associated with uncontrolled asthma and a higher likelihood of preeclampsia. Conclusions Pregnant women with asthma have more common colds during pregnancy than pregnant women without asthma. Colds during pregnancy were associated with adverse maternal and pregnancy outcomes. Prevention of viral infection in pregnancy may improve the health of mothers with asthma.',
         'title': 'A Prospective Study of Respiratory Viral Infection in Pregnant Women With and Without Asthma'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': '6BsxsnEBuCitufzFM24_',
        '_score': 0.65569013,
        '_source': {'abstract': 'BACKGROUND: FIV infection frequently compromises pregnancy under experimental conditions and is accompanied by aberrant expression of some placental cytokines. Trophoblasts produce numerous immunomodulators that play a role in placental development and pregnancy maintenance. We hypothesized that FIV infection may cause dysregulation of trophoblast immunomodulator expression, and aberrant expression of these molecules may potentiate inflammation and compromise pregnancy. The purpose of this project was to evaluate the expression of representative pro-(TNF-α, IFN-γ, IL-1β, IL-2, IL-6, IL-12p35, IL-12p40, IL-18, and GM-CSF) and anti-inflammatory cytokines (IL-4, IL-5, and IL-10); CD134, a secondary co-stimulatory molecule expressed on activated T cells (FIV primary receptor); the chemokine receptor CXCR4 (FIV co-receptor); SDF-1α, the chemokine ligand to CXCR4; and FIV gag in trophoblasts from early-and late-term pregnancy. METHODS: We used an anti-cytokeratin antibody in immunohistochemistry to identify trophoblasts selectively, collected these cells using laser capture microdissection, and extracted total RNA from the captured cell populations. Real time, reverse transcription-PCR was used to quantify gene expression. RESULTS: We detected IL-4, IL-5, IL-6, IL-1β, IL-12p35, IL-12p40, and CXCR4 in trophoblasts from early-and late-term pregnancy. Expression of cytokines increased from early to late pregnancy in normal tissues. A clear, pro-inflammatory microenvironment was not evident in trophoblasts from FIV-infected queens at either stage of pregnancy. Reproductive failure was accompanied by down-regulation of both pro-and anti-inflammatory cytokines. CD134 was not detected in trophoblasts, and FIV gag was detected in only one of ten trophoblast specimens collected from FIV-infected queens. CONCLUSION: Feline trophoblasts express an array of pro-and anti-inflammatory immunomodulators whose expression increases from early to late pregnancy in normal tissues. Non-viable pregnancies were associated with decreased expression of immunomodulators which regulate trophoblast invasion in other species. The detection of FIV RNA in trophoblasts was rare, suggesting that the high rate of reproductive failure in FIV-infected queens was not a direct result of viral replication in trophoblasts. The influence of placental immune cells on trophoblast function and pregnancy maintenance in the FIV-infected cat requires additional study.',
         'title': 'Immunomodulator expression in trophoblasts from the feline immunodeficiency virus (FIV)-infected cat'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'jRsxsnEBuCitufzFzX1c',
        '_score': 0.65522134,
        '_source': {'abstract': 'Seasonal influenza viruses are typically restricted to the human upper respiratory tract whereas influenza viruses with greater pathogenic potential often also target extra-pulmonary organs. Infants, pregnant women, and breastfeeding mothers are highly susceptible to severe respiratory disease following influenza virus infection but the mechanisms of disease severity in the mother-infant dyad are poorly understood. Here we investigated 2009 H1N1 influenza virus infection and transmission in breastfeeding mothers and infants utilizing our developed infant-mother ferret influenza model. Infants acquired severe disease and mortality following infection. Transmission of the virus from infants to mother ferrets led to infection in the lungs and mother mortality. Live virus was also found in mammary gland tissue and expressed milk of the mothers which eventually led to milk cessation. Histopathology showed destruction of acini glandular architecture with the absence of milk. The virus was localized in mammary epithelial cells of positive glands. To understand the molecular mechanisms of mammary gland infection, we performed global transcript analysis which showed downregulation of milk production genes such as Prolactin and increased breast involution pathways indicated by a STAT5 to STAT3 signaling shift. Genes associated with cancer development were also significantly increased including JUN, FOS and M2 macrophage markers. Immune responses within the mammary gland were characterized by decreased lymphocyte-associated genes CD3e, IL2Ra, CD4 with IL1β upregulation. Direct inoculation of H1N1 into the mammary gland led to infant respiratory infection and infant mortality suggesting the influenza virus was able to replicate in mammary tissue and transmission is possible through breastfeeding. In vitro infection studies with human breast cells showed susceptibility to H1N1 virus infection. Together, we have shown that the host-pathogen interactions of influenza virus infection in the mother-infant dyad initiate immunological and oncogenic signaling cascades within the mammary gland. These findings suggest the mammary gland may have a greater role in infection and immunity than previously thought.',
         'title': 'Influenza Transmission in the Mother-Infant Dyad Leads to Severe Disease, Mammary Gland Infection, and Pathogenesis by Regulating Host Responses'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'Sxw3snEBuCitufzF5ROI',
        '_score': 0.65384275,
        '_source': {'abstract': 'Abstract Objective To describe the clinical presentation and laboratory diagnosis of pregnant women with respiratory syncytial virus (RSV) infection. Methods Pregnant women in their second and third trimester were enrolled during the course of routine prenatal care visits when they were asymptomatic within the preceding two weeks (healthy controls) or when they reported symptoms of acute respiratory illness (ARI) of ≤7 days of duration (cases). Clinical outcomes were assessed at enrollment and two weeks after. Re-enrollment was allowed. Nasal-pharyngeal secretions were evaluated for respiratory pathogens by real-time reverse transcription polymerase chain reaction (PCR). Sera were tested for RSV-specific antibody responses by Western Blot, microneutralization assay, and palivizumab competitive antibody assay. Results During the 2015–2016 respiratory virus season, 7 of 65 (11%) pregnant women with ARI at their initial enrollment and 8 of 77 (10%) pregnant women with ARI during the study period (initial or re-enrollment) had PCR-confirmed RSV infection. Four (50%) PCR-confirmed RSV ARI cases reported symptoms of a lower respiratory tract illness (LRTI), one was hospitalized. Combining PCR and serology data, the RSV attack rate at initial enrollment was 12% (8 of 65), and 13% (10 of 77) based on ARI episodes. Among healthy controls, 28 of 88 (32%) had a Western Blot profile suggestive of a recent RSV infection either in the prior and/or current season. Conclusion RSV had an attack rate of 10–13% among ambulatory pregnant women receiving routine prenatal care during the respiratory virus season. The serology results of healthy controls suggest a potentially higher attack rate. Future studies should be aware of the combined diagnostic strength of PCR and serology to identify RSV infection. As maternal RSV vaccine candidates are evaluated to protect young infants, additional priority should be placed on outcomes of pregnant women.',
         'title': 'Clinical characteristics and outcomes of respiratory syncytial virus infection in pregnant women'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'vhs1snEBuCitufzFRNGr',
        '_score': 0.6529785,
        '_source': {'abstract': 'Abstract Both clinical and experimental studies indicate that viruses can interact with the developing nervous system to produce a spectrum of neurological damage and brain malformations. Following infection of the pregnant woman, virus may indirectly or directly involve the fetus. Direct involvement is generally due to transplacental passage of the virus and invasion of fetal tissue. Resultant disease is determined by a variety of virus-host factors, including the developmental stage of the fetus at the time it is infected, the neural cell populations which are susceptible to infection, the consequent virus-infected cell interactions, and the mechanism and timing of viral clearance. There is a growing list of human viruses which injure the developing nervous system. There are also several experimental models in which congenital viral infections have been shown to result in a variety of brain malformations but with no evidence of the prior infection remaining at the time of birth.',
         'title': 'Viral infections of the developing nervous system'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'Shs0snEBuCitufzFVroh',
        '_score': 0.64972216,
        '_source': {'abstract': 'OBJECTIVE: Pregnancy is accompanied by dramatic physiologic changes in maternal plasma proteins. Characterization of the maternal plasma proteome in normal pregnancy is an essential step for understanding changes to predict pregnancy outcome. The objective of this study was to describe maternal plasma proteins that change in abundance with advancing gestational age, and determine biological processes that are perturbed in normal pregnancy. MATERIALS AND METHODS: A longitudinal study included 43 normal pregnancies that had a term delivery of an infant who was appropriate for gestational age (AGA) without maternal or neonatal complications. For each pregnancy, 3 to 6 maternal plasma samples (median=5,) were profiled to measure the abundance of 1,125 proteins using multiplex assays. Linear mixed effects models with polynomial splines were used to model protein abundance as a function of gestational age, and significance of the association was inferred via likelihood ratio tests. Proteins considered to be significantly changed were defined as having: 1) more than 1.5 fold change between 8 and 40 weeks of gestation; and 2) a false discovery rate (FDR) adjusted p-value <0.1. Gene ontology enrichment analysis was employed to identify biological processes over-represented among the proteins that changed with advancing gestation. RESULTS: 1) Ten percent (112/1,125) of the profiled proteins changed in abundance as a function of gestational age; 2) of the 1,125 proteins analyzed Glypican-3, sialic acid-binding immunoglobulin-type lectins (Siglec)-6, placental growth factor (PlGF), C-C motif (CCL)-28, carbonic anhydrase 6, Prolactin (PRL), interleukin-1 receptor 4 (IL-1 R4), dual specificity mitogen-activated protein kinase 4 (MP2K4) and pregnancy-associated plasma protein-A (PAPP-A) had more than 5 fold change in abundance across gestation. These 9 proteins are known to be involved in a wide range of both physiologic and pathologic processes, such as growth regulation, embryogenesis, angiogenesis immunoregulation, inflammation etc.; and 3) biological processes associated with protein changes in normal pregnancy included defense response, defense response to bacteria, proteolysis and leukocyte migration (FDR=10%). CONCLUSIONS: The plasma proteome of normal pregnancy demonstrates dramatic changes in both magnitude of changes and the fraction of the proteins involved. Such information is important to understand the physiology of pregnancy, development of biomarkers to differentiate normal vs. abnormal pregnancy, and determine the response to interventions.',
         'title': 'The Maternal Plasma Proteome Changes as a Function of Gestational Age in Normal Pregnancy: a Longitudinal Study'}},
       {'_index': 'covid',
        '_type': '_doc',
        '_id': 'jxw3snEBuCitufzFPgO-',
        '_score': 0.6494953,
        '_source': {'abstract': 'As new infectious diseases, such as West Nile virus, monkeypox, and severe acute respiratory syndrome (SARS) are recognized in the United States, there are critical questions about how these infectious diseases will affect pregnant women and their infants. In addition, the implications of bioterrorist attacks for exposed pregnant women need to be considered. In this article, the authors address the following questions for a number of infectious disease threats: (1) does pregnancy affect the clinical course of these novel infectious diseases?, (2) what are the implications for prophylaxis and treatment of exposed or infected pregnant women?, and (3) are these novel infectious diseases transmitted during pregnancy, labor and delivery, or breastfeeding?',
         'title': 'Emerging Infections and Pregnancy: West Nile Virus, Monkeypox, Severe Acute Respiratory Syndrome, and Bioterrorism'}}]}}



We can work a little bit to increase the readability of the output and add  input to type the query and the final result is

Pigliare le query da https://covidex.ai/

We can try a few queries 

## Conclusions

In this project we build a semantic browser to search on more than 50k COVID-19 papers. The original project from in which I have been working with students from the Universita of Triste is [here](https://github.com/gsarti/covid-papers-browser). A live demo is available [here](http://covidbrowser.areasciencepark.it/)


```python

```
