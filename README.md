# COVID Search Papers

![alt](https://github.com/FrancescoSaverioZuppichini/Search-COVID-papers-with-Deep-Learning/blob/develop/medium/images/cl.gif?raw=true)
A semantic browser that uses deep learning to search in more than 50k papers about the recent COVID-19 disease. 


It uses a deep learning model from [HuggingFace's `transformers`](https://github.com/huggingface/transformers)  to embed each paper in a fixed `[768]` vector. We load all the papers + the embeddings into Elastic Search. Search is performed by computing cosine similarity between a query and all the documents' embedding.

My medium article []

## Getting Started
We assume you have Elastic Search installed and running on your machine. We provided the embeddings and the index file from here (TODO). 

### Fill up the database

To recreate the database you have to first install [elasticsearch-dump](https://github.com/taskrabbit/elasticsearch-dump) 

Then, download the mapping and the data files from [here](https://drive.google.com/file/d/1ab_1e7lPOjQ4my3ok-7ARvBIwkJyJ8f_/view?usp=sharing) and unzip. Fire up a terminal an `cd` in the unzipped folder, from there run:

```
elasticdump \
--input=./covid_mapping.json \
--output=http://localhost:9200/covid \
--type=mapping
```

and

```
elasticdump \
--input=./covid_data.json \
--output=http://localhost:9200/covid \
--type=data
```

This may take a while.

### Run command line interface
#### Python
Run

```
pip install -r requirements.txt
python main.py
```

#### Docker (suggested)
To create the container run

```
// at root level
docker build -t covid-search .
docker run --net="host" -i covid-search
```

### Dump the database
We dumped the database using [elasticsearch-dump](https://github.com/taskrabbit/elasticsearch-dump) by running

```
elasticdump \
  --input=http://localhost:9200/covid \
  --output=./covid_mapping.json \               
  --type=mapping
```

and 

```
elasticdump \
  --input=http://localhost:9200/covid \
  --output=./covid_data.json \               
  --type=data
```