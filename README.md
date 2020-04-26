# Covid Search Papers

![alt](https://github.com/FrancescoSaverioZuppichini/Search-COVID-papers-with-Deep-Learning/blob/develop/images/cl.gif?raw=true)

A sematic browser that uses deep learning to search in more than 50k papers about the recent COVID-19 disease.  

My medium article []

## Getting Started
We assume you have elastic search running on your machine. We provided the embeddings and the index file from here (TODO). 

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
You can create the container by run

```
// at root level
docker build -t covid-search .
docker run --net="host" -i covid-search
```

### Dump the database
We dump the database using [elasticsearch-dump](https://github.com/taskrabbit/elasticsearch-dump) by running

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