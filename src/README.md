# Covid Search Papers
## Getting Started
We assume you have elastic search running on your machine. We provided the embeddings and the index file from here (TODO).

After you load all the data inside elastic search, you can run the command line with  `python main.py` file or build the docker (suggested so you don't have to install new libraries) container by

```
// at root level
docker build -t covid-search .
docker run --net="host" -i covid-search
```