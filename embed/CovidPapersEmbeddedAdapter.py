import numpy as np

class CovidPapersEmbeddedAdapter:
    def __call__(self, x, embs):
        for el, emb in zip(x, embs):
            el['embed'] = np.array(emb).tolist()
        return x