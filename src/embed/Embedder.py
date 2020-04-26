from dataclasses import dataclass
from sentence_transformers import models, SentenceTransformer

@dataclass
class Embedder:
    """
    This class uses Hugging's face transformers library to encode a list of strings into a 768 dim vetor.
    """
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
        
    def __call__(self, text: list):
        return self.model.encode(text) 