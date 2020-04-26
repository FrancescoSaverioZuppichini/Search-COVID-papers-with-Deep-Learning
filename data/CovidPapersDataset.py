from torch.utils.data import Dataset
import pandas as pd

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