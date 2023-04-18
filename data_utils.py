import csv 
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
#import contractions
import re 
import string
from tqdm import tqdm 

nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove("not")
lemmatizer = WordNetLemmatizer()

TRAIN_PATH = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/train.csv'
VAL_PATH = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/validation.csv'
TEST_PATH = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/test.csv'
TRAIN_P_PATH = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/train_processed.csv'
VAL_P_PATH = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/validation_processed.csv'
TEST_P_PATH = '/mnt/raptor/shihpo/cs6120/project/kaggle/cnn_dailymail/test_processed.csv'

def open_csv_file(file_path):
    ids_list = []
    articles_list = []
    highlights_list = []
    with open(file_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count==0:
                col_names = row
            else:
                if len(row[1])>0 and len(row[2])>0:
                    articles_list.append(row[1])
                    highlights_list.append(row[2])
                    ids_list.append(row[0])
                else:
                    print('row skipped.')

            line_count += 1

    return ids_list, articles_list, highlights_list

def preprocess_entry(in_str):
    #print(in_str)
    # Setence tokenization
    sents_list = sent_tokenize(in_str)
    #print(sents_list)
    # Lower case  all words
    sents_list = [s.lower() for s in sents_list]
    # Expand contraction
    #sents_list = [contractions.fix(s) for s in sents_list]
    # Remove text in paranthesis and brackets
    sents_list = [re.sub("[\(\[].*?[\)\]]", "", s) for s in sents_list]
    sents_list = [re.sub(r'[^a-zA-z0-9.,!?/:;\"\'\s]', '', s) for s in sents_list]
    # Remove punctuation
    sents_list = [''.join([c for c in s if c not in string.punctuation]) for s in sents_list]
    # Word tokenization
    tokenized_sents_list = [word_tokenize(s) for s in sents_list]
    # Remove stop words
    tokenized_sents_list = [[word for word in s if word not in stopwords] for s in tokenized_sents_list]
    # Lemmatization
    tokenized_sents_list = [[lemmatizer.lemmatize(word) for word in s] for s in tokenized_sents_list]
    # Remove sentences with very low length
    tokenized_sents_list = [s for s in tokenized_sents_list if len(s)>2]
    # Add special start and end tokens to summaries
    tokenized_sents_list2 = []
    lc = 0
    if len(tokenized_sents_list)>0:
        
        for sent in tokenized_sents_list:
            if lc == 0:
                s = ['<s>']
                lc += 1 
            else:
                s = []

            for w in sent:
                s.append(w)
            s.append('</eos>')
            tokenized_sents_list2.append(s)
        tokenized_sents_list2[-1].append('</s>')
    
    return tokenized_sents_list2


class GloveEmb():
    def __init__(self, file_path='/mnt/raptor/parnian/glove/glove.6B.300d.txt'):
        self.embedding_dict = {}
        self.word2id = {}
        self.id2word = {}

        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f:
                word = line.split()[0]
                emb = np.asarray(line.split()[1:], "float32")
                self.embedding_dict[word] = emb

        self.vocab = self.embedding_dict.keys()
        glove_v_size = len(self.vocab)
        vocab_size = glove_v_size + 4
        self.vocab.append('<s>')
        self.vocab.append('</s>')
        self.vocab.append('</eos>')
        self.vocab.append('<UNK>')

        for i in range(vocab_size):
            self.word2id[i] = self.vocab[i]
            self.id2word[self.vocab[i]] = i
    
    def get_id2word(self, id):
        return self.id2word[id]

    def get_word2id(self, word):         
        if word in self.vocab:
            return self.word2id[word]
        else:
            return self.word2id['<UNK>']     
    
    def get_embedding(self, word):
        if word in self.embedding_dict.keys():
            return self.embedding_dict[word]
        else:
            return np.zeros(300)
        
def process_data(split='train'):
    if split=='train':
        path = TRAIN_PATH
    elif split=='val':
        path = VAL_PATH
    elif split=='test':
        path = TEST_PATH
    
    tr_ids, tr_articles, tr_highlights = open_csv_file(path)
    #print(preprocess_entry(tr_articles[64142]))
    #print(tr_highlights[64142])
    #print(preprocess_entry(tr_highlights[64142]))
    
    for i in tqdm(range(len(tr_articles))):
        tr_articles[i] = preprocess_entry(tr_articles[i])
        tr_highlights[i] = preprocess_entry(tr_highlights[i])

    return tr_ids, tr_articles, tr_highlights 

def save_processed(ids, articles, highlights, split):
    if split=='train':
        path = TRAIN_P_PATH
    elif split=='val':
        path = VAL_P_PATH
    elif split=='test':
        path = TEST_P_PATH

    with open(path, 'w') as csv_file:
        col_names = ['id', 'article', 'highlights']
        writer = csv.DictWriter(csv_file, fieldnames=col_names)

        writer.writeheader()
        for i in tqdm(range(len(ids))):
            if len(articles[i])<1 or len(highlights[i])<1:
                a = ' '
                h = ' '
            else:
                a = ' '.join([ ' '.join(sent) for sent in articles[i]])
                h = ' '.join([ ' '.join(sent) for sent in highlights[i]])
            

            writer.writerow({'id':ids[i], 'article': a, 'highlights': h})
     
    
if __name__=="__main__":
    # Process and save the data
    ids, arts, highs = process_data('train')
    save_processed(ids, arts, highs, 'train')
    # Load the processed data
    ids2, arts2, highs2 = open_csv_file(TRAIN_P_PATH)
    """print(len(arts), len(arts2))
    for i in range(10):
        if arts2[i]!=' '.join([ ' '.join(sent) for sent in arts[i]]):
            print('not match.')
        if highs2[i]!=' '.join([ ' '.join(sent) for sent in highs[i]]):
            print('not match.')"""

    ids, arts, highs = process_data('val')
    save_processed(ids, arts, highs, 'val')
    ids2, arts2, highs2 = open_csv_file(VAL_P_PATH)

    ids, arts, highs = process_data('test')
    save_processed(ids, arts, highs, 'test')
    ids2, arts2, highs2 = open_csv_file(TEST_P_PATH)
    
    
