import json
import string
import random
import time
## for vectorizer
from dtw import *
from sklearn import feature_extraction, manifold
## for word embedding
import gensim.downloader as gensim_api
## for topic modeling
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

import matplotlib.pyplot as plt 
from scipy.stats import kendalltau
from scipy.spatial import distance
import scipy
from rouge import Rouge 
import pymeteor.pymeteor as pymeteor
import pickle
from pprint import pprint as pp

from config import CONFIG


FAST_DEBUG = False

if not FAST_DEBUG:
    embdr = gensim_api.load("glove-wiki-gigaword-300")
    # with open('./glove.pickle', 'rb') as handle:
    #     embdr = pickle.load(handle)
rouge = Rouge()
def get_embedding(word):
    if FAST_DEBUG:
        return np.random.random(size=300)
    if word in embdr.wv.vocab:
        return embdr[word]
    else:
        return np.zeros(300)


def header_print(message):
  print(f'{"*" * 20} {message} {"*" * 20}')
def title_print(message):
  print(f'\n\n{"<>" * 20} {message} {"<>" * 20}')

class EmbeddedObject:
    def __init__(self, doc, reference, predicted) -> None:
        self.doc = doc
        self.doc_sentences = self.text_preprocess(doc)
        self.doc_emb_sentences = self.emb(self.doc_sentences)

        self.reference = reference
        self.reference_sentences = self.text_preprocess(reference)
        self.reference_emb_sentences = self.emb(self.reference_sentences)

        self.predicted = predicted
        self.predicted_sentences = self.text_preprocess(predicted)
        self.predicted_emb_sentences = self.emb(self.predicted_sentences)
    
    def emb(self, sentences):
        sentences_embeddings = []
        for sentence in sentences:
            sentence_embedding = []
            for word in sentence:
                sentence_embedding.append(get_embedding(word))
            if len(np.array(sentence_embedding).shape) == 2 and np.array(sentence_embedding).shape[1] == 300:
                sentences_embeddings.append(np.array(sentence_embedding))
        
        return sentences_embeddings

    def text_preprocess(self, text):
        """
            takes in raw text from the dataset where sentences are denoted by <s> </eos>
        """
        sentences = []
        sentence = []
        punct = string.punctuation
        lemmatizer = WordNetLemmatizer()
        for word in str.split(text, " "):
            if word == '<s>':
                sentence = []
            elif word == '</eos>':
                sentences.append(sentence)
                sentence = []
            elif word in punct:
                pass
            elif word == "</s>":
                break
            else:
                sentence.append(word.lower())
        return sentences

    def rouge_score(self):
        return rouge.get_scores(self.reference, self.predicted)
    
    def meteor_score(self):
        start = time.time()
        ref_sentences = [" ".join(sent) for sent in self.reference_sentences]
        pred_sentences = [" ".join(sent) for sent in self.predicted_sentences]


        meteor_matrix = np.zeros((len(pred_sentences), len(ref_sentences)))
        for i, sent1 in enumerate(pred_sentences):
            for k, sent2 in enumerate(ref_sentences):
                meteor_matrix[i, k] = pymeteor.meteor(sent1, sent2)

        meteor_argmaxes = meteor_matrix.argmax(axis=1)
        meteor_maxes = meteor_matrix.max(axis=1)
        return (len(pred_sentences), len(ref_sentences), meteor_argmaxes, meteor_maxes)
    
    def dtw_score(self, revers=False):
        if revers:
            query1 = np.array([np.mean(sent, axis=0) for sent in reversed(self.reference_emb_sentences)])
            query2 = np.array([np.mean(sent, axis=0) for sent in reversed(self.predicted_emb_sentences)])
        else:
            query1 = np.array([np.mean(sent, axis=0) for sent in self.reference_emb_sentences])
            query2 = np.array([np.mean(sent, axis=0) for sent in self.predicted_emb_sentences])
        template = np.array([np.mean(sent, axis=0) for sent in self.doc_emb_sentences])
        
        return (
            dtw(query1, template).distance if query1.shape[0] > 1 else None, 
            dtw(query2, template).distance if query2.shape[0] > 1 else None, 
            dtw(query1, query2).distance if query1.shape[0] > 1 and query2.shape[0] > 1 else None
        )
    
    def other_alignment(self, revers=False):
        def spearmans_ordering_coefficient(numbers):
            # Compute the ranks of the elements in the array
            ranks = np.argsort(numbers).argsort() + 1
            # Calculate the Spearman's rank correlation coefficient
            return 1 - 6*np.sum((ranks - np.arange(1, len(ranks)+1))**2) / (len(ranks)*(len(ranks)**2 - 1))

        if revers:
            query1 = np.array([np.mean(sent, axis=0) for sent in reversed(self.reference_emb_sentences)])
            query2 = np.array([np.mean(sent, axis=0) for sent in reversed(self.predicted_emb_sentences)])
        else:
            query1 = np.array([np.mean(sent, axis=0) for sent in self.reference_emb_sentences])
            query2 = np.array([np.mean(sent, axis=0) for sent in self.predicted_emb_sentences])
        template = np.array([np.mean(sent, axis=0) for sent in self.doc_emb_sentences])

        # ref_argmax = distance.cdist(query1, template).argmin(axis=1)
        # pred_argmax = distance.cdist(query2, template).argmin(axis=1)
        # refpred_argmax = distance.cdist(query2, query1).argmin(axis=1)

        return (
            spearmans_ordering_coefficient(distance.cdist(query1, template).argmin(axis=1)) if len(query1.shape) == 2 else None, 
            spearmans_ordering_coefficient(distance.cdist(query2, template).argmin(axis=1)) if len(query2.shape) == 2 else None, 
            spearmans_ordering_coefficient(distance.cdist(query2, query1).argmin(axis=1)) if len(query1.shape) == 2 and len(query2.shape) == 2 else None
        )


    def to_string(self):
        title_print("Showing the operations for a document")

        header_print("Here is the document")
        print(self.doc)
        header_print("Here are the sentence embedding matrices")
        print([sent.shape for sent in self.doc_emb_sentences])

        header_print("Here is the reference")
        print(self.reference)
        header_print("Here are the sentence embedding matrices")
        print([sent.shape for sent in self.reference_emb_sentences])

        header_print("Here is the predicted summary")
        print(self.predicted)
        header_print("Here are the sentence embedding matrices")
        print([sent.shape for sent in self.predicted_emb_sentences])


def avg_sentence_length(emb_obs):
    reference_sent_lengths = [len(ob.reference_emb_sentences) for ob in emb_obs]
    predicted_sent_lengths = [len(ob.predicted_emb_sentences) for ob in emb_obs]
    doc_sent_lengths = [len(ob.doc_emb_sentences) for ob in emb_obs]
    return {
        'ref': scipy.histogram(reference_sent_lengths, bins=range(max(reference_sent_lengths))),
        'pred': scipy.histogram(predicted_sent_lengths, bins=range(max(predicted_sent_lengths))),
        'doc': scipy.histogram(doc_sent_lengths, bins=range(max(doc_sent_lengths))),
    }

def avg_rouge(emb_obs):
    hyps, refs = map(list, zip(*[[d.predicted, d.reference] for d in emb_obs]))
    return rouge.get_scores(hyps, refs, avg=True)

def avg_meteor(emb_obs):
    meteor_tuples = [emb_ob.meteor_score() for emb_ob in emb_obs]
    collision_probs = [np.unique(argmaxes).shape[0] / ref_len for pred_len, ref_len, argmaxes, maxes in meteor_tuples]
    collision_probs_mean = np.mean(collision_probs)
    collision_probs_std = np.std(collision_probs)

    meteor_mean = np.mean([np.mean(maxes) for pred_len, ref_len, argmaxes, maxes in meteor_tuples if maxes.shape[0] > 1])
    meteor_std = np.mean([np.std(maxes) for pred_len, ref_len, argmaxes, maxes in meteor_tuples if maxes.shape[0] > 1])
    return {
        'collision_probability': (collision_probs_mean, collision_probs_std), 
        'meteor_metric': (meteor_mean, meteor_std)
    }

def avg_dtw(emb_obs, min_num_sentences=3, revers=False):
    ref_dtw = [
        emb_ob.dtw_score(revers)[0] for emb_ob in emb_obs \
            if emb_ob.dtw_score(revers)[0] is not None and len(emb_ob.reference_emb_sentences) == min_num_sentences
    ]
    pred_dtw = [
        emb_ob.dtw_score(revers)[1] for emb_ob in emb_obs \
            if emb_ob.dtw_score(revers)[1] is not None and len(emb_ob.predicted_emb_sentences) == min_num_sentences
    ]
    refpred_dtw = [
        emb_ob.dtw_score(revers)[2] for emb_ob in emb_obs \
            if emb_ob.dtw_score(revers)[2] is not None and len(emb_ob.predicted_emb_sentences) == min_num_sentences
    ]
    return {
        'ref -> doc': (np.mean(ref_dtw), np.std(ref_dtw)), 
        'pred -> doc': (np.mean(pred_dtw), np.std(pred_dtw)), 
        'ref -> pred': (np.mean(refpred_dtw), np.std(refpred_dtw))
    }

def avg_argmax_ordering(emb_obs, num_sentences=3, revers=False):
    ref_alignment = [
        emb_ob.other_alignment(revers)[0] for emb_ob in emb_obs \
            if emb_ob.dtw_score(revers)[0] is not None and len(emb_ob.reference_emb_sentences) == num_sentences
    ]
    pred_alignment = [
        emb_ob.other_alignment(revers)[1] for emb_ob in emb_obs \
            if emb_ob.dtw_score(revers)[1] is not None and len(emb_ob.predicted_emb_sentences) == num_sentences
    ]
    refpred_alignment = [
        emb_ob.other_alignment(revers)[2] for emb_ob in emb_obs \
            if emb_ob.dtw_score(revers)[2] is not None and len(emb_ob.predicted_emb_sentences) == num_sentences
    ]
    return {
        'ref -> doc': (np.mean(ref_alignment), np.std(ref_alignment)), 
        'pred -> doc': (np.mean(pred_alignment), np.std(pred_alignment)), 
        'ref -> pred': (np.mean(refpred_alignment), np.std(refpred_alignment))
    }


def plot_alignment_over_num_sentences(
        data, X, xlabel, ylabel, title, filename, ax=plt
    ):
    plt.clf()
    N = len(data['ref -> doc'])
    ind = np.arange(N) 
    width = 0.25
    
    bars = []
    labels = []
    for i, (color, (label, data)) in enumerate(zip(['r','g','b'], data.items())):
        labels.append(label)
        bars.append(plt.bar(ind + i * width, data, width, color = color, label=label))
    
    plt.xticks(ind+width, X)
    plt.legend( tuple(bars), tuple(labels) )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # plt.legend( (bar1, bar2, bar3), ('Player1', 'Player2', 'Player3') )
    plt.savefig(f'./plots/{filename}')


if __name__ == '__main__':

    # for i, (json_filename1, _) in enumerate(CONFIG.INPUT_JSONS):
    #     with open(json_filename1) as f:
    #         input_json1 = json.loads(f.read())[:600]
    #     embedded_objects1 = [EmbeddedObject(ob["input"], ob['target'], ob['pred']) for ob in input_json1[:590]]
    #     for i, (json_filename2, _) in enumerate(CONFIG.INPUT_JSONS):
    #         with open(json_filename2) as f:
    #             input_json2 = json.loads(f.read())[:600]
    #         embedded_objects2 = [EmbeddedObject(ob["input"], ob['target'], ob['pred']) for ob in input_json2[:590]]
    #         b = False
            
    #         min_sent = 3
    #         ref_dtw1 = [
    #             emb_ob.dtw_score()[0] for emb_ob in embedded_objects1 \
    #                 if emb_ob.dtw_score()[0] is not None and len(emb_ob.reference_emb_sentences) == min_sent
    #         ]
    #         ref_dtw2 = [
    #             emb_ob.dtw_score()[0] for emb_ob in embedded_objects2 \
    #                 if emb_ob.dtw_score()[0] is not None and len(emb_ob.reference_emb_sentences) == min_sent
    #         ]
    #         if ref_dtw1 != ref_dtw2:
    #             print(f"\nerror ({json_filename1}, {json_filename2})")
    #             print(f"length of {json_filename1} ref-dtw = {len(ref_dtw1)}")
    #             print(f"length of {json_filename2} ref-dtw = {len(ref_dtw2)}")
    #             continue

    # exit(1)
    for i, (json_filename, model_name) in enumerate(CONFIG.INPUT_JSONS):
        print(f"\n\n\n{'*' * 60}\n{'*' * 60}\nJSON IS {json_filename}\n{'*' * 60}\n{'*' * 60}\n")
        with open(json_filename) as f:
            input_json = json.loads(f.read())

        embedded_objects = [EmbeddedObject(ob["input"], ob['target'], ob['pred']) for ob in input_json[:590]]
        # embedded_objects[0].to_string()

        header_print("\nAverage sentence length for dataset is!\n")
        pp(avg_sentence_length(embedded_objects))

        header_print("\nAverage rouge score for dataset is!\n")
        pp(avg_rouge(embedded_objects))

        # header_print("\nMeteor score for dataset is!")
        # print(avg_meteor(embedded_objects))

        x_axis = [2, 3, 4, 5, 6, 7]
        dtw_data = {
            'ref -> doc': [],
            'pred -> doc': [],
            'ref -> pred': [],
        }
        argmax_data = {
            'ref -> doc': [],
            'pred -> doc': [],
            'ref -> pred': [],
        }
        for num_sentence_min in x_axis:
            header_print(f"\nNum sentence alignment evaluation: {num_sentence_min}\nref-pred is a min-number, others are equivalencies\n")
            print(f"\nDTW score for dataset is (only for {num_sentence_min})!")
            dtw_result = avg_dtw(embedded_objects, min_num_sentences=num_sentence_min)
            pp(dtw_result)
            
            print(f"\nordinal argmax alignment score for dataset is (only for {num_sentence_min})!")
            argmax_result = avg_argmax_ordering(embedded_objects, num_sentences=num_sentence_min)
            pp(argmax_result)

            for source, dest in [(dtw_result, dtw_data), (argmax_result, argmax_data)]:
                dest['ref -> doc'].append(source['ref -> doc'][0])
                dest['pred -> doc'].append(source['pred -> doc'][0])
                dest['ref -> pred'].append(source['ref -> pred'][0])

        plot_alignment_over_num_sentences(
            data=dtw_data,
            X=x_axis,
            xlabel='Number of Sentences Evaluated',
            ylabel="DTW Distance",
            title=f"DTW Distance for {model_name} model",
            filename='DTW-' + model_name
        )
        
        plot_alignment_over_num_sentences(
            data=argmax_data,
            X=x_axis,
            xlabel='Number of Sentences Evaluated',
            ylabel="Spearmans Ordinal Rank Coefficient",
            title=f"Spearmans Ordinal Rank Alignment for {model_name} model",
            filename='Spearmans-' + model_name
        )