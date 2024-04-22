from nltk.tokenize import sent_tokenize
import nltk.tokenize.punkt as pkt
import re, string
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class CustomLanguageVars(pkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        (\s*)                        # capture trailing whitespace
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

custom_tknzr = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

def find_BOS_index(sentences, i):
    char_index = 0
    sent_index = 0
    for sentence in sentences:
        rel_char_index = 0
        for char in sentence:
            char_index+=1
            if char_index > i:
                return i - (len(sentences[sent_index][0:rel_char_index])) -1
            rel_char_index+=1
        sent_index+=1
    print(sent_index)
  
def find_EOS_index(sentences, j):
    char_index = 0
    sent_index = 0
    for sentence in sentences:
        rel_char_index = 0
        for char in sentence:
            char_index+=1
            if char_index >= j:
                return j + len(sentences[sent_index][rel_char_index:]) + 1
            rel_char_index+=1
        sent_index+=1

def find_sentences_around_match(text,begin,end):
    sentences = custom_tknzr.tokenize(text)
    length = len(text)
    sent_start_idx = find_BOS_index(text,begin)
    sent_end_idx = find_EOS_index(text,end)
    return text[sent_start_idx:sent_end_idx]

# def combine_and_norm_lists(start_list, end_list, norm_to=0):
#     return [np.array([start_list[i] - norm_to, end_list[i] - norm_to],dtype=object) for i in range(len(start_list))]
def norm_list(list,norm_to=0):
    return np.array([list[0] - norm_to,list[1]-norm_to],dtype=object)

def find_smallest_first_element(row,list1,list2):
    e1 = row[list1][0]
    e2 = row[list2][0]
    if e1 < e2:
        return e1
    return e2

def find_largest_last_element(row,list1,list2):
    e1 = row[list1][1]
    e2 = row[list2][1]
    if e1 > e2:
        return e1
    return e2

def read_file(path):
    with open(path,'r',encoding='utf8') as file:
        text = file.read()
    if path.endswith('.ann'):
        text = re.sub(r'\r\n',r'\n',text)
        text = re.sub(r'\r','',text)
        text = re.sub(r'\n+',r'\n',text)
    if path.endswith('.txt'):
        text = re.sub("\n", " ", text)
    return text


# def find_BOS_index(sentences, pos):
#     if not isinstance(sentences,list):
#         sentences = sent_tokenize(sentences)
#     char_index = 0
#     for sentence in sentences:
#         len_sentence = len(sentence)
#         if char_index + len_sentence >= pos:
#             return char_index
#         char_index += len_sentence + 1
#     return char_index

# def find_EOS_index(sentences, pos):
#     char_index = 0
#     for sentence in sentences:
#         len_sentence = len(sentence)
#         if pos <= char_index + len_sentence:
#             return char_index + len_sentence
#         char_index += len_sentence + 1
#     return char_index - 1

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = logits.argmax(-1)
    print(confusion_matrix(labels, preds))
    result = {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
    return result