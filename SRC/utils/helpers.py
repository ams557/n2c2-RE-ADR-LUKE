from nltk.tokenize import sent_tokenize
import re


def find_BOS_index(sentences, pos):
    if not isinstance(sentences,list):
        sentences = sent_tokenize(sentences)
    start_index = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if start_index + sentence_length >= pos:
            return start_index
        start_index += sentence_length
    return None

def find_EOS_index(sentences, pos):
    start_idx = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if start_idx + sentence_length >= pos:
            return start_idx + sentence_length
        start_idx += sentence_length
    return None

def find_sentences_around_match(text, begin, end):
    sentences = sent_tokenize(text)
    sent_start_idx = find_BOS_index(sentences,begin)
    sent_end_idx = find_EOS_index(sentences,end)
    return text[sent_start_idx:sent_end_idx]

def combine_and_norm_lists(start_list, end_list, norm_to=0):
    return [[start_list[i] - norm_to, end_list[i] - norm_to] for i in range(len(start_list))]

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