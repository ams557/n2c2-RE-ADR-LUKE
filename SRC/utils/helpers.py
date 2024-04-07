from nltk.tokenize import sent_tokenize
import re, string

def find_BOS_index(sentences, pos):
    if not isinstance(sentences, list):
        sentences = sent_tokenize(sentences)
    sentence_start = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if sentence_start + sentence_length >= pos:
            break
        sentence_start += sentence_length
    return max(0,sentence_start)

# def find_BOS_index(text, pos, length):
#     if pos >= length:
#         return None
#     sentence_start = pos
#     while sentence_start > 0 and text[sentence_start] not in '?!.':
#         sentence_start -= 1
#     return max(0,sentence_start + 2)

# def find_EOS_index(text,pos,length):
#     if pos >= length:
#         return None
#     sentence_end = pos
#     while sentence_end < length and text[sentence_end] not in '?!.':
#         sentence_end+=1
#     return min(sentence_end - 1,length)


def find_EOS_index(sentences, pos, length):
    start_idx = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if start_idx + sentence_length >= pos:
            break
        start_idx += sentence_length
    return min(start_idx + sentence_length,length)

def find_sentences_around_match(text,begin,end):
    sentences = sent_tokenize(text)
    length = len(text)
    sent_start_idx = find_BOS_index(text,begin)
    sent_end_idx = find_EOS_index(text,end,length)
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