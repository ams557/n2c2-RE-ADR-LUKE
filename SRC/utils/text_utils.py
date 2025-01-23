# -*- coding: utf-8 -*-
"""Text utilities

Example:
    To use this module, import it into your python environment.

        >>> from src.utils.text_utils import *

Source(s):
    * RELEX refine.py (https://github.com/NLPatVCU/RelEx/blob/master/relex/N2C2/refine.py)
    * MTL_Preprocessing_v1.2.py (given by Dr. McInnes)
"""

from nltk.tokenize import sent_tokenize
import nltk.tokenize.punkt as pkt
import re, string
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class CustomLanguageVars(pkt.PunktLanguageVars):
    """Custom class derived from PunktLanguageVars for sentence tokenization that preserves trailing white space.

    Args:
        pkt.PunktLanguageVars: PunktLanguageVars class

    Source(s):
        * Taken from https://stackoverflow.com/a/33153483
    """
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        (\s*)                        # capture trailing whitespace
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

sent_with_trailing_ws_tokenize = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

def find_BOS_index(sentences:list[str], i:int) -> int:
    """Find the starting index of a sentence in a text
    
    Args:
        sentences (list): A list of sentences (order preserved from text that it is derived from)
        i (int): The index of the character in the text
    Returns:
        index of the start of sentence that the ith character is in
    """
    char_index = 0
    sent_index = 0
    for sentence in sentences:
        rel_char_index = 0
        for char in sentence:
            char_index+=1
            if char_index > i:
                return i - (len(sentences[sent_index][0:rel_char_index])) - 1
            rel_char_index+=1
        sent_index+=1
  
def find_EOS_index(sentences:list[str], j:int) -> int:
    """Find the ending index of a sentence in a text
    
    Args:
        sentences (list): A list of sentences (order preserved from text that it is derived from)
        j (int): The index of the character in the text
    Returns:
        index of the end of sentence that the jth character is in
    """
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

def find_sentences_around_match(text:str,begin:int,end:int) -> str:
    """Find the segment of full sentences that contains a range of character indices w/in the text

    Args:
        text (str): Entire text
        begin: Beginning of first sentence
        end: End of last sentence
    Returns:
        Section of sentences that contains the beginning and ending characters
    """
    sentences = sent_with_trailing_ws_tokenize.tokenize(text)
    length = len(text)
    sent_start_idx = find_BOS_index(text,begin)
    sent_end_idx = find_EOS_index(text,end)
    return text[sent_start_idx:sent_end_idx]

def norm_list(list:list[int,int],norm_to:int=0) -> ndarray:
    """Normalize list to a given index

    Args:
        list: List of numbers
        norm_to: Index to normalize to

    Return:
        A numpy array w/ the normalized to the given index
    """
    return np.array([list[0] - norm_to,list[1]-norm_to],dtype=object)

def find_smallest_first_element(row:Series,list1:str,list2:str) -> int:
    """Find the smallest first element between two lists in a row of a dataframe

    Args:
        row (pd.Series): Row of dataframe
        list1, list2 (str): Key of 1st & 2nd columns to look at
    
    Return:
        The smallest first element between the two lists
    """
    e1 = row[list1][0]
    e2 = row[list2][0]
    if e1 < e2:
        return e1
    return e2

def find_largest_last_element(row:Series,list1:str,list2:str) -> int:
    """Find the largest first element between two lists in a row of a dataframe

    Args:
        row (pd.Series): Row of dataframe
        list1, list2 (str): Key of 1st & 2nd columns to look at
    
    Return:
        The largest first element between the two lists
    """
    e1 = row[list1][1]
    e2 = row[list2][1]
    if e1 > e2:
        return e1
    return e2

def read_file(path:str) -> str:
    """Read a file into memory given a path

    Args:
        path (string): The path to the file
    Returns:
        text (string): The text from the file
    
    - Adapted from RELEX refine.py
    """
    with open(path,'r',encoding='utf8') as file:
        text = file.read()
    if path.endswith('.ann'):
        text = re.sub(r'\r\n',r'\n',text)
        text = re.sub(r'\r','',text)
        text = re.sub(r'\n+',r'\n',text)
    if path.endswith('.txt'):
        text = re.sub("\n", " ", text)
    return text