import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
import pickle as pkl
from collections import defaultdict, Counter
from itertools import permutations, combinations
from functools import reduce
import numpy as np
import os,sys, io
from io import FileIO
import re, fnmatch
import csv
from utils.helpers import *
from tqdm import tqdm

class InvalidAnnotationError(ValueError):
    pass

def BRATtoDFconvert(path):
    annotations = {
        'entities' : pd.DataFrame(), 
        'relations' : pd.DataFrame()
    }
    files = [file for file in os.listdir(path) if file.endswith('.ann')]
    files.sort(key=lambda f : os.path.splitext(f)[1])
    for file in files:
        annotation = read_file(path + '/' + file)
        annotations['entities'] = pd.concat([annotations['entities'],process_annotation(path + file)['entities']],ignore_index=True) 
        annotations['relations'] = pd.concat([annotations['relations'],process_annotation(path + file)['relations']],ignore_index=True)
    if not annotations['relations'].empty:
        annotations['relations'].drop(columns=['tag'],inplace=True)
        df = pd.merge(annotations['relations'],annotations['entities'][['file','tag','entity_span','entity']],left_on=['file','relation_start'],right_on=['file','tag'])
        df.drop(columns=['tag','relation_start'],inplace=True)
        df.rename(columns={'entity_span' : 'relation_start','entity' : 'start_entity', 'relation_name' : 'string_id'},inplace=True)
        df = pd.merge(df,annotations['entities'][['file','tag','entity_span','entity']],left_on=['file','relation_end'],right_on=['file','tag'])
        df.drop(columns=['tag','relation_end'],inplace=True)
        df.rename(columns={'entity_span' : 'relation_end', 'entity' :'end_entity'},inplace=True)
        df['entities'] = [[start, end] for start, end in zip(df['start_entity'], df['end_entity'])]
        df.drop(columns=['start_entity','end_entity'],inplace=True)
        df['original_article'] = [read_file(path + file + '.txt') for file in df['file']]
        df.drop(columns='file')
        df['start_idx'] = df.apply(lambda row : find_smallest_first_element(row, 'relation_start', 'relation_end'), axis=1)
        df['end_idx'] = df.apply(lambda row : find_largest_last_element(row, 'relation_start', 'relation_end'), axis=1)
        df['match'] = df.apply(lambda row : row['original_article'][row['start_idx']:row['end_idx']],axis=1)
        df['sentences'] = df.apply(lambda row : find_sentences_around_match(text=row['original_article'],begin=row['start_idx'],end=row['end_idx']),axis=1)
        df['BOS_idx'] = df.apply(lambda row : find_BOS_index(row['original_article'],row['start_idx']),axis=1)
        df['entity_spans'] = df.apply(lambda row : np.array([norm_list(row['relation_start'],row['BOS_idx']),norm_list(row['relation_end'],row['BOS_idx'])],dtype=object),axis=1)
        cols = ['end_idx', 'entities','entity_spans','match','original_article','sentences','start_idx','string_id']
        df = df[cols].astype(object)
        df.reset_index(drop=True,inplace=True)
        return df
    return annotations['entities']

def grab_entity_info(line):
    tags = line[1].split(" ")
    entity_name = str(tags[0])
    entity_start = int(tags[1])
    entity_end = int(tags[-1])
    return pd.DataFrame({'tag' : line[0], 'entity_name' : entity_name, 'entity_span' : [np.array([entity_start, entity_end],dtype=object)], 'entity' : line[-1]},index=[0],dtype=object)

def grab_relation_info(line):
    tags = line[1].split(" ")
    assert len(tags) == 3, "Incorrect relation format"
    relation_name = tags[0]
    relation_start = tags[1].split(':')[1]
    relation_end = tags[2].split(':')[1]
    return pd.DataFrame({'tag' : line[0], 'relation_name' : relation_name, 'relation_start' : relation_start, 'relation_end' : relation_end},index=[0],dtype=object)

def process_annotation(path):
    annotations = {
        'entities' : pd.DataFrame(), 
        'relations' : pd.DataFrame()
    }
    with open(path,'r') as file:
        annotation = file.readlines()
    for line in annotation:
        line = line.strip()
        annotations['entities']['file'] = os.path.split(path)[1].replace(".ann","")
        if line == "" or line.startswith("#"):
            continue
        if "\t" not in line:
            InvalidAnnotationError("Line chunks in ANN files must be separated by tabs (See BRAT Guidelines).")
        line = line.split("\t")
        if line[0][0] == 'T':
            # print(f"{os.path.split(path)[1].replace(".ann","")}")
            annotations['entities'] = pd.concat([annotations['entities'],grab_entity_info(line)],ignore_index=True)
        if line[0][0] == 'R':
            # print(os.path.split(path)[1].replace(".ann",""))
            annotations['relations'] = pd.concat([annotations['relations'],grab_relation_info(line)],ignore_index=True)
        annotations['relations']['file'] = os.path.split(path)[1].replace(".ann","")
    return annotations