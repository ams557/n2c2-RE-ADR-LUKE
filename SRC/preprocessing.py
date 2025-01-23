# -*- coding: utf-8 -*-
"""Conversion file for BRAT to Dataframe Needed by LUKE for Entity Pair Classification

Example:
    To use this module, import it into your python environment.

        >>> import preproccesing

Todo:
    * Handle BRAT format without relation annotations

Sources:
    * (LUKE Example) https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb#scrollTo=hDkptorP9Koh
    * (RELEX) https://github.com/NLPatVCU/RelEx/tree/master/relex
"""

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
from src.utils.text_utils import *
from tqdm import tqdm

class InvalidAnnotationError(ValueError):
    """Error raised for directories where format is invalid 
    (eg. number of .ann files is not equal to the number of .txt files)
    - Taken from RELEX
    """
    pass

def BRATtoDFconvert(path: str) -> pd.DataFrame:
    """Function to convert directory from BRAT format to dataframe usable for LUKE for entity pair classification
    Args:
        path (string): The directory to convert.
    Returns:
        df: Dataframe with the following items::
            end_idx: Position of the end of the relation w/in the entire text
            entity_spans: The starting end ending positions of the HEAD & TAIL entities normalized to the sentence length the relation is from
            match: The substring that comprises the relation
            original_article: The full-text of file that the relation is in
            sentence: The text of the sentence(s) that the relation is in
            start_idx: Position of the start of the relation w/in the entire text
            string_id: The label of the relation
    """
    annotations = {
        'entities' : pd.DataFrame(), 
        'relations' : pd.DataFrame()
    }
    # only grab files that are relevant to BRAT annotations
    files = [file for file in os.listdir(path) if file.endswith('.ann')]
    # sort files 
    files.sort(key=lambda f : os.path.splitext(f)[1])
    for file in files:
        annotation = read_file(path + '/' + file)
        annotations['entities'] = pd.concat([annotations['entities'],process_annotation(path + file)['entities']],ignore_index=True) 
        annotations['relations'] = pd.concat([annotations['relations'],process_annotation(path + file)['relations']],ignore_index=True)
    
    candidates = pd.merge(annotations['entities'], annotations['entities'], on='file', suffixes=['1','2']).query("tag1 != tag2 and (entity_name1 == 'Drug') and entity_name1 != entity_name2")
    candidates.rename(columns={'entity_span2' : 'relation_start', 'entity_span1' : 'relation_end', 'entity2' : 'start_entity', 'entity1' : 'end_entity'},inplace=True)
    candidates.drop(columns=['entity_name1', 'tag1','tag2','entity_name2'],inplace=True)
    candidates = create_entity_links(df=candidates,path=path)

    if not annotations['relations'].empty:
        annotations['relations'].drop(columns=['tag'],inplace=True)
        # Inner join relations dataframe to entities sub-dataframe on the relatio start in the correct file
        relations = pd.merge(annotations['relations'],annotations['entities'][['file','tag','entity_span','entity']],left_on=['file','relation_start'],right_on=['file','tag'])
        relations.drop(columns=['tag','relation_start'],inplace=True)
        relations.rename(columns={'entity_span' : 'relation_start','entity' : 'start_entity', 'relation_name' : 'string_id'},inplace=True)

        # Inner join the combined dataframe to get the relation end within the correct file
        relations = pd.merge(relations,annotations['entities'][['file','tag','entity_span','entity']],left_on=['file','relation_end'],right_on=['file','tag'])
        relations.drop(columns=['tag','relation_end'],inplace=True)
        relations.rename(columns={'entity_span' : 'relation_end', 'entity' :'end_entity'},inplace=True)


        relations = create_entity_links(df=relations,path=path)
        cols = ['end_idx', 'match','original_article','sentences','start_idx']

        candidates = pd.merge(candidates,relations[cols+['string_id']],on=cols,how="left")
        candidates.fillna({'string_id': 'Unrelated'}, inplace=True)

    return candidates

def grab_entity_info(line: list[str,str,str]) -> pd.DataFrame:
    """Function list of entity info from a line of BRAT format to dataframe usable for LUKE for entity pair classification
    Args:
        line (list[str,str,str]): The annotation line (for entities [prefixed T])
    Returns:
        Dataframe with the following items::
            tag: The entity tag w/in a file (eg. T1 for the first entity in that file)
            entity_name: The entity (type) in a line
            entity_span: A list of the start & end index of an entity in a text file
            entity : The entity mention
    """
    tags = line[1].split(" ") # this middle segment contains the entity label, the entity start & the entity end
    entity_name = str(tags[0])
    entity_start = int(tags[1])
    entity_end = int(tags[-1])
    return pd.DataFrame({
        'tag' : line[0], 
        'entity_name' : entity_name, 
        'entity_span' : [np.array([entity_start, entity_end],dtype=object)], 
        'entity' : line[-1]
    },index=[0],dtype=object)

def grab_relation_info(line: list[str,str,str]) -> pd.DataFrame:
    """Function list of relation info from a line of BRAT format to dataframe usable for LUKE for entity pair classification
    Args:
        line (list[str,str,str]): The annotation line (for relations [prefixed R])
    Returns:
        Dataframe with the following items::
            tag: The relation tag w/in a file (eg. R1 for the first relation in that file)
            relation_name: The relation (type)
            relation_start: The starting index of the relation
            relation_end: The ending index of the relation
    """
    tags = line[1].split(" ")
    assert len(tags) == 3, "Incorrect relation format" # from RELEX preprocessing
    relation_name = tags[0]
    relation_start = tags[1].split(':')[1]
    relation_end = tags[2].split(':')[1]
    return pd.DataFrame({
        'tag' : line[0], 
        'relation_name' : relation_name, 
        'relation_start' : relation_start, 
        'relation_end' : relation_end
    },index=[0],dtype=object)

def create_entity_links(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Function to convert 
    Args:
        df (pd.DataFrame): Dataframe to generate linking data for
        path (string): The file that metadata relates to
    Returns:
        df: Dataframe with the following items::
            end_idx: Position of the end of the relation w/in the entire text
            entity_spans: The starting end ending positions of the HEAD & TAIL entities normalized to the sentence length the relation is from
            match: The substring that comprises the relation
            original_article: The full-text of file that the relation is in
            sentence: The text of the sentence(s) that the relation is in
            start_idx: Position of the start of the relation w/in the entire text
            string_id: The label of the relation
    """
    df['entities'] = [[start,end] for start, end in zip(df['relation_start'],df['relation_end'])]
    # df.drop(columns=['relation_start','relation_end'],inplace=True)
    df['original_article'] = [read_file(path + str(file) + '.txt') for file in df['file']]
    df.drop(columns='file',inplace=True)
    # get the start idx by finding the smallest starting index of an entity in a relation
    df['start_idx'] = df.apply(lambda row : find_smallest_first_element(row, 'relation_start', 'relation_end'), axis=1)

    # get the end idx by finding the largest starting index of an entity in a relation
    df['end_idx'] = df.apply(lambda row : find_largest_last_element(row, 'relation_start', 'relation_end'), axis=1)

    # using the starting and ending indices of the relation get the match
    df['match'] = df.apply(lambda row : row['original_article'][row['start_idx']:row['end_idx']],axis=1)

    # grab all the sentences relevant to a single relation
    df['sentences'] = df.apply(lambda row : find_sentences_around_match(text=row['original_article'],begin=row['start_idx'],end=row['end_idx']),axis=1)

    # find the beginning index of a sentence
    df['BOS_idx'] = df.apply(lambda row : find_BOS_index(row['original_article'],row['start_idx']),axis=1)

    # normalize the entity spans to the length of the group of sentences that they are found in
    df['entity_spans'] = df.apply(lambda row : np.array([norm_list(row['relation_start'],row['BOS_idx']),norm_list(row['relation_end'],row['BOS_idx'])],dtype=object),axis=1)
    cols = ['end_idx', 'entities','entity_spans','match','original_article','sentences','start_idx']
    if 'string_id' in df.columns:
        cols.append('string_id')
    
    df = df[cols].astype(object)
    df.reset_index(drop=True,inplace=True)
    return df

def process_annotation(path: str) -> dict({str : pd.DataFrame, str : pd.DataFrame}):
    """Function to convert relation line (prefixed w/ R) from BRAT format to dataframe usable for LUKE for entity pair classification
    Args:
        line (string): Tab delimited line
    Returns:
        annotations (dict): Dictionary of entity & relation dataframes
    
    - Adapted from RELEX
    """
    annotations = {
        'entities' : pd.DataFrame(), 
        'relations' : pd.DataFrame()
    }
    with open(path,'r') as file:
        annotation = file.readlines()
    for line in annotation:
        line = line.strip()
        if line == "" or line.startswith("#"): # ignore lines that are empty or start w/ `#` (borrowed from RELEX)
            continue
        if "\t" not in line:
            InvalidAnnotationError("Line chunks in ANN files must be separated by tabs (See BRAT Guidelines).")
        line = line.split("\t")
        if line[0][0] == 'T':
            annotations['entities'] = pd.concat([annotations['entities'],grab_entity_info(line)],ignore_index=True)
            annotations['entities']['file'] = str(os.path.split(path)[1].replace(".ann",""))
        if line[0][0] == 'R':
            annotations['relations'] = pd.concat([annotations['relations'],grab_relation_info(line)],ignore_index=True)
            annotations['relations']['file'] = str(os.path.split(path)[1].replace(".ann",""))
    return annotations