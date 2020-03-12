import os
import numpy as np
import re
import matplotlib.pyplot as plt
from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission,
    obs_map_reverser)
def parse_observations_af(text):
    '''This edit of parse_observations'''
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]
    # test comment
    num_array = np.arange(1,155).astype(str).tolist()
    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []

        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
            cur_marker = 0
            if (word not in obs_map) and (word not in num_array):
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
                cur_marker = 1

            # Add the encoded word.
            if cur_marker:
                obs_elem.append(obs_map[word])

        # Add the encoded sequence.
        if cur_marker:
            obs.append(obs_elem)

    return obs, obs_map

def sonnet_parser_af(text):
    '''Separates sonnets into mapped observation words.  Note, use edited shakespear by Alex which deletes the sonnet with
    too few lines, and the other with too many lines.  All should have 14 now'''
    lines = [line.split() for line in text.split('\n') if line.split()]
    obs = []
    obs_map = {}
    obs_counter = 0
    i_start =0
    i_stop = 15
    while i_stop < len(lines)+14:
        obs_elem = []
        for line in lines[i_start+1: i_stop]:
            for word in line:
                word = re.sub(r'[^\w]', '', word).lower()
                if word not in obs_map:
                    # Add unique words to the observations map.
                    obs_map[word] = obs_counter
                    obs_counter += 1
                obs_elem.append(obs_map[word])
        obs.append(obs_elem)
        i_start += 15
        i_stop += 15
    return obs, obs_map
# 
# text = open(os.path.join(os.getcwd(), 'data/shakespeare_af.txt')).read()
# obs, obs_map = sonnet_parser_af(text)
# print(obs[-1])
# sonnet = []
# reverser = obs_map_reverser(obs_map)
# for token in obs[-1]:
#     sonnet.append(' ' + reverser[token])
# print(" ".join(sonnet))
