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

# Data Preprocessing
text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
obs, obs_map = parse_observations_af(text)
print(obs[0])
print(obs_map)
