import sys
sys.path.insert(0, '~/git/ShakespeareanPoemGeneration⁩/CS155_SET6⁩/release/code/HMM.py')
from HMM.py import unsupervised_HMM
from HMM_helper.py import (
	text_to_wordcloud,
	states_to_wordcloud,
	states_to_wordclouds,
	parse_observations,
	sample_sentence,
	visualize_sparsities,
	animate_emission,
	obs_map_reverser)

# Data Preprocessing
text = open(os.path.join(os.getcwd(), '~/git/ShakespeareanPoemGeneration/Miniproject_3/data/shakespeare.txt')).read()
print(text)