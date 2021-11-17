from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# Convert
input_file = 'glove.6B.300d.txt'
output_file = 'gensim_glove.6B.300d.txt'
glove2word2vec(input_file, output_file)


# Test Glove model
model = KeyedVectors.load_word2vec_format(output_file, binary=False)
word = 'cat'
print(word)
print('Most similar:\n{}'.format(model.most_similar(word)))