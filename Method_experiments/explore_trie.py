import pickle

trie_path = "/home/lars/GENRE/data/candidate_methods_trie.pkl"

# Load Wikipedia titles trie
with open(trie_path, "rb") as f:
    trie_dict = pickle.load(f)
print("hi")
k1 = list(trie_dict.keys())
print(trie_dict[k1[0]])