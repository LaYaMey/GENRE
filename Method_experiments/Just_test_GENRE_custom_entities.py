import sys
sys.path.append("/home/lars/fairseq")
from genre.fairseq_model import GENRE
from genre.trie import Trie


sentences_with_entities = [
    {
        "sentence": "U–Pb dates were acquired after applying a chemical abrasion pretreatment and using a TIMS instrument.",
        "mention": "TIMS",
        "entity": "CHEMICAL ABRASION ISOTOPE-DILUTION THERMAL-IONIZATION MASS SPECTROMETRY"
    },
    {
        "sentence": "High-precision isotope measurements were obtained by thermal ionization mass spectrometry with isotope dilution.",
        "mention": "thermal ionization mass spectrometry with isotope dilution",
        "entity": "ISOTOPE-DILUTION THERMAL-IONIZATION MASS SPECTROMETRY"
    },
    {
        "sentence": "We used an electron microprobe to measure major and minor elements in the mineral grains.",
        "mention": "electron microprobe",
        "entity": "ELECTRON MICROPROBE ANALYSIS"
    },
    {
        "sentence": "Zircons were ablated with a laser and analyzed via ICP-MS for trace element concentrations.",
        "mention": "ICP-MS",
        "entity": "LASER ABLATION INDUCTIVELY-COUPLED PLASMA MASS SPECTROMETRY"
    },
    {
        "sentence": "The sample surfaces were bombarded with a primary ion beam to extract secondary ions for mass analysis.",
        "mention": "secondary ions for mass analysis",
        "entity": "SECONDARY IONIZATION MASS SPECTROMETRY"
    },
    {
        "sentence": "High-precision Sr isotope ratios were acquired with a multi-collector plasma-source mass spectrometer.",
        "mention": "multi-collector plasma-source mass spectrometer",
        "entity": "MULTI-COLLECTOR INDUCTIVELY COUPLED PLASMA MASS SPECTROMETRY"
    }
]


candidate_entities = [
  "ISOTOPE-DILUTION THERMAL-IONIZATION MASS SPECTROMETRY",
  "CHEMICAL ABRASION ISOTOPE-DILUTION THERMAL-IONIZATION MASS SPECTROMETRY",
  "ELECTRON MICROPROBE ANALYSIS",
  "LASER ABLATION INDUCTIVELY-COUPLED PLASMA MASS SPECTROMETRY",
  "SECONDARY IONIZATION MASS SPECTROMETRY",
  "MULTI-COLLECTOR INDUCTIVELY COUPLED PLASMA MASS SPECTROMETRY"
]


model = GENRE.from_pretrained("/home/lars/GENRE/models/fairseq_entity_disambiguation_aidayago").eval()

trie = Trie([
        [2] + model.encode(e).tolist()[1:]
        for e in candidate_entities
    ])


# 2. Build trie from candidate entity strings
#trie = Trie(sequences=tokenized_entities)


# Load Wikipedia titles trie
#import pickle
#with open("/home/lars/GENRE/data/kilt_titles_trie_dict.pkl", "rb") as f:
#    trie = Trie.load_from_dict(pickle.load(f))

model = GENRE.from_pretrained("/home/lars/GENRE/models/fairseq_entity_disambiguation_aidayago").eval()

# Run GENRE on each sentence
for element in sentences_with_entities:
    print("====================================================================")
    sentence = element["sentence"]
    mention = element["mention"]
    correct_entity = element["entity"]
    # Replace the mention with the tagged version
    sentence = sentence.replace(mention, f"[START_ENT] {mention} [END_ENT]")
    print(sentence+"\n")
    
    results = model.sample(
        [sentence],
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
    )

    for r in results[0][:3]:  # top 3 candidates
        info = " "
        if r["text"] == correct_entity:
            info = "→"
        print(f"  {info} {r['text']} (score: {r['score'].item():.4f})")