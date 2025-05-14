from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from functools import lru_cache
# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Get all supersense classes from WordNet
def get_supersense_classes() -> list[str]:
    """Extract all supersense classes from WordNet."""
    supersenses = set()
    for synset in wordnet.all_synsets():
        if hasattr(synset, "lexname"):
            supersenses.add(synset.lexname())
    return sorted(list(supersenses))

SUPERSENSE_CLASSES = get_supersense_classes()
SUPERSENSE_TO_ID = {
    supersense: idx for idx, supersense in enumerate(SUPERSENSE_CLASSES)
}
NUM_SUPERSENSE_CLASSES = len(SUPERSENSE_CLASSES)

@lru_cache(maxsize=200000)
def get_word_supersenses(word) -> list[int]:
    # Lemmatize the word
    word = word.strip()
    word = lemmatizer.lemmatize(word)
    synsets = wordnet.synsets(word)
    s = set(synset.lexname() for synset in synsets if synset is not None)
    s_ids = {SUPERSENSE_TO_ID[s] for s in s if s in SUPERSENSE_TO_ID}
    if not s_ids:
        return []
    else:
        encoded = [0] * NUM_SUPERSENSE_CLASSES
        for s_id in s_ids:
            encoded[s_id] = 1
    return encoded
