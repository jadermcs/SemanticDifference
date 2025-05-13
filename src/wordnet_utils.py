from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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

@lru_cache(maxsize=200000)
def get_word_supersenses(word) -> set[str]:
    if len(word) < 4:
        return set()
    # Lemmatize the word
    word = lemmatizer.lemmatize(word)
    synsets = wordnet.synsets(word)
    return set(synset.lexname() for synset in synsets if synset is not None)

SUPERSENSE_CLASSES = get_supersense_classes()
SUPERSENSE_TO_ID = {
    supersense: idx for idx, supersense in enumerate(SUPERSENSE_CLASSES)
}
NUM_SUPERSENSE_CLASSES = len(SUPERSENSE_CLASSES)

def encode_supersenses(tokens) -> list[list[int]]:
    encoded_list = []
    for word in tokens:
        s_ids = {SUPERSENSE_TO_ID[s] for s in get_word_supersenses(word) if s in SUPERSENSE_TO_ID} # Ensure s is valid
        if not s_ids:
            encoded_list.append([-100] * NUM_SUPERSENSE_CLASSES)
        else:
            encoded_word = [0] * NUM_SUPERSENSE_CLASSES
            for s_id in s_ids:
                encoded_word[s_id] = 1
            encoded_list.append(encoded_word)
    return encoded_list
