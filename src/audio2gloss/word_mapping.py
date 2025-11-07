from typing import Optional


def _create_word_mapping(valid_glosses) -> dict:
    """Create mapping from English words/lemmas to valid glosses"""
    mapping = {}

    # Direct mappings
    word_mappings = {
        # Numbers
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        # Common words
        "i": "ME",
        "me": "ME",
        "my": "ME",
        "myself": "ME",
        "stomach": "ABDOMEN",
        "belly": "ABDOMEN",
        "tummy": "ABDOMEN",
        "crash": "ACCIDENT",
        "collision": "ACCIDENT",
        "later": "AFTER",
        "afterwards": "AFTER",
        "once more": "AGAIN",
        "repeat": "AGAIN",
        "liquor": "ALCOHOL",
        "beer": "ALCOHOL",
        "wine": "ALCOHOL",
        "too": "ALSO",
        "as well": "ALSO",
        "forever": "ALWAYS",
        "constantly": "ALWAYS",
        "mad": "ANGRY",
        "furious": "ANGRY",
        "upset": "ANGRY",
        "creature": "ANIMAL",
        "beast": "ANIMAL",
        "reach": "ARRIVE",
        "get": "ARRIVE",
        "come": "ARRIVE",
        "start": "BEGIN",
        "commence": "BEGIN",
        "deaf": "BECOME_DEAF",
        "deafened": "BECOME_DEAF",
        "earlier": "BEFORE",
        "prior": "BEFORE",
        "kid": "CHILD",
        "baby": "CHILD",
        "toddler": "CHILD",
        "kids": "CHILDREN",
        "babies": "CHILDREN",
        "vehicle": "CAR",
        "automobile": "CAR",
        "smart": "CLEVER",
        "intelligent": "CLEVER",
        "bright": "CLEVER",
        "hard": "DIFFICULT",
        "challenging": "DIFFICULT",
        "tough": "DIFFICULT",
        "physician": "DOCTOR",
        "doc": "DOCTOR",
        "medic": "DOCTOR",
        "beverage": "DRINK",
        "sip": "DRINK",
        "hot": "FEVERISH",
        "fever": "FEVERISH",
        "pal": "FRIEND",
        "buddy": "FRIEND",
        "mate": "FRIEND",
        "lady": "GIRL",
        "lass": "GIRL",
        "provide": "GIVE",
        "donate": "GIVE",
        "offer": "GIVE",
        "grandpa": "GRANDFATHER",
        "granddad": "GRANDFATHER",
        "grandma": "GRANDMOTHER",
        "granny": "GRANDMOTHER",
        "nana": "GRANDMOTHER",
        "possess": "HAVE",
        "own": "HAVE",
        "well": "HEALTHY",
        "fit": "HEALTHY",
        "fine": "HEALTHY",
        "tall": "HIGH",
        "elevated": "HIGH",
        "short": "LOW",
        "small": "LOW",
        "guy": "MAN",
        "male": "MAN",
        "gentleman": "MAN",
        "encounter": "MEET",
        "greet": "MEET",
        "motorcycle": "MOTORBIKE",
        "bike": "MOTORBIKE",
        "rest": "OTHERS",
        "remaining": "OTHERS",
        "mom": "PARENTS",
        "dad": "PARENTS",
        "mother": "PARENTS",
        "father": "PARENTS",
        "guard": "PROTECT",
        "defend": "PROTECT",
        "shield": "PROTECT",
        "recall": "REMEMBER",
        "recollect": "REMEMBER",
        "cycle": "RIDE_BICYCLE",
        "bike": "RIDE_BICYCLE",
        "pedal": "RIDE_BICYCLE",
        "jog": "RUN",
        "sprint": "RUN",
        "dash": "RUN",
        "watch": "SEE",
        "look": "SEE",
        "view": "SEE",
        "observe": "SEE",
        "brother": "SIBLING",
        "sister": "SIBLING",
        "ill": "SICK",
        "unwell": "SICK",
        "ailing": "SICK",
        "talk": "SPEAK",
        "say": "SPEAK",
        "tell": "SPEAK",
        "communicate": "SPEAK",
        "dumb": "STUPID",
        "foolish": "STUPID",
        "idiotic": "STUPID",
        "thanks": "THANK_YOU",
        "thankyou": "THANK_YOU",
        "toss": "THROW",
        "hurl": "THROW",
        "fling": "THROW",
        "period": "TIME",
        "moment": "TIME",
        "abuse": "VIOLENCE",
        "fight": "VIOLENCE",
        "attack": "VIOLENCE",
        "puke": "VOMIT",
        "throw up": "VOMIT",
        "lady": "WOMAN",
        "female": "WOMAN",
        "hate": "I_DISLIKE_YOU",
        "dislike": "I_DISLIKE_YOU",
        "dunno": "I_DONT_KNOW",
        "don't know": "I_DONT_KNOW",
    }

    # Add all direct mappings
    for word, gloss in word_mappings.items():
        mapping[word.lower()] = gloss

    # Add valid glosses mapping to themselves
    for gloss in valid_glosses:
        mapping[gloss.lower()] = gloss

    return mapping


def _map_to_valid_gloss(word: str, word_to_gloss_map, valid_glosses) -> Optional[str]:
    """Map a word to a valid gloss, or return None if no mapping exists"""
    word_lower = word.lower()

    # Check direct mapping
    if word_lower in word_to_gloss_map:
        return word_to_gloss_map[word_lower]

    # Check if word.upper() is already a valid gloss
    word_upper = word.upper()
    if word_upper in valid_glosses:
        return word_upper

    # Try removing common suffixes and check again
    suffixes = ["s", "ed", "ing", "er", "est", "ly"]
    for suffix in suffixes:
        if word_lower.endswith(suffix):
            stem = word_lower[: -len(suffix)]
            if stem in word_to_gloss_map:
                return word_to_gloss_map[stem]
            if stem.upper() in valid_glosses:
                return stem.upper()

    return None
