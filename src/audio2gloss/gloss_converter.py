from typing import List, Optional, Tuple

from ..audio2gloss.word_mapping import _map_to_valid_gloss


def text_to_glosses(
    text: str, nlp, debug, valid_glosses, word_to_gloss_map
) -> List[List[str]]:
    """Convert text to NSL glosses (only valid vocabulary)."""
    if not nlp:
        raise RuntimeError("spaCy model not loaded. Call load_model() first.")

    doc = nlp(text)
    all_clause_glosses = []

    for sent in doc.sents:
        clause_glosses, skipped = _sentence_to_glosses(sent, debug, valid_glosses, word_to_gloss_map)
        if clause_glosses:
            all_clause_glosses.append(clause_glosses)
        if debug and skipped:
            print(f"Skipped words in sentence: {', '.join(skipped)}")

    return all_clause_glosses

def _sentence_to_glosses(
    sent,
    debug,
    valid_glosses,
    word_to_gloss_map,
) -> Tuple[List[str], List[str]]:
    """
    Convert a sentence to NSL glosses with proper SOV ordering.
    Returns: (glosses, skipped_words)
    """
    glosses = []
    skipped_words = []
    tokens = list(sent)
    added_tokens = set()
    added_glosses = set()

    # Step 1: Reconstruct compounds
    compounds_map = _reconstruct_compounds(tokens)

    if debug:
        print("\n" + "=" * 70)
        print("ENGLISH SENTENCE")
        print("=" * 70)
        print(f"Input: {sent.text}")
        print()

    # Helper function to add gloss safely
    def add_gloss_safe(token):
        """Add gloss only if not already added AND it's in valid vocabulary"""
        if token in added_tokens:
            return False

        gloss = _get_gloss_token(valid_glosses, word_to_gloss_map, token, compounds_map)

        # Handle multi-digit numbers
        if gloss and gloss.startswith("__DIGITS__"):
            digit_string = gloss.replace("__DIGITS__", "")
            for digit in digit_string:
                if digit in valid_glosses and digit not in added_glosses:
                    glosses.append(digit)
                    added_glosses.add(digit)
                    if debug:
                        print(f"  ✓ Added digit: {digit}")
            added_tokens.add(token)
            return True

        if gloss and gloss in valid_glosses and gloss not in added_glosses:
            glosses.append(gloss)
            added_tokens.add(token)
            added_glosses.add(gloss)
            if debug:
                print(f"  ✓ Added: {token.text} -> {gloss}")
            return True
        elif gloss is None and _is_content_word(token):
            skipped_words.append(token.text)
            if debug:
                print(f"  ✗ Skipped: {token.text} (no valid mapping)")

        added_tokens.add(token)
        return False

    # Step 2: Time expressions
    time_markers = ["after", "before", "again", "ago", "always"]
    for token in tokens:
        if token.text.lower() in time_markers and _is_content_word(token):
            add_gloss_safe(token)

    # Step 3: Identify sentence components
    subjects = []
    direct_objects = []
    verbs = []
    modals = []

    for token in tokens:
        if token in added_tokens:
            continue

        if token.dep_ in ["nsubj", "nsubjpass"]:
            subjects.append(token)
        elif token.dep_ in ["dobj", "attr"]:
            direct_objects.append(token)
        elif token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
            if token.lemma_ not in ["be", "have", "do"]:
                verbs.append(token)
        elif token.tag_ == "MD":
            modals.append(token)

    if debug:
        print("\n" + "=" * 70)
        print("SOV REORDERING (Vocabulary Constrained)")
        print("=" * 70)

    # SUBJECTS
    for subj in subjects:
        phrase_tokens = _get_noun_phrase_tokens(subj)
        for token in phrase_tokens:
            add_gloss_safe(token)

    # DIRECT OBJECTS
    for obj in direct_objects:
        phrase_tokens = _get_noun_phrase_tokens(obj)
        for token in phrase_tokens:
            add_gloss_safe(token)

    # PREPOSITIONAL PHRASES
    for token in tokens:
        if token.pos_ == "ADP" and _is_content_word(token):
            if token not in added_tokens:
                add_gloss_safe(token)
                # Add object of preposition
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj_phrase = _get_noun_phrase_tokens(
                            child
                        )
                        for obj_token in obj_phrase:
                            add_gloss_safe(obj_token)

    # REMAINING CONTENT WORDS
    for token in tokens:
        if _is_content_word(token) and token not in added_tokens:
            add_gloss_safe(token)

    # VERBS
    for verb in verbs:
        add_gloss_safe(verb)

    if debug:
        print()
        print("=" * 70)
        print("NSL GLOSS SEQUENCE (Valid Vocabulary Only)")
        print("=" * 70)
        print(f"Output: {' '.join(glosses)}")
        if skipped_words:
            print(f"Skipped: {', '.join(skipped_words)}")
        print("=" * 70)
        print()

    return glosses, skipped_words

def _get_gloss_token(
    valid_glosses, word_to_gloss_map, token, compounds_map
) -> Optional[str]:
    """Convert a single token to its gloss representation (CONSTRAINED)"""
    if token.i in compounds_map:
        compound = compounds_map[token.i]
        if compound is None:
            return None
        # Check if compound is valid
        if compound in valid_glosses:
            return compound
        # Try to map it
        mapped = _map_to_valid_gloss(compound, word_to_gloss_map, valid_glosses)
        return mapped

    # Check if token is a number that needs separation
    if token.pos_ == "NUM" and token.text.isdigit() and len(token.text) > 1:
        # Return special marker for multi-digit numbers
        return f"__DIGITS__{token.text}"

    # Get base form
    if token.tag_ in ["PRP$", "WP$"]:
        word = token.text
    elif "'s" in token.text:
        word = token.text.replace("'s", "")
    elif token.pos_ == "NOUN" and token.tag_ in ["NNS", "NNPS"]:
        word = token.text
    else:
        word = token.lemma_

    # Map to valid gloss
    mapped_gloss = _map_to_valid_gloss(word, word_to_gloss_map, valid_glosses, valid_glosses)
    return mapped_gloss

def _is_content_word(token) -> bool:
    """Check if token should be included in glosses"""
    if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"]:
        if token.pos_ == "VERB" and token.lemma_ in ["be", "have", "do"]:
            return False
        if token.pos_ == "AUX" and token.lemma_ == "be":
            return False
        if token.text.lower() == "it" and token.dep_ == "expl":
            return False
        return True

    if token.tag_ == "MD":
        return True

    if token.pos_ == "ADP":
        important_preps = {"with", "after", "before"}
        return token.text.lower() in important_preps

    return False

def _get_noun_phrase_tokens(head_token):
    """
    Get all tokens that are part of a noun phrase.
    Returns list of tokens in correct order, handling compounds.
    """
    phrase_token_set = set()
    phrase_tokens = []

    # Collect modifiers
    for child in head_token.children:
        if child.dep_ == "poss" and child.i < head_token.i:
            if child not in phrase_token_set:
                phrase_token_set.add(child)
                phrase_tokens.append(child)

        elif child.dep_ == "compound" and child.i < head_token.i:
            if child not in phrase_token_set:
                phrase_token_set.add(child)
                phrase_tokens.append(child)

        elif child.dep_ == "amod":
            if child not in phrase_token_set:
                phrase_token_set.add(child)
                phrase_tokens.append(child)

        elif child.dep_ == "nummod" and child.i < head_token.i:
            if child not in phrase_token_set:
                phrase_token_set.add(child)
                phrase_tokens.append(child)

    # Add the head noun itself
    if head_token not in phrase_token_set:
        phrase_token_set.add(head_token)
        phrase_tokens.append(head_token)

    # Sort by position
    phrase_tokens.sort(key=lambda t: t.i)

    return phrase_tokens

def _reconstruct_compounds(tokens):
    """
    Reconstruct hyphenated compounds that spaCy splits.
    Returns a dict mapping token indices to their compound forms.
    """
    compounds = {}
    i = 0

    while i < len(tokens):
        if i + 2 < len(tokens):
            if tokens[i + 1].text == "-" or tokens[i + 1].tag_ == "HYPH":
                parts = [tokens[i].text.upper()]
                j = i + 2
                parts.append(tokens[j].text.upper())

                while j + 2 < len(tokens) and tokens[j + 1].text == "-":
                    parts.append(tokens[j + 2].text.upper())
                    j += 2

                compound = "_".join(parts)  # Use underscore for compounds
                compounds[j] = compound

                for k in range(i, j):
                    compounds[k] = None

                i = j + 1
                continue
        i += 1

    return compounds

def _separate_digits(text: str) -> List[str]:
    """Separate digits in a string into individual digit glosses"""
    digits = []
    for char in text:
        if char.isdigit():
            digits.append(char)
    return digits
