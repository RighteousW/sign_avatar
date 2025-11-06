import csv
import random
from typing import List, Tuple


class SyntheticGlossTextGenerator:
    """
    Generates synthetic gloss-text pairs using NSL gesture recognizer class names.
    """

    def __init__(self):
        # Your actual gesture recognizer vocabulary
        self.VOCABULARY = {
            # Numbers
            "0": ["zero"],
            "1": ["one"],
            "2": ["two"],
            "3": ["three"],
            "4": ["four"],
            "5": ["five"],
            "6": ["six"],
            "7": ["seven"],
            "8": ["eight"],
            "9": ["nine"],
            # Alphabet (can be used for fingerspelling or names)
            "A": ["a"],
            "B": ["b"],
            "C": ["c"],
            "D": ["d"],
            "E": ["e"],
            "F": ["f"],
            "G": ["g"],
            "H": ["h"],
            "I": ["i"],
            "J": ["j"],
            "K": ["k"],
            "L": ["l"],
            "M": ["m"],
            "N": ["n"],
            "O": ["o"],
            "P": ["p"],
            "Q": ["q"],
            "R": ["r"],
            "S": ["s"],
            "T": ["t"],
            "U": ["u"],
            "V": ["v"],
            "W": ["w"],
            "X": ["x"],
            "Y": ["y"],
            "Z": ["z"],
            # Medical/Body terms
            "ABDOMEN": ["abdomen", "stomach", "belly"],
            "BLOOD": ["blood"],
            "KIDNEY": ["kidney", "kidneys"],
            "FEVERISH": ["feverish", "fever", "hot"],
            "SICK": ["sick", "ill"],
            "HEALTHY": ["healthy", "well"],
            "VOMIT": ["vomit", "throw up"],
            "BROKEN": ["broken", "fractured"],
            "BREAST_FEED": ["breastfeed", "nurse"],
            # People
            "MAN": ["man"],
            "WOMAN": ["woman"],
            "BOY": ["boy"],
            "GIRL": ["girl"],
            "CHILD": ["child"],
            "CHILDREN": ["children", "kids"],
            "PERSON": ["person"],
            "FRIEND": ["friend"],
            "DOCTOR": ["doctor"],
            "BOSS": ["boss"],
            "PARENTS": ["parents"],
            "GRANDFATHER": ["grandfather", "grandpa"],
            "GRANDMOTHER": ["grandmother", "grandma"],
            "SIBLING": ["sibling", "brother", "sister"],
            # Pronouns
            "ME": ["me", "I"],
            "YOU": ["you"],
            # Verbs/Actions
            "ARRIVE": ["arrive", "reach", "get there"],
            "BEGIN": ["begin", "start"],
            "BECOME_DEAF": ["become deaf", "lost hearing"],
            "BORN": ["born"],
            "COME": ["come"],
            "DRINK": ["drink"],
            "GIVE": ["give"],
            "HAVE": ["have"],
            "MEET": ["meet"],
            "PROTECT": ["protect", "guard"],
            "REMEMBER": ["remember", "recall"],
            "RIDE_BICYCLE": ["ride bicycle", "cycle", "bike"],
            "RUN": ["run"],
            "SEE": ["see", "watch", "look"],
            "SPEAK": ["speak", "talk", "say"],
            "THROW": ["throw", "toss"],
            # Adjectives/States
            "ANGRY": ["angry", "mad"],
            "BORING": ["boring", "dull"],
            "CLEAN": ["clean"],
            "CLEVER": ["clever", "smart", "intelligent"],
            "DIFFICULT": ["difficult", "hard"],
            "HIGH": ["high", "tall"],
            "LOW": ["low", "short"],
            "STUPID": ["stupid", "dumb"],
            # Time expressions
            "TIME": ["time"],
            "AFTER": ["after", "later"],
            "BEFORE": ["before", "earlier"],
            "AGAIN": ["again"],
            "AGO": ["ago"],
            "ALWAYS": ["always"],
            # Objects
            "BOOK": ["book"],
            "BOX": ["box"],
            "CAR": ["car"],
            "MOTORBIKE": ["motorbike", "motorcycle"],
            "BANANA": ["banana"],
            # Animals
            "ANIMAL": ["animal"],
            "BIRD": ["bird"],
            # Places
            "CHURCH": ["church"],
            # Connectors
            "BUT": ["but"],
            "ALSO": ["also", "too"],
            "WITH": ["with"],
            # Questions
            "WHAT": ["what"],
            # Other
            "ACCIDENT": ["accident", "crash"],
            "ALCOHOL": ["alcohol", "drink", "liquor"],
            "OTHERS": ["others", "rest"],
            "VIOLENCE": ["violence", "fight", "abuse"],
            # Complex actions/descriptions
            "PERSON_STRADDLES_BRANCH": ["person straddles branch", "sit on branch"],
            "PERSON_SWINGS_FROM_BRANCH": [
                "person swings from branch",
                "swing from tree",
            ],
            # Phrases
            "THANK_YOU": ["thank you", "thanks"],
            "I_DISLIKE_YOU": ["I dislike you", "I hate you"],
            "I_DONT_KNOW": ["I don't know", "I do not know"],
            "YOU_DONT_KNOW_ANYTHING": ["you don't know anything", "you know nothing"],
        }

        # Create reverse mapping for text to gloss
        self.text_to_gloss = {}
        for gloss, texts in self.VOCABULARY.items():
            for text in texts:
                self.text_to_gloss[text.lower()] = gloss

        # Sentence templates following NSL SOV (Subject-Object-Verb) order
        self.templates = self._create_templates()

    def _create_templates(self) -> List[dict]:
        """Create sentence templates with NSL gloss order (SOV)"""
        return [
            # Simple Subject-Verb
            {
                "pattern": "{subject} {verb}",
                "text_pattern": "{subject} {verb}",
                "subjects": [
                    "ME",
                    "YOU",
                    "MAN",
                    "WOMAN",
                    "BOY",
                    "GIRL",
                    "CHILD",
                    "DOCTOR",
                    "FRIEND",
                ],
                "verbs": ["RUN", "COME", "SPEAK", "ARRIVE"],
            },
            # Subject-Verb-Adjective
            {
                "pattern": "{subject} {adjective}",
                "text_pattern": "{subject} is {adjective}",
                "subjects": [
                    "ME",
                    "YOU",
                    "MAN",
                    "WOMAN",
                    "BOY",
                    "GIRL",
                    "CHILD",
                    "FRIEND",
                    "DOCTOR",
                ],
                "adjectives": [
                    "SICK",
                    "HEALTHY",
                    "ANGRY",
                    "CLEVER",
                    "STUPID",
                    "FEVERISH",
                    "HAPPY",
                ],
            },
            # Subject-Object-Verb (SOV - core NSL structure)
            {
                "pattern": "{subject} {object} {verb}",
                "text_pattern": "{subject} {verb} {object}",
                "subjects": [
                    "ME",
                    "YOU",
                    "MAN",
                    "WOMAN",
                    "DOCTOR",
                    "FRIEND",
                    "BOY",
                    "GIRL",
                ],
                "objects": ["BOOK", "BOX", "CAR", "BANANA", "MOTORBIKE"],
                "verbs": ["SEE", "HAVE", "GIVE", "THROW"],
            },
            # Subject-Person-Verb
            {
                "pattern": "{subject} {person} {verb}",
                "text_pattern": "{subject} {verb} {person}",
                "subjects": ["ME", "YOU"],
                "persons": [
                    "DOCTOR",
                    "FRIEND",
                    "GRANDFATHER",
                    "GRANDMOTHER",
                    "BOSS",
                    "SIBLING",
                ],
                "verbs": ["SEE", "MEET", "REMEMBER"],
            },
            # Time-Subject-Verb
            {
                "pattern": "{time} {subject} {verb}",
                "text_pattern": "{subject} {verb} {time}",
                "times": ["ALWAYS", "AGAIN"],
                "subjects": ["ME", "YOU", "FRIEND", "CHILD"],
                "verbs": ["COME", "ARRIVE", "SPEAK", "RUN"],
            },
            # Subject-Action (activities)
            {
                "pattern": "{subject} {action}",
                "text_pattern": "{subject} {action}",
                "subjects": ["ME", "YOU", "BOY", "GIRL", "CHILD", "FRIEND"],
                "actions": ["RIDE_BICYCLE", "DRINK", "VOMIT", "BREAST_FEED"],
            },
            # Possessive constructions
            {
                "pattern": "{person} {body_part}",
                "text_pattern": "{person} has {body_part} pain",
                "persons": ["ME", "YOU", "MAN", "WOMAN", "CHILD"],
                "body_parts": ["ABDOMEN", "KIDNEY"],
            },
            # Comparative/State
            {
                "pattern": "{object} {state}",
                "text_pattern": "{object} is {state}",
                "objects": ["CAR", "BOX", "BOOK", "CHURCH"],
                "states": ["CLEAN", "BROKEN", "HIGH", "LOW"],
            },
            # With preposition
            {
                "pattern": "{subject} {person} WITH {verb}",
                "text_pattern": "{subject} {verb} with {person}",
                "subjects": ["ME", "YOU"],
                "persons": ["FRIEND", "DOCTOR", "GRANDFATHER", "GRANDMOTHER"],
                "verbs": ["COME", "SPEAK", "ARRIVE"],
            },
            # Complex medical
            {
                "pattern": "{person} {symptom} {verb}",
                "text_pattern": "{person} {verb} {symptom}",
                "persons": ["CHILD", "MAN", "WOMAN", "BOY", "GIRL"],
                "symptoms": ["SICK", "FEVERISH", "BLOOD"],
                "verbs": ["HAVE", "SEE"],
            },
            # Time expressions
            {
                "pattern": "{time} {subject} {person} {verb}",
                "text_pattern": "{subject} {verb} {person} {time}",
                "times": ["BEFORE", "AFTER", "AGO"],
                "subjects": ["ME", "YOU"],
                "persons": ["DOCTOR", "FRIEND"],
                "verbs": ["SEE", "MEET"],
            },
            # Numbers with objects
            {
                "pattern": "{number} {object}",
                "text_pattern": "{number} {object}",
                "numbers": ["1", "2", "3", "4", "5"],
                "objects": ["BOOK", "BOX", "CAR", "CHILD", "CHILDREN", "FRIEND"],
            },
            # Emotional states
            {
                "pattern": "{subject} {emotion}",
                "text_pattern": "{subject} is {emotion}",
                "subjects": ["ME", "YOU", "MAN", "WOMAN", "CHILD", "FRIEND"],
                "emotions": ["ANGRY", "BORING"],
            },
            # Actions with objects and location
            {
                "pattern": "{subject} {object} {location} {verb}",
                "text_pattern": "{subject} {verb} {object} at {location}",
                "subjects": ["ME", "YOU", "FRIEND"],
                "objects": ["BOOK", "BOX"],
                "locations": ["CHURCH"],
                "verbs": ["SEE", "GIVE"],
            },
            # Negative/Difficult
            {
                "pattern": "{action} DIFFICULT",
                "text_pattern": "{action} is difficult",
                "actions": ["SPEAK", "RUN", "RIDE_BICYCLE", "REMEMBER"],
            },
            # Animals
            {
                "pattern": "{subject} {animal} {verb}",
                "text_pattern": "{subject} {verb} {animal}",
                "subjects": ["ME", "YOU", "CHILD"],
                "animals": ["BIRD", "ANIMAL"],
                "verbs": ["SEE", "PROTECT"],
            },
            # Questions
            {
                "pattern": "WHAT {subject} {verb}",
                "text_pattern": "what does {subject} {verb}",
                "subjects": ["YOU", "MAN", "WOMAN", "CHILD"],
                "verbs": ["HAVE", "DRINK", "SEE"],
            },
            # Fixed phrases
            {"pattern": "THANK_YOU", "text_pattern": "thank you"},
            {"pattern": "I_DONT_KNOW", "text_pattern": "I don't know"},
            {"pattern": "I_DISLIKE_YOU", "text_pattern": "I dislike you"},
            {
                "pattern": "YOU_DONT_KNOW_ANYTHING",
                "text_pattern": "you don't know anything",
            },
            # Events
            {
                "pattern": "{subject} ACCIDENT {verb}",
                "text_pattern": "{subject} had an accident while {verb}",
                "subjects": ["ME", "YOU", "MAN", "WOMAN", "CHILD"],
                "verbs": ["RUN", "RIDE_BICYCLE"],
            },
            # Compound sentences with BUT
            {
                "pattern": "{subject} {adj1} BUT {adj2}",
                "text_pattern": "{subject} is {adj1} but {adj2}",
                "subjects": ["MAN", "WOMAN", "CHILD", "FRIEND"],
                "adj1": ["CLEVER", "HEALTHY", "ANGRY"],
                "adj2": ["SICK", "BORING", "STUPID"],
            },
            # Alcohol/substances
            {
                "pattern": "{subject} ALCOHOL DRINK",
                "text_pattern": "{subject} drinks alcohol",
                "subjects": ["MAN", "WOMAN", "FRIEND", "BOSS"],
            },
            # Counting sequences
            {
                "pattern": "0 1 2 3 4 5 6 7 8 9",
                "text_pattern": "zero one two three four five six seven eight nine",
            },
            {
                "pattern": "1 2 3 4 5 6 7 8 9 0",
                "text_pattern": "one two three four five six seven eight nine zero",
            },
            {"pattern": "1 2 3 4 5", "text_pattern": "one two three four five"},
            {"pattern": "5 4 3 2 1", "text_pattern": "five four three two one"},
            {"pattern": "0 1 2 3 4", "text_pattern": "zero one two three four"},
            {"pattern": "5 6 7 8 9", "text_pattern": "five six seven eight nine"},
            {"pattern": "1 2 3", "text_pattern": "one two three"},
            {"pattern": "3 2 1", "text_pattern": "three two one"},
            {"pattern": "6 7 8 9", "text_pattern": "six seven eight nine"},
            {"pattern": "2 4 6 8", "text_pattern": "two four six eight"},
            {"pattern": "1 3 5 7 9", "text_pattern": "one three five seven nine"},
            # Complex actions
            {
                "pattern": "PERSON_STRADDLES_BRANCH",
                "text_pattern": "person straddles branch",
            },
            {
                "pattern": "PERSON_SWINGS_FROM_BRANCH",
                "text_pattern": "person swings from branch",
            },
            # Born/Background
            {
                "pattern": "{subject} {time} BORN",
                "text_pattern": "{subject} was born {time}",
                "subjects": ["ME", "YOU", "CHILD"],
                "times": ["AGO"],
            },
            # Become deaf
            {
                "pattern": "{subject} BECOME_DEAF",
                "text_pattern": "{subject} became deaf",
                "subjects": ["ME", "YOU", "CHILD", "MAN", "WOMAN"],
            },
            # Begin activities
            {
                "pattern": "{subject} {action} BEGIN",
                "text_pattern": "{subject} begins to {action}",
                "subjects": ["ME", "YOU", "CHILD", "DOCTOR"],
                "actions": ["SPEAK", "RUN", "DRINK", "RIDE_BICYCLE"],
            },
            # Violence
            {
                "pattern": "{subject} VIOLENCE SEE",
                "text_pattern": "{subject} saw violence",
                "subjects": ["ME", "YOU", "CHILD", "WOMAN"],
            },
            # Also/Additional
            {
                "pattern": "{subject} {verb} ALSO",
                "text_pattern": "{subject} also {verb}",
                "subjects": ["ME", "YOU", "FRIEND"],
                "verbs": ["COME", "ARRIVE", "SPEAK", "DRINK"],
            },
        ]

    def _fill_template(self, template: dict) -> Tuple[str, str]:
        """Fill a template with random valid words"""
        gloss_parts = []
        text_parts = []

        pattern = template["pattern"]
        text_pattern = template["text_pattern"]

        # Extract placeholders
        import re

        placeholders = re.findall(r"\{(\w+)\}", pattern)

        # Fill each placeholder
        replacements = {}
        for placeholder in placeholders:
            plural = placeholder + "s"
            if plural in template:
                options = template[plural]
                choice = random.choice(options)
                replacements[placeholder] = choice

        # Replace in patterns
        gloss = pattern
        text = text_pattern

        for key, value in replacements.items():
            gloss = gloss.replace(f"{{{key}}}", value)
            # For text, get the English word(s)
            text_value = random.choice(self.VOCABULARY.get(value, [value.lower()]))
            text = text.replace(f"{{{key}}}", text_value)

        # Remove extra spaces and convert gloss to proper format
        gloss = " ".join(gloss.split())
        text = " ".join(text.split())

        return gloss, text

    def generate_pairs(self, num_pairs: int = 10000) -> List[Tuple[str, str]]:
        """Generate specified number of gloss-text pairs"""
        pairs = []
        seen = set()

        attempts = 0
        max_attempts = num_pairs * 10

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1

            template = random.choice(self.templates)

            try:
                gloss, text = self._fill_template(template)

                # Check word count (3-20 words)
                word_count = len(text.split())
                if word_count < 3 or word_count > 20:
                    continue

                # Avoid duplicates
                pair_key = (gloss, text)
                if pair_key in seen:
                    continue

                seen.add(pair_key)
                pairs.append((gloss, text))

            except Exception as e:
                continue

        return pairs

    def save_to_csv(self, pairs: List[Tuple[str, str]], output_file: str):
        """Save pairs to CSV file"""
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["gloss", "text"])
            for gloss, text in pairs:
                writer.writerow([gloss, text])

        print(f"Saved {len(pairs)} pairs to {output_file}")


def main():
    generator = SyntheticGlossTextGenerator()

    # Generate maximum diverse pairs
    MAX_DATASET_SIZE = 100_000
    OUTPUT_FILE = "data/dataset/synthetic/synthetic_NSL_gesture_vocab.csv"

    print(f"Generating up to {MAX_DATASET_SIZE:,} gloss-text pairs...")
    pairs = generator.generate_pairs(num_pairs=MAX_DATASET_SIZE)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Valid samples generated: {len(pairs):,}")
    print(f"\nSample pairs:")
    for i, (gloss, text) in enumerate(pairs[:10], 1):
        print(f"{i}. Gloss: {gloss}")
        print(f"   Text:  {text}\n")

    generator.save_to_csv(pairs, OUTPUT_FILE)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
