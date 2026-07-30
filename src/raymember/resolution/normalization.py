"""
Normalization pipeline for location strings and entity names.
"""

import re


def normalize_string(text: str) -> str:
    """Basic whitespace stripping and lowercase conversion."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_separators(text: str) -> str:
    """
    Normalizes common separators in location/entity names:
    - living-room -> living room
    - living_room -> living room
    - livingroom -> living room (via known concatenated pattern handling)
    """
    s = normalize_string(text)
    if not s:
        return ""

    # Replace dashes and underscores with spaces
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Handle known concatenated words for common room names
    concatenated_map = {
        "livingroom": "living room",
        "bedroom": "bedroom",  # canonical standard
        "washroom": "washroom",
        "restroom": "restroom",
        "bathroom": "bathroom",
        "sittingroom": "sitting room",
        "diningroom": "dining room",
    }

    words = s.split(" ")
    normalized_words = [concatenated_map.get(w, w) for w in words]
    return " ".join(normalized_words)


def normalize_entity_name(text: str) -> str:
    """
    Normalizes entity names while preserving distinction:
    - 'Black_Backpack' -> 'black backpack'
    - 'black-backpack' -> 'black backpack'
    - 'black backpack' -> 'black backpack'
    - 'blue backpack' -> 'blue backpack' (remains distinct)
    """
    return normalize_separators(text)
