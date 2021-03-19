from dataclasses import dataclass

@dataclass
class Words:
    ...

@dataclass
class Entities:
    ...

@dataclass
class Example:
    """
    Data to be forward passed to daLUKE
    """
    words: Words
    entities: Entities

