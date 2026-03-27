import re

import ast
import json
import logging
from abc import ABC, abstractmethod
from .en import EnglishTextNormalizer
from .basic import BasicTextNormalizer
from .cn_tn import TextNorm

logger = logging.getLogger(__name__)

class Process(ABC):
    @abstractmethod
    def __call__(self, answer: str) -> str:
        raise NotImplementedError()

class TextNormalization(Process):

    def __init__(self, lang: str = ""):
        if lang == "en":
            self.normalizer = EnglishTextNormalizer()
        elif lang == "zh":
            self.normalizer = TextNorm(
                to_banjiao=False,
                to_upper=False,
                to_lower=False,
                remove_fillers=False,
                remove_erhua=False,
                check_chars=False,
                remove_space=False,
                cc_mode="",
            )
        else:
            self.normalizer = BasicTextNormalizer()

    def __call__(self, answer: str) -> str:
        return self.normalizer(answer)