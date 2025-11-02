#%%
# import modules
from typing import Union
import logging
from typing import Literal
from word2number import w2n
from sentence_transformers import SentenceTransformer
from langdetect import detect
import re 
import unicodedata
from spellchecker import SpellChecker
import contractions
import numpy as np
import spacy 
import string

from .custom_types.class_types import BaseClass
from .custom_types.config_types import BaseConfig
from .custom_types.ctx_types import BaseCtx

class NLProcessor(BaseClass):
    SENTENCE_MODELS = {
        "bert_l6": SentenceTransformer("all-MiniLM-L6-v2"),
        # "bert_l12": SentenceTransformer("all-MiniLM-L12-v2"),
        # "roberta": SentenceTransformer("all-roberta-large-v1"),
        # "distilbert": SentenceTransformer("stsb-distilbert-base"),
    }
        
    def __init__(self,
                 config: Union[BaseConfig, None]=None,
                 ctx: Union[BaseCtx, None]=None,
                 verbose: int=logging.INFO):
        super().__init__(config=config, ctx=ctx, verbose=verbose)
        self.language = "unknown"
        self.spacy_model = None
        
    def preprocess(self, 
                   text: str)->str:
        # Step 0: Language Detection
        self.language=self.detect_language(text)
        
        # Step 1: Text Cleaning
        text = self.remove_noise(text)
        # text = self.check_spelling(text)
        text = self.normalize(text)
        text = self.remove_special_characters(text)
        text = self.remove_punctuation(text)

        return text
    
    def _load_spacy_model(self, lang_code: str):
        """Loads or switches the spaCy model based on language code."""
        model_name = f"{lang_code}_core_web_sm"
        
        try:
            self.spacy_model = spacy.load(model_name)
            self.spacy_model_name = model_name
            self.language = lang_code
            self.logger.info(f'Loaded spaCy model: {model_name}')
        except OSError:
            # Fallback to English if model not installed
            self.logger.warning(f"spaCy model '{model_name}' not found. Falling back to 'en_core_web_sm'.")
            self.spacy_model = spacy.load("en_core_web_sm")
            self.spacy_model_name = "en_core_web_sm"

    def detect_language(self, text: str) -> str:
        self.logger.info('Detecting language ...')
        try:
            detected_lang = detect(text)
            self.logger.info(f'Detected text in {detected_lang}')
            self._load_spacy_model(detected_lang)
            
            return detected_lang
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}. Defaulting to 'en'.")
            self._load_spacy_model("en")
            return "en"
        
    def remove_noise(self, text: str)->str:
        self.logger.debug('Removing noise ...')
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
        # Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
    
        # Remove extra spaces/newlines
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def check_spelling(self, text: str) -> str:
        self.logger.debug("Checking spelling ...")
        spell = SpellChecker()
    
        if not isinstance(text, str) or not text.strip():
            return text
    
        corrected_words = []
        for word in text.split():
            if word.isalpha():
                corrected = spell.correction(word)
                # If SpellChecker returns None, keep original word
                corrected_words.append(corrected if corrected else word)
            else:
                corrected_words.append(word)
    
        return " ".join(corrected_words)
        
    def normalize(self, text: str) -> str:
        self.logger.debug('Normalizing text ...')

        # lower text
        text = text.lower() 

        # handle accents 
        text = self.handle_accents(text)
        
        # handle numbers 
        # text = self.handle_numbers(text, 'normalize') 

        # handle contractions
        text = self.handle_contractions(text)
        
        return text
    
    def handle_accents(self, text: str)->str:
        if self.language not in ['fr', 'es']:
            # remove accents
            normalized = unicodedata.normalize('NFD', text)
            text= ''.join(c for c in normalized if not unicodedata.combining(c))
        return text
    
    def handle_numbers(self, text: str, mode: Literal['extract', 'remove', 'normalize'] = "extract") -> list | str:
        doc = self.spacy_model(text)
        
        if mode == "extract":
            # Return all numeric tokens (NUM or like_num)
            return [token.text for token in doc if token.like_num]
        
        elif mode == "remove":
            # Rebuild sentence without numeric tokens
            return " ".join(token.text for token in doc if not token.like_num)
        
        elif mode == "normalize":
            # Convert number words ("ten") → digits ("10")
            normalized_tokens = []
            for token in doc:
                if token.like_num:
                    try:
                        normalized_tokens.append(str(w2n.word_to_num(token.text)))
                    except Exception:
                        normalized_tokens.append(token.text)
                else:
                    normalized_tokens.append(token.text)
            return " ".join(normalized_tokens)
        
        else:
            raise ValueError("Mode must be 'extract', 'remove', or 'normalize'")
    
    def remove_special_characters(self, text: str) -> str:
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'"-]', '', text)
        return text

    def remove_punctuation(self, text: str) -> str:
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def handle_contractions(self, text: str) -> str:
        text = contractions.fix(text)
        return text

    def get_embeddings(self, 
                       texts, 
                       method: Literal["bert_l6", "bert_l12", "roberta", "distilbert"]="bert_l6"):
        if isinstance(texts, str):
            texts = [texts]

        if method in self.SENTENCE_MODELS:
            return np.array(self.SENTENCE_MODELS[method].encode(texts, show_progress_bar=False))
        else:
            raise ValueError(f"Embedding method '{method}' not recognized.")
