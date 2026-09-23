import re
import unicodedata
from typing import List

# Standard curated Bangla stop words that carry minimal semantic meaning for proverb intent
BANGLA_STOPWORDS = {
    "এবং", "ও", "কিন্তু", "বা", "যদি", "তবে", "তা", "তো", "যে", "সে", "তার", "তাদের",
    "তিনি", "তারাই", "তার", "তাকে", "তাহার", "ইহাদের", "ইহা", "এই", "ওই", "একটি", "একটা",
    "কোন", "কোনো", "কিছু", "সব", "সকল", "হতে", "থেকে", "চেয়ে", "দ্বারা", "দিয়ে", "কর্তৃক",
    "জন্য", "কারণে", "ফলে", "পরে", "পূর্বে", "আগে", "পর্যন্ত", "সাথে", "সঙ্গে", "দ্বারা",
     "করা", "করতে", "করল", "করলেন", "করেছিল", "হয়ে", "হওয়া", "হতে", "হলো", "হয়েছিল",
    "যায়", "গেল", "ছিল", "আছে", "থাকেন", "থাকা", "নি", "আর", "এখন", "তখন",
    "যখন", "যেখানে", "সেখানে", "কখন", "কী", "কেন", "কীভাবে", "কীসের", "বটে", "মতো", "মতন"
}

# Bangla punctuation characters including Dari (|), double Dari, quotes, etc.
BANGLA_PUNCTUATION_PATTERN = re.compile(r'[\।\,\;\:\'\"\?\!\-\–\—\(\)\[\]\{\}\<\>\/\\\|@#\$%\^&\*_\+=~`]')


class BanglaPreprocessor:
    """Bangla Text Normalizer and Tokenizer for Proverb Retrieval."""

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    @staticmethod
    def normalize(text: str) -> str:
        """Apply Unicode normalization and whitespace cleaning."""
        if not text:
            return ""
        # Standardize Unicode composition (NFC)
        normalized = unicodedata.normalize("NFC", text)
        # Remove zero-width characters often found in web-scraped Bangla text
        normalized = normalized.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
        # Remove punctuation
        cleaned = BANGLA_PUNCTUATION_PATTERN.sub(" ", normalized)
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into words."""
        cleaned = self.normalize(text)
        if not cleaned:
            return []
        tokens = cleaned.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in BANGLA_STOPWORDS and len(t) > 1]
        return tokens

    def preprocess_sentence(self, text: str) -> str:
        """Returns normalized space-separated tokens."""
        tokens = self.tokenize(text)
        return " ".join(tokens)


