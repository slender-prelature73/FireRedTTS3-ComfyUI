"""FireRedTTS3 text frontend: cleaning, language ID, text normalization, sentence split.

Ported from upstream fireredtts3/utils/text_normalize.py (CosyVoice/VoxCPM derived)
with the LLM-API normalizer removed. FastText language ID (lid.176) is supported
through either the `fasttext` or `fasttext-predict` package.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Callable, List, Optional

try:
    import regex
except ImportError:  # pragma: no cover
    regex = None

logger = logging.getLogger("FireRedTTS3-ComfyUI")

FASTTEXT_FILENAME = "lid.176.ftz"
FASTTEXT_URL = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{FASTTEXT_FILENAME}"

_WETEXT_LANGS = {"Chinese", "English"}

_LOCALE_TO_LANG_TAG = {
    "zh-CN": "Chinese",
    "en-US": "English",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "es-MX": "Spanish",
    "fr-FR": "French",
    "ru-RU": "Russian",
    "ar-SA": "Arabic",
    "tr-TR": "Turkish",
    "id-ID": "Indonesian",
    "pt-BR": "Portuguese",
    "it-IT": "Italian",
    "nl-NL": "Dutch",
    "vi-VN": "Vietnamese",
    "de-DE": "German",
    "uk-UA": "Ukrainian",
    "th-TH": "Thai",
    "pl-PL": "Polish",
    "ro-RO": "Romanian",
    "el-GR": "Greek",
    "cs-CZ": "Czech",
    "fi-FI": "Finnish",
    "hi-IN": "Hindi",
}

# FastText ISO-639 codes -> locales (only locales the model supports).
_FT_LANG_TO_LOCALE = {
    "ar": "ar-SA", "cs": "cs-CZ", "de": "de-DE", "el": "el-GR",
    "en": "en-US", "es": "es-MX", "fi": "fi-FI", "fr": "fr-FR",
    "hi": "hi-IN", "id": "id-ID", "it": "it-IT", "ja": "ja-JP",
    "ko": "ko-KR", "lt": "lt-LT", "nl": "nl-NL", "pl": "pl-PL",
    "pt": "pt-BR", "ro": "ro-RO", "ru": "ru-RU", "th": "th-TH",
    "tr": "tr-TR", "uk": "uk-UA", "vi": "vi-VN", "zh": "zh-CN",
}


# --------------------------------------------------------------------------- #
# Basic cleaning
# --------------------------------------------------------------------------- #
def preprocess_text(sentence: str) -> str:
    if not sentence:
        return ""
    sentence = bytes(sentence, "utf-8").decode("utf-8", "ignore")
    if regex is not None:
        sentence = regex.sub(r"[\p{Cf}--[\u200d]]", "", sentence, flags=regex.V1)
        sentence = regex.sub(r"\p{Co}", "", sentence)
    else:
        sentence = re.sub(r"[​-‏ - ⁠-⁯\ufeff]", "", sentence)
        sentence = re.sub(r"[\ue000-\uf8ff]", "", sentence)
    sentence = sentence.replace("\u00a0", " ")
    sentence = sentence.replace("\ufffd", "")
    sentence = sentence.replace(" ", "\n")
    sentence = sentence.replace(" ", "\n")
    return sentence


def clean_markdown(md_text: str) -> str:
    if not md_text:
        return md_text
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    md_text = re.sub(r"^(\s*)-\s+", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*\*\*([^*]+)\*\*", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*__([^_]+)__", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*\*([^*]+)\*", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*~~([^~]+)~~", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*[*~]\s+", "", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"\n\s*\n", "\n", md_text)
    return md_text.strip()


def remove_emoji(text: str) -> str:
    if regex is None:
        return text
    return regex.compile(r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=regex.UNICODE).sub("", text)


_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
_RE_CJK_SPACE_CJK = re.compile(r"([{}])\s+([{}])".format(_CJK, _CJK))
_RE_CJK_SPACE_LATIN = re.compile(r"([{0}])\s+([a-zA-Z0-9])".format(_CJK))
_RE_LATIN_SPACE_CJK = re.compile(r"([a-zA-Z0-9])\s+([{0}])".format(_CJK))


def clean_tn_spaces(text: str) -> str:
    """Remove spurious spaces TN inserts between CJK chars and CJK/Latin boundaries."""
    if not text:
        return text
    text = _RE_CJK_SPACE_CJK.sub(r"\1\2", text)
    text = _RE_CJK_SPACE_LATIN.sub(r"\1\2", text)
    text = _RE_LATIN_SPACE_CJK.sub(r"\1\2", text)
    return text


_SYMBOL_REDUCTION = {
    "〜": "~", "～": "~",
    "・": "·", "•": "·", "‧": "·",
    "…": "...", "⋯": "...", "〰": "...", "﹏": "...",
}
_SYMBOL_TO_SPACE = re.compile(r"[·•‧│|¦/\\]")
_SYMBOL_TO_COMMA = re.compile(r"[…~&*%$#^:;/\\|]+")
_WETEXT_KEEP = re.compile(
    r"[^"
    r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"0-9A-Za-z"
    r"，。、？！：；…—～·"
    r".,:;!?()\[\]'\"\-_"
    r"\s"
    r"]"
)


def _apply_symbol_reduction(text: str) -> str:
    return "".join(_SYMBOL_REDUCTION.get(ch, ch) for ch in text)


def clean_wetext_output(text: str) -> str:
    if not text:
        return text
    text = _apply_symbol_reduction(text)
    text = _SYMBOL_TO_SPACE.sub(" ", text)
    text = _SYMBOL_TO_COMMA.sub(",", text)
    text = _WETEXT_KEEP.sub("", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"[,，]{2,}", ",", text)
    text = re.sub(r"\s+[,，]", ",", text)
    text = re.sub(r"[,，]+\s*$", "", text)
    return text


def clean_text(text: str) -> str:
    if not text:
        return text
    text = preprocess_text(text)
    text = clean_markdown(text)
    text = remove_emoji(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Sentence splitting (VoxCPM derived)
# --------------------------------------------------------------------------- #
def _is_decimal_dot(text: str, i: int) -> bool:
    return (i > 0 and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit())


def split_paragraph(text: str, tokenize: Optional[Callable[[str], int]] = None, lang: str = "zh",
                    token_max_n: int = 80, token_min_n: int = 60, merge_len: int = 20) -> List[str]:
    def _measure(_text: str) -> int:
        if lang == "zh":
            return len(_text)
        if tokenize is not None:
            n = tokenize(_text)
            return n if isinstance(n, int) else len(n)
        return len(_text)

    pounc = ["。", "？", "！", "；", "、", ".", "?", "!", ";"] if lang == "zh" else [".", "?", "!", ";"]

    st = 0
    utts: List[str] = []
    for i, c in enumerate(text):
        if c in pounc:
            if c == "." and _is_decimal_dot(text, i):
                continue
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                utts.append(utts.pop(-1) + text[i + 1])
                st = i + 2
            else:
                st = i + 1
    trailing = text[st:] if st < len(text) else ""
    if trailing:
        utts.append(trailing)
    elif len(utts) == 0:
        utts.append(text + ("。" if lang == "zh" else ""))

    final_utts: List[str] = []
    cur_utt = ""
    for utt in utts:
        if _measure(cur_utt + utt) > token_max_n and _measure(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if _measure(cur_utt) < merge_len and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)
    return final_utts


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
def lang_tag_to_locale(lang_tag: str) -> Optional[str]:
    if not lang_tag:
        return None
    for locale, tag in _LOCALE_TO_LANG_TAG.items():
        if tag == lang_tag:
            return locale
    if lang_tag == "Cantonese" or lang_tag.startswith("ZH_"):
        return "zh-CN"
    return None


class FastTextLangDetector:
    """Lazy lid.176 wrapper; works with `fasttext` or `fasttext-predict`."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._model = None
        self._kind = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self):
        if self._loaded:
            return self._model
        with self._lock:
            if self._loaded:
                return self._model
            self._loaded = True
            if not self.model_path.is_file():
                return None
            try:
                import fasttext

                self._model = fasttext.load_model(str(self.model_path))
                self._kind = "fasttext"
            except Exception:
                try:
                    from fasttext_predict import FastText

                    self._model = FastText(str(self.model_path))
                    self._kind = "fasttext_predict"
                except Exception as exc:
                    logger.warning("FastText language ID unavailable: %s", exc)
                    self._model = None
        return self._model

    @staticmethod
    def _first_label(result) -> Optional[str]:
        if isinstance(result, str):
            return result
        if isinstance(result, (list, tuple)):
            for item in result:
                label = FastTextLangDetector._first_label(item)
                if label is not None:
                    return label
        return None

    def detect_locale(self, text: str) -> Optional[str]:
        model = self._load()
        if model is None:
            return None
        snippet = text.replace("\n", " ").strip()
        if not snippet:
            return None
        try:
            if self._kind == "fasttext":
                # Low-level predictor avoids the numpy-2.0 incompatibility in predict().
                with self._lock:
                    preds = model.f.predict(snippet, 1, 0.0, "strict")
                label = self._first_label(preds)
            else:
                label = self._first_label(model.predict(snippet))
        except Exception:
            return None
        if not label:
            return None
        iso = label.replace("__label__", "").strip()
        return _FT_LANG_TO_LOCALE.get(iso)


def detect_language(text: str, fasttext_detector: Optional[Callable[[str], Optional[str]]] = None) -> str:
    """Detect language as a FireRedTTS3 lang tag; falls back to zh/ja/en heuristics."""
    if fasttext_detector is not None:
        try:
            locale = fasttext_detector(text)
            if locale:
                tag = _LOCALE_TO_LANG_TAG.get(locale)
                if tag:
                    return tag
        except Exception:
            pass
    if re.search(r"[一-鿿]", text):
        return "Chinese"
    if re.search(r"[぀-ヿㇰ-ㇿ]", text):
        return "Japanese"
    return "English"


def build_wetext_normalizer() -> Optional[Callable[[str], str]]:
    """Local wetext TN (zh/en). Returns a normalize callable or None."""
    try:
        from wetext import Normalizer

        zh_tn_model = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        en_tn_model = Normalizer(lang="en", operator="tn")
        chinese_char_pattern = re.compile(r"[一-鿿]+")

        def _normalize(text: str) -> str:
            if not text:
                return text
            if chinese_char_pattern.search(text):
                out = zh_tn_model.normalize(text)
            else:
                out = en_tn_model.normalize(text)
            return clean_wetext_output(out)

        return _normalize
    except Exception as exc:
        logger.warning("Could not build wetext normalizer: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Frontend pipeline shared by the generation nodes
# --------------------------------------------------------------------------- #
class TextFrontend:
    def __init__(self, use_wetext: bool = True, fasttext_detector: Optional[FastTextLangDetector] = None):
        self.wetext_normalizer = build_wetext_normalizer() if use_wetext else None
        self.fasttext_detector = fasttext_detector

    def detect(self, text: str) -> str:
        return detect_language(
            text,
            fasttext_detector=self.fasttext_detector.detect_locale if self.fasttext_detector is not None else None,
        )

    def normalize(self, text: str, language: str) -> str:
        can_wetext = language in _WETEXT_LANGS or language == "Cantonese" or language.startswith("ZH_")
        if can_wetext and self.wetext_normalizer is not None:
            try:
                return clean_tn_spaces(self.wetext_normalizer(text))
            except Exception as exc:
                logger.warning("wetext normalization failed, using raw text: %s", exc)
        return clean_tn_spaces(text)

    def apply(self, text: str, language: Optional[str] = None, do_clean: bool = True, do_tn: bool = True,
              do_split: bool = True, token_max_n: int = 80, token_min_n: int = 60, merge_len: int = 20,
              tokenize: Optional[Callable[[str], int]] = None) -> tuple[str, str, List[str]]:
        """Clean, detect language, split and normalize. Returns (text, language, sentences)."""
        if do_clean:
            text = clean_text(text)
        if not text:
            raise ValueError("text is empty after cleaning")
        if language is None or language == "auto":
            language = self.detect(text)

        if do_split:
            if language == "Chinese" or language.startswith("ZH_") or language == "Cantonese":
                sentences = split_paragraph(text, lang="zh", token_max_n=token_max_n,
                                            token_min_n=token_min_n, merge_len=merge_len)
            else:
                sentences = split_paragraph(text, tokenize=tokenize, lang="en", token_max_n=token_max_n,
                                            token_min_n=token_min_n, merge_len=merge_len)
        else:
            sentences = [text]

        if do_tn:
            sentences = [self.normalize(s, language) for s in sentences]
        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            raise ValueError("all sentences are empty after normalization")
        return "".join(sentences), language, sentences
