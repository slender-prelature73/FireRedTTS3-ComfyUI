"""FireRedTTS3 text tokenizer, language tags, and chatml prompt composers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CHATML_LATENT_IN_PAD_SYM = "<|image_pad|>"
CHATML_LATENT_IN_PAD_ID = 151655
CHATML_LATENT_OUT_PAD_SYM = "<|video_pad|>"
CHATML_LATENT_OUT_PAD_ID = 151656

# Prompt latents are scaled for the instruct model only (upstream REDAE_SCALE).
REDAE_PROMPT_SCALE = {"base": 1.0, "instruct": 0.4}

MULTI_LANG_TAGS = [
    "<|Chinese|>", "<|English|>", "<|Cantonese|>",
    "<|Japanese|>", "<|Korean|>", "<|Spanish|>",
    "<|French|>", "<|Russian|>", "<|Arabic|>",
    "<|Turkish|>", "<|Indonesian|>", "<|Portuguese|>",
    "<|Italian|>", "<|Dutch|>", "<|Vietnamese|>",
    "<|German|>", "<|Ukrainian|>", "<|Thai|>",
    "<|Polish|>", "<|Romanian|>", "<|Greek|>",
    "<|Czech|>", "<|Finnish|>", "<|Hindi|>",
]

MULTI_DIALECT_TAGS = [
    "<|ZH_Anhui|>", "<|ZH_Fujian|>", "<|ZH_Gansu|>",
    "<|ZH_Guizhou|>", "<|ZH_Hebei|>", "<|ZH_Henan|>",
    "<|ZH_Hubei|>", "<|ZH_Hunan|>", "<|ZH_Jiangxi|>",
    "<|ZH_Liaoning|>", "<|ZH_Minnan|>", "<|ZH_Ningxia|>",
    "<|ZH_Shaanxi|>", "<|ZH_Shandong|>", "<|ZH_Shanghai|>",
    "<|ZH_Shanxi|>", "<|ZH_Sichuan|>", "<|ZH_Tianjin|>",
    "<|ZH_Wenzhou|>", "<|ZH_Wu|>", "<|ZH_Yunnan|>",
]

LANGUAGE_CHOICES = ["auto"] + [tag.strip("<|>") for tag in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS]

SPECIAL_TOKENS = [
    "<|sosp|>", "<|eosp|>", "<|empty|>", "<|Human|>", "<|SpeechLM|>",
    "<|sostm|>", "<|eostm|>", "<|sot|>", "<|eot|>",
    "<|TEXT_ONLY|>", "<|AUDIO_ONLY|>", "<|ASR|>", "<|TTS|>",
    "<|INTERLEAVE|>", "<|UNDERSTANDING|>",
    *[f"<|placeholder_{i:03d}|>" for i in range(1, 193)],
    *MULTI_LANG_TAGS,
    *MULTI_DIALECT_TAGS,
    "<|edit|>", "<|frame_patch|>", "<|end_edit|>",
]


def load_text_tokenizer(tokenizer_dir: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=str(tokenizer_dir))
    vocab = tokenizer.get_vocab()
    missing = [token for token in SPECIAL_TOKENS if token not in vocab]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
    return tokenizer


def convert_to_chatml(text_in: str, latent_in_len: int = 0, text_out: str = "", latent_out_len: int = 0,
                      latent_in_pad: str = CHATML_LATENT_IN_PAD_SYM,
                      latent_out_pad: str = CHATML_LATENT_OUT_PAD_SYM) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    input_message = {
        "role": "user",
        "content": [{"type": "text", "text": text_in + " /no_think"}],
    }
    if latent_in_len > 0:
        input_message["content"].insert(0, {"type": "audio", "audio": latent_in_pad * latent_in_len})
    messages.append(input_message)
    output_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "<think>\n\n</think>\n\n" + text_out}],
    }
    if latent_out_len > 0:
        output_message["content"].append({"type": "audio", "audio": latent_out_pad * latent_out_len})
    messages.append(output_message)

    chatml_str_list: list[str] = []
    for msg in messages:
        if isinstance(msg["content"], str):
            chatml_str_list.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        else:
            chatml_str_list.append(f"<|im_start|>{msg['role']}\n")
            for content in msg["content"]:
                if content["type"] == "text":
                    chatml_str_list.append(content["text"])
                elif content["type"] == "audio":
                    chatml_str_list.append(f"<|sosp|>{content['audio']}<|eosp|>\n")
            chatml_str_list.append("<|im_end|>\n")
    return "".join(chatml_str_list)


def compose_generate_input_tts(prompt_latent_len: int, prompt_text: str, text: str) -> str:
    text_in = "Convert text to speech.\n{}".format(prompt_text + text)
    chatml_str = convert_to_chatml(text_in=text_in, latent_out_len=prompt_latent_len)
    return chatml_str.removesuffix("<|eosp|>\n<|im_end|>\n")


def compose_generate_input_voice_design(instruction: str, text: str) -> str:
    text_in = "{}\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n{}".format(instruction, text)
    chatml_str = convert_to_chatml(text_in=text_in, text_out="<|sot|>")
    return chatml_str.removesuffix("<|im_end|>\n")


def compose_generate_input_semantic_edit(instruction: str, audio_in_latent_len: int) -> str:
    text_in = "Identify the content of the audio. {}".format(instruction.strip())
    chatml_str = convert_to_chatml(text_in=text_in, latent_in_len=audio_in_latent_len, text_out="<|sot|>")
    return chatml_str.removesuffix("<|im_end|>\n")


def compose_generate_input_acoustic_edit(instruction: str, audio_in_latent_len: int) -> str:
    chatml_str = convert_to_chatml(text_in=instruction, latent_in_len=audio_in_latent_len, latent_out_len=1)
    return chatml_str.removesuffix("<|video_pad|><|eosp|>\n<|im_end|>\n")
