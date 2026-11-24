"""Testmodeller.

Default: en liten slumpvis initierad Qwen2-modell med en lokalt tränad
BPE-tokenizer (ingen nedladdning). Med miljövariabeln NEXIS_REAL_MODEL
(t.ex. Qwen/Qwen2.5-1.5B-Instruct) körs samma tester på den riktiga modellen.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexis3.attention import Measurer, freeze, load_model  # noqa: E402
from nexis3.textgen import TASK_TEMPLATE, TOPICS, REPORT_QUESTION  # noqa: E402

# Som Qwen2.5: standardsystemprompt läggs till tyst om ingen systemprompt ges.
CHAT_TEMPLATE = (
    "{% if not messages or messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful "
    "assistant created to help. Answer questions.<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [TASK_TEMPLATE, REPORT_QUESTION, "You are a helpful assistant created to help. Answer questions."]
    for _, s, q, vals in TOPICS:
        corpus += [s.format(v=v) for v in vals] + [q]
    corpus += ["Dr. Smith arrived at 3.5 p.m. today. He left! Did she? Yes.",
               "Jag heter Anders. Vad heter du? Åsa och Örjan bor i Göteborg."]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=600, special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus * 3, tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=["<|im_start|>"])
    fast.chat_template = CHAT_TEMPLATE
    return fast


def _tiny_model(vocab, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      attn_implementation="eager")
    m = Qwen2ForCausalLM(cfg)
    m.generation_config.eos_token_id = None
    return freeze(m)


@pytest.fixture(scope="session")
def model_tok():
    real = os.environ.get("NEXIS_REAL_MODEL")
    if real:
        return load_model(real)
    tok = _tiny_tokenizer()
    return _tiny_model(len(tok)), tok


@pytest.fixture(scope="session")
def measurer(model_tok):
    m, t = model_tok
    return Measurer(m, t)
