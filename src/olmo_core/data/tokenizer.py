from dataclasses import dataclass
from typing import Optional

from ..config import Config, StrEnum

__all__ = [
    "TokenizerConfig",
    "TokenizerName",
]


class TokenizerName(StrEnum):
    """
    An enumeration of tokenizer identifiers commonly used OLMo researchers.
    """

    dolma2 = "allenai/dolma2-tokenizer"
    """
    The dolma2 tokenizer.
    """

    dolma2_sigdig = "allenai/dolma2-tokenizer-sigdig"
    """
    The R2L dolma2 tokenizer.
    """

    gpt_neox_olmo_dolma_v1_5 = "allenai/gpt-neox-olmo-dolma-v1_5"
    """
    A modified GPT NeoX tokenizer.
    """

    gpt2 = "gpt2"
    """
    The base GPT2 tokenizer.
    """


@dataclass
class TokenizerConfig(Config):
    """
    A configuration class that represents a tokenizer.
    """

    vocab_size: int
    """
    The vocab size.
    """

    eos_token_id: int
    """
    The end-of-sentence token ID.
    """

    pad_token_id: int
    """
    The padding token ID.
    """

    bos_token_id: Optional[int] = None
    """
    The begin-of-sentence token ID.
    """

    identifier: Optional[str] = None
    """
    The identifier of the tokenizer. Could be a path or HuggingFace identifier.
    """

    def padded_vocab_size(self, pad_multiple: int = 128) -> int:
        """
        Returns the vocab size padded to be a multiple of ``pad_multiple``.
        This is useful to set model embeddings to this number to increase throughput.
        """
        return pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)

    @classmethod
    def dolma2(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.dolma2` tokenizer config.
        """
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            identifier=TokenizerName.dolma2,
        )

    @classmethod
    def dolma2_sigdig(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.dolma2_sigdig` tokenizer config.
        """
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            bos_token_id=100257,
            identifier=TokenizerName.dolma2_sigdig,
        )

    @classmethod
    def gpt_neox_olmo_dolma_v1_5(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.gpt_neox_olmo_dolma_v1_5` tokenizer config.
        """
        return cls(
            vocab_size=50280,
            eos_token_id=50279,
            pad_token_id=1,
            identifier=TokenizerName.gpt_neox_olmo_dolma_v1_5,
        )

    @classmethod
    def gpt2(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.gpt2` tokenizer config.
        """
        return cls(
            vocab_size=50257,
            eos_token_id=50256,
            bos_token_id=50256,
            pad_token_id=50256,
            identifier=TokenizerName.gpt2,
        )

    @classmethod
    def from_hf(cls, identifier: str) -> "TokenizerConfig":
        """
        Initialize a tokenizer config from a model on HuggingFace.

        :param identifier: The HF model identifier, e.g. "meta-llama/Llama-3.2-1B".
        """
        import json

        from cached_path import cached_path

        tokenizer_config_path = cached_path(f"hf://{identifier}/tokenizer_config.json")
        with tokenizer_config_path.open() as f:
            tokenizer_config = json.load(f)
        model_config_path = cached_path(f"hf://{identifier}/config.json")
        with model_config_path.open() as f:
            model_config = json.load(f)

        return cls(
            vocab_size=model_config["vocab_size"],
            eos_token_id=model_config["eos_token_id"] if tokenizer_config.get("eos_token") else None,
            pad_token_id=model_config.get("pad_token_id", model_config["eos_token_id"]) if tokenizer_config.get("pad_token") or tokenizer_config.get("eos_token") else None,
            bos_token_id=model_config.get("bos_token_id") if tokenizer_config.get("bos_token") else None,
            identifier=identifier,
        )
