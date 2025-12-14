"""Base LLM solver for AIMO problems."""

import os
from typing import Any, Optional, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class LLMSolver:
    """
    Base LLM solver that can use HuggingFace models or API-based models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_api: bool = False,
        api_provider: str = "openai",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        device: str = "cuda",
    ):
        """
        Initialize LLM solver.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            use_api: Whether to use API instead of local model
            api_provider: API provider ("openai", "anthropic", etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device to use ("cuda", "cpu", "mps")
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_provider = api_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

        if not use_api and model_name:
            self._load_model()

    def _load_model(self) -> None:
        """Load HuggingFace model and tokenizer."""
        if self.model is not None:
            return
        if self.model_name is None:
            raise ValueError("model_name must be set to load a local model")

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _call_api(self, prompt: str) -> str:
        """Call API-based model."""
        if self.api_provider == "openai":
            import openai

            openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            openai_response = openai_client.chat.completions.create(
                model=self.model_name or "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = openai_response.choices[0].message.content
            if content is None:
                raise RuntimeError("OpenAI response had empty message content")
            return content
        elif self.api_provider == "anthropic":
            import anthropic

            anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            anthropic_response = anthropic_client.messages.create(
                model=self.model_name or "claude-3-opus-20240229",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            first_block = anthropic_response.content[0]
            return cast(Any, first_block).text
        else:
            raise ValueError(f"Unknown API provider: {self.api_provider}")

    def _call_local_model(self, prompt: str) -> str:
        """Call local HuggingFace model."""
        if self.tokenizer is None or self.model is None:
            self._load_model()
        assert self.tokenizer is not None and self.model is not None

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id set")

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": pad_token_id,
            }
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature
            outputs = cast(Any, self.model).generate(**inputs, **generate_kwargs)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the newly generated part
        return generated_text[len(prompt):].strip()

    def solve(self, problem_statement: str) -> int:
        """
        Solve a problem using the LLM.

        Args:
            problem_statement: LaTeX problem statement

        Returns:
            Answer as integer in [0, 99999]
        """
        prompt = self._create_prompt(problem_statement)

        if self.use_api:
            response = self._call_api(prompt)
        else:
            response = self._call_local_model(prompt)

        answer = self._extract_answer(response)
        return answer

    def _create_prompt(self, problem_statement: str) -> str:
        """Create prompt for the LLM."""
        from .prompt_engineer import AIMPOPromptEngineer
        
        engineer = AIMPOPromptEngineer(use_few_shot=True, num_examples=2)
        return engineer.create_prompt(problem_statement, strategy="chain_of_thought")

    def _extract_answer(self, response: str) -> int:
        """
        Extract answer from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted integer answer
        """
        from .answer_extractor import extract_answer
        return extract_answer(response, use_structured=True)

