from __future__ import annotations

import logging
import re
from .prompts import build_prompt

log = logging.getLogger(__name__)
STOP_WORDS = {"a", "an", "and", "are", "can", "do", "does", "explain", "for", "how", "i", "in", "is", "it", "of", "on", "the", "to", "what", "when", "where", "which", "who", "why", "with"}


def _terms(text: str) -> set[str]:
    terms = re.findall(r"[a-z0-9]+", text.lower())
    return {"warehouse" if term == "warehousing" else term for term in terms if term not in STOP_WORDS and len(term) > 2}


class Generator:
    def __init__(self, model_name: str, device: str, max_new_tokens: int, strict_extractive: bool = True):
        self.model_name, self.device, self.max_new_tokens, self.strict_extractive, self.pipe = model_name, device, max_new_tokens, strict_extractive, None

    def _load(self):
        if self.pipe:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        use_gpu = self.device != "cpu" and torch.cuda.is_available()
        if self.device != "cpu" and not use_gpu:
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or set DEVICE=cpu in .env.")
        model_kwargs = {"torch_dtype": torch.float16 if use_gpu else torch.float32}
        if use_gpu:
            model_kwargs.update(device_map="auto", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=self.max_new_tokens)

    def answer(self, question: str, matches: list[dict], threshold: float, history: list[dict] | None = None) -> str:
        question_terms = _terms(question)
        retrieved_terms = _terms(" ".join(x["text"] for x in matches[:2]))
        required_matches = min(2, len(question_terms))
        if not matches or max(x.get("dense_score", 0.0) for x in matches) < threshold or len(question_terms & retrieved_terms) < required_matches:
            return "This information is not available in the uploaded course material, so I cannot answer it from the notes."
        if self.strict_extractive:
            return self._extract(question, question_terms, matches) + self._citations(matches)
        evidence = self._extract(question, question_terms, matches).replace("Relevant passages from the uploaded notes:\n\n", "")
        context = evidence
        try:
            self._load(); result = self.pipe(build_prompt(question, context, history or []), return_full_text=False)[0]["generated_text"]
            return result.strip() + self._citations(matches)
        except Exception as exc:
            log.warning("Generation unavailable: %s", exc)
            return "Generation model is unavailable. Retrieved course context is shown below."

    @staticmethod
    def _citations(matches: list[dict]) -> str:
        unique = []
        for match in matches:
            label = f"{match.get('source', 'uploaded notes')} | {match.get('module') or match.get('chapter') or match.get('topic') or 'course material'} | p. {match.get('page_number', 'n/a')}"
            if label not in unique:
                unique.append(label)
        return "\n\nSources:\n" + "\n".join(f"- {label}" for label in unique[:3])

    @staticmethod
    def _extract(question: str, question_terms: set[str], matches: list[dict]) -> str:
        """Return original note passages only; this mode cannot invent unsupported facts."""
        candidates = []
        definition_question = bool(re.match(r"\s*(what is|define|meaning of)", question.lower()))
        for match in matches:
            prose = re.sub(r"\s+", " ", match["text"]).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", prose):
                sentence = sentence.strip(" -•")
                score = len(question_terms & _terms(sentence))
                if definition_question and re.search(r"\b(is|refers to|defined)\b", sentence, re.I):
                    score += 2
                if score and len(sentence) > 60:
                    candidates.append((score, sentence))
        selected = [sentence for _, sentence in sorted(candidates, key=lambda item: item[0], reverse=True)[:3]]
        if not selected:
            selected = [re.sub(r"\s+", " ", matches[0]["text"]).strip()]
        return "Relevant passages from the uploaded notes:\n\n" + "\n\n".join(f"- {sentence}" for sentence in selected)
