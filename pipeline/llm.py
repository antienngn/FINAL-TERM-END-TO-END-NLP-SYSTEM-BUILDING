"""

V100 hỗ trợ float16
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default models (V100 fp16)
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  


class LLM:
    def __init__(self, model_id: str, device: str = "cuda",
                 torch_dtype=torch.float16, max_new_tokens: int = 128):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        print(f"Loading LLM: {model_id} (dtype={torch_dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_ids = self._get_eos_ids()
        print(f"LLM ready (eos_ids={self.eos_ids})")

    def _get_eos_ids(self):
        ids = [self.tokenizer.eos_token_id]
        if "llama" in self.model_id.lower() or "Llama" in self.model_id:
            try:
                eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot is not None and eot != self.tokenizer.unk_token_id:
                    ids.append(eot)
            except Exception:
                pass
        return ids

    @torch.no_grad()
    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,  
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_ids,
        )
        gen_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def load_llm(name: str, **kwargs) -> LLM:
    aliases = {
        "llama": LLAMA_MODEL_ID,
        "llama3": LLAMA_MODEL_ID,
        "llama-3": LLAMA_MODEL_ID,
        "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen": QWEN_MODEL_ID,
        "qwen2.5": QWEN_MODEL_ID,
    }
    model_id = aliases.get(name.lower(), name)
    return LLM(model_id, **kwargs)
