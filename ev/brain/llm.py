import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ev.config import LLM_MODEL, LLM_ADAPTER, SYSTEM_PROMPT


class LLMBrain:
    def __init__(self, model_path: str | None = None, adapter_path: str | None = None):
        self._model_path = model_path or LLM_MODEL
        self._adapter_path = adapter_path or LLM_ADAPTER
        print(f"  Loading LLM: {self._model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map="auto",
            dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            attn_implementation="sdpa",
        )

        if self._adapter_path:
            from peft import PeftModel
            print(f"  Loading adapter: {self._adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

        self._model.eval()
        print("  LLM loaded.")

    def think(self, user_message: str, context: dict | None = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            context_parts = []
            if "face_recognized" in context:
                who = "Piolo (owner)" if context["face_recognized"] else "unknown person"
                context_parts.append(f"You can see: {who}")
            if "voice_verified" in context:
                status = "verified as owner" if context["voice_verified"] else "not verified"
                context_parts.append(f"Speaker: {status}")
            if context_parts:
                messages.append({
                    "role": "system",
                    "content": "Current perception: " + "; ".join(context_parts),
                })

        messages.append({"role": "user", "content": user_message})

        inputs = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return response.strip()
