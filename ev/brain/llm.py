from mlx_lm import load, generate

from ev.config import LLM_MODEL, SYSTEM_PROMPT


class LLMBrain:
    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or LLM_MODEL
        print(f"  Loading LLM: {self._model_path}")
        self._model, self._tokenizer = load(self._model_path)
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

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=256,
            verbose=False,
        )
        return response.strip()
