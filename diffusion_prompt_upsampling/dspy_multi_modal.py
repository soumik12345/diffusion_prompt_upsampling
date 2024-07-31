import os
import requests
from typing import Any, Dict, Literal, List

import weave
from weave.integrations.dspy.dspy_sdk import dspy_wrapper
from dsp import GPT3

from .utils import find_base64_images


class DSPyOpenAIMultiModalLM(GPT3):

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: str | None = None,
        api_provider: Literal["openai"] = "openai",
        api_base: str | None = None,
        model_type: Literal["chat", "text"] = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model, api_key, api_provider, api_base, model_type, system_prompt, **kwargs
        )

    @weave.op()
    def create_messages(self, prompt: str):
        images = find_base64_images(prompt)
        for image in images:
            prompt = prompt.replace(image, "")

        user_prompt = [{"type": "text", "text": prompt}]
        for image in images:
            user_prompt.append({"type": "image_url", "image_url": {"url": image}})
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @weave.op()
    def get_chat_reponse(
        self, messages: List[Dict[str, Any]], kwargs
    ) -> Dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY")
        return requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                **kwargs,
                "model": "gpt-4o",
                "messages": messages,
            },
        ).json()

    @weave.op()
    def basic_request(self, prompt: str, **kwargs):
        messages = self.create_messages(prompt)
        response = self.get_chat_reponse(messages, kwargs)
        self.history.append({"prompt": prompt, "response": response, "kwargs": kwargs})
        return response

    @weave.op()
    def request(self, prompt: str, **kwargs):
        return super().request(prompt, **kwargs)

    @dspy_wrapper(
        name="diffusion_prompt_upsampling.dspy_multi_modal.DSPyOpenAIMultiModalLM"
    )
    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        return super().__call__(prompt, only_completed, return_sorted, **kwargs)
