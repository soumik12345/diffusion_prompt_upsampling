import os
from typing import Any, Literal

import weave
from weave.integrations.dspy.dspy_sdk import dspy_wrapper
from dsp import GPT3
from openai import OpenAI

from .utils import find_base64_images


class DSPyOpenAIMultiModalLM(GPT3):

    def __init__(
        self,
        model: str = "gpt-4o",
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
        self.model_type = model
        self._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    def basic_request(self, prompt: str, **kwargs):
        messages = self.create_messages(prompt)
        response = self._openai_client.chat.completions.create(
            model=self.model_type, messages=messages, **kwargs
        )
        self.history.append({"prompt": prompt, "response": response, "kwargs": kwargs})
        return response

    @weave.op()
    def request(self, prompt: str, **kwargs):
        return super().request(prompt, **kwargs)

    @dspy_wrapper(name="DSPyOpenAIMultiModalLM")
    def __call__(
        self, prompt: str, only_completed: bool = True, **kwargs
    ) -> list[dict[str, Any]]:
        response = self.request(prompt, **kwargs)
        choices = (
            [choice for choice in response.choices if choice.finish_reason == "stop"]
            if only_completed and len(response.choices) != 0
            else response.choices
        )
        return [choice.message.content for choice in choices]
