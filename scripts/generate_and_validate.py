from typing import Optional

import dspy
import fire
import rich
import weave
from diffusion_prompt_upsampling.diffusion_model import (
    StableDiffusionXLModel,
    upsampler_llm,
)
from diffusion_prompt_upsampling.judge_model import OpenAIJudgeModel


def generate_and_validate(
    base_prompt: str,
    project_name: str,
    entity_name: Optional[str] = None,
    diffusion_model_name_or_path: Optional[
        str
    ] = "stabilityai/stable-diffusion-xl-base-1.0",
    diffusion_model_enable_cpu_offfload: Optional[bool] = True,
    openai_model: Optional[str] = "gpt-4-turbo",
    judge_model_max_retries: Optional[int] = 5,
    judge_model_seed: Optional[int] = 42,
):
    project_name = (
        project_name if entity_name is None else f"{entity_name}/{project_name}"
    )
    weave.init(project_name=project_name)
    diffusion_model = StableDiffusionXLModel(
        model_name_or_path=diffusion_model_name_or_path,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
    )
    judge_model = OpenAIJudgeModel(
        openai_model=openai_model,
        max_retries=judge_model_max_retries,
        seed=judge_model_seed,
    )
    with dspy.context(lm=upsampler_llm):
        base_prompt_image = diffusion_model.predict(
            base_prompt=base_prompt, upsample_prompt=False
        )
        upsampled_prompt_image = diffusion_model.predict(
            base_prompt=base_prompt, upsample_prompt=True
        )
    judgement_base_prompt_image = judge_model.predict(
        base_prompt=base_prompt, generated_image=base_prompt_image
    )
    judgement_upsampled_prompt_image = judge_model.predict(
        base_prompt=base_prompt, generated_image=upsampled_prompt_image
    )

    rich.print(f"{judgement_base_prompt_image=}")
    rich.print(f"{judgement_upsampled_prompt_image=}")


if __name__ == "__main__":
    fire.Fire(generate_and_validate)
