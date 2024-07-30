import asyncio
from typing import Optional

import fire
import weave

from diffusion_prompt_upsampling.diffusion_model import StableDiffusionXLModel
from diffusion_prompt_upsampling.judge_model import OpenAIJudgeModel


def evaluate_upsampling(
    dataset_ref: str,
    upsample_prompt: bool,
    project_name: Optional[str] = "diffusion-prompt-upsample",
    entity_name: Optional[str] = None,
    evaluation_name: Optional[str] = None,
    diffusion_model_name_or_path: Optional[
        str
    ] = "stabilityai/stable-diffusion-xl-base-1.0",
    diffusion_model_enable_cpu_offfload: Optional[bool] = True,
    use_stock_negative_prompt: Optional[bool] = False,
    openai_model: Optional[str] = "gpt-4-turbo",
    judge_model_max_retries: Optional[int] = 5,
    judge_model_seed: Optional[int] = 42,
):
    project_name = (
        project_name if entity_name is None else f"{entity_name}/{project_name}"
    )
    weave.init(project_name=project_name)
    dataset = weave.ref(dataset_ref).get()
    diffusion_model = StableDiffusionXLModel(
        model_name_or_path=diffusion_model_name_or_path,
        enable_cpu_offfload=diffusion_model_enable_cpu_offfload,
        upsample_prompt=upsample_prompt,
        use_stock_negative_prompt=use_stock_negative_prompt,
    )
    judge_model = OpenAIJudgeModel(
        openai_model=openai_model,
        max_retries=judge_model_max_retries,
        seed=judge_model_seed,
    )
    evaluation = weave.Evaluation(
        name=evaluation_name, dataset=dataset, scorers=[judge_model.score]
    )
    with weave.attributes(
        {
            "upsample_prompt": upsample_prompt,
            "use_stock_negative_prompt": use_stock_negative_prompt,
            "enable_cpu_offfload": diffusion_model_enable_cpu_offfload,
        }
    ):
        asyncio.run(evaluation.evaluate(diffusion_model.predict))


if __name__ == "__main__":
    fire.Fire(evaluate_upsampling)
