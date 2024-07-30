from typing import Optional

import fire
import weave
from datasets import load_dataset


def publish_dataset(
    project_name: Optional[str] = "diffusion-prompt-upsample",
    entity_name: Optional[str] = None,
):
    project_name = (
        project_name if entity_name is None else f"{entity_name}/{project_name}"
    )
    weave.init(project_name=project_name)
    dataset = load_dataset("shunk031/DrawBench")["test"].rename_column(
        "prompts", "prompt"
    )
    rows = []
    for data in dataset:
        data["category"] = dataset.features["category"].names[int(data["category"])]
        rows.append(data)
    weave_dataset = weave.Dataset(
        name="DrawBench",
        rows=rows,
        description="""
Source: shunk031/DrawBench

Citation:
```
@article{saharia2022photorealistic,
  title={Photorealistic text-to-image diffusion models with deep language understanding},
  author={Saharia, Chitwan and Chan, William and Saxena, Saurabh and Li, Lala and Whang, Jay and Denton, Emily L and Ghasemipour, Kamyar and Gontijo Lopes, Raphael and Karagol Ayan, Burcu and Salimans, Tim and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={36479--36494},
  year={2022}
}
```""",
    )
    weave.publish(weave_dataset)


if __name__ == "__main__":
    fire.Fire(publish_dataset)
