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
    dataset = load_dataset("nateraw/parti-prompts")["train"]
    rows = [{"base_prompt": data["Prompt"]} for data in dataset]
    weave_dataset = weave.Dataset(
        name="parti-prompts",
        rows=rows,
        description="""
Source: nateraw/parti-prompts

PartiPrompts (P2) is a rich set of over 1600 prompts in English that we release as
part of this work. P2 can be used to measure model capabilities across various
categories and challenge aspects.

P2 prompts can be simple, allowing us to gauge the progress from scaling.
They can also be complex, such as the following 67-word description we created for
Vincent van Gogh's The Starry Night (1889):

> Oil-on-canvas painting of a blue night sky with roiling energy. A fuzzy and bright
    yellow crescent moon shining at the top. Below the exploding yellow stars and
    radiating swirls of blue, a distant village sits quietly on the right. Connecting
    earth and sky is a flame-like cypress tree with curling and swaying branches on
    the left. A church spire rises as a beacon over rolling blue hills.
""",
    )
    weave.publish(weave_dataset)


if __name__ == "__main__":
    fire.Fire(publish_dataset)
