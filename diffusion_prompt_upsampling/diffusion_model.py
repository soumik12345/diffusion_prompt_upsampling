from typing import List, Optional

import dspy
import torch
import weave
from diffusers import AutoPipelineForText2Image, DiffusionPipeline

from .utils import base64_encode_image


STOCK_NEGATIVE_PROMPT = "frame, border, 2d, ugly, static, dull, monochrome, distorted face, deformed fingers, scary, horror, nightmare, deformed lips, deformed eyes, deformed hands, deformed legs, impossible physics, absurdly placed objects"


class PromptUpsamplingSignature(dspy.Signature):
    base_prompt = dspy.InputField()
    answer = dspy.OutputField(
        desc="Create an imaginative image descriptive caption for the given base prompt."
    )


class StableDiffusionXLModel(weave.Model):

    model_name_or_path: str
    enable_cpu_offfload: bool
    completions: Optional[List[dspy.Prediction]] = None
    diffusion_prompt_upsampler: Optional[dspy.Module] = None
    _pipeline: DiffusionPipeline
    _upsampler_llm: dspy.Module

    def __init__(self, model_name_or_path: str, enable_cpu_offfload: bool):
        super().__init__(
            model_name_or_path=model_name_or_path,
            enable_cpu_offfload=enable_cpu_offfload,
        )
        self.completions = (
            self.get_completion_rationales()
            if self.completions is None
            else self.completions
        )
        self.diffusion_prompt_upsampler = dspy.MultiChainComparison(
            PromptUpsamplingSignature, M=len(self.completions)
        )
        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        if self.enable_cpu_offfload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline = self._pipeline.to("cuda")
        self._upsampler_llm = dspy.OpenAI(
            model="gpt-4o",
            system_prompt="""
You are part of a team of bots that creates images. You work with an assistant bot that will draw anything
you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking
through the trees" will trigger your partner bot to output an image of a forest morning, as described.
You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to
take their short prompts and make them extremely detailed and descriptive.

There are a few rules to follow:
- You will only ever output a single image description per user request.
- Often times, the base prompt might consist of spelling mistakes or grammatical errors. You should correct
    such errors before making them extremely detailed and descriptive.
- Image descriptions must be between 15-80 words. Extra words will be ignored.
""",
        )

    def get_completion_rationales(self) -> List[dspy.Prediction]:
        return [
            dspy.Prediction(
                rationale="a man holding a sword",
                answer="a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head.",
            ),
            dspy.Prediction(
                rationale="a frog playing dominoes",
                answer="a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth, and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest, with a large tree in the background.",
            ),
            dspy.Prediction(
                rationale="A bird scaring a scarecrow",
                answer="A large, vibrant bird with an impressive wingspan swoops down from the sky, letting out a piercing call as it approaches a weathered scarecrow in a sunlit field. The scarecrow, dressed in tattered clothing and a straw hat, appears to tremble, almost as if it's coming to life in fear of the approaching bird.",
            ),
            dspy.Prediction(
                rationale="Paying for a quarter-sized pizza with a pizza-sized quarter",
                answer="A person is standing at a pizza counter, holding a gigantic quarter the size of a pizza. The cashier, wide-eyed with astonishment, hands over a tiny, quartersized pizza in return. The background features various pizza toppings and other customers, all of them equally amazed by the unusual transaction.",
            ),
            dspy.Prediction(
                rationale="a quilt with an iron on it",
                answer="a quilt is laid out on a ironing board with an iron resting on top. the quilt has a patchwork design with pastel-colored strips of fabric and floral patterns. the iron is turned on and the tip is resting on top of one of the strips. the quilt appears to be in the process of being pressed, as the steam from the iron is visible on the surface. the quilt has a vintage feel and the colors are yellow, blue, and white, giving it an antique look.",
            ),
            dspy.Prediction(
                rationale="a furry humanoid skunk",
                answer="In a fantastical setting, a highly detailed furry humanoid skunk with piercing eyes confidently poses in a medium shot, wearing an animal hide jacket. The artist has masterfully rendered the character in digital art, capturing the intricate details of fur and clothing texture.",
            ),
            dspy.Prediction(
                rationale="An icy landscape under a starlit sky",
                answer="An icy landscape under a starlit sky, where a magnificent frozen waterfall flows over a cliff. In the center of the scene, a f ire burns bright, its flames seemingly frozen in place, casting a shimmering glow on the surrounding ice and snow.",
            ),
            dspy.Prediction(
                rationale="A fierce garden gnome warrior",
                answer="A fierce garden gnome warrior, clad in armor crafted from leaves and bark, brandishes a tiny sword and shield. He stands valiantly on a rock amidst a blooming garden, surrounded by colorful flowers and towering plants. A determined expression is painted on his face, ready to defend his garden kingdom.",
            ),
            dspy.Prediction(
                rationale="A ferret in a candy jar",
                answer="A mischievous ferret with a playful grin squeezes itself into a large glass jar, surrounded by colorful candy. The jar sits on a wooden table in a cozy kitchen, and warm sunlight filters through a nearby window.",
            ),
            dspy.Prediction(
                rationale="cartoon drawing of an astronaut riding a horse",
                answer="Cartoon drawing of an outer space scene. Amidst floating planets and twinkling stars, a whimsical horse with exaggerated features rides an astronaut, who swims through space with a jetpack, looking a tad overwhelmed.",
            ),
            dspy.Prediction(
                rationale="A smafml vessef epropoeilled on watvewr by ors, sauls, or han engie.",
                answer="A small vessel, propelled on water by oars, sails, or an engine, floats gracefully on a serene lake. the sun casts a warm glow on the water, reflecting the vibrant colors of the sky as birds fly overhead.",
            ),
        ]

    @weave.op()
    def predict(
        self,
        base_prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = 50,
        image_size: Optional[int] = 1024,
        guidance_scale: Optional[float] = 7.0,
        upsample_prompt: Optional[bool] = True,
    ) -> str:
        with dspy.context(lm=self._upsampler_llm):
            prompt_upsampler_response = (
                self.diffusion_prompt_upsampler(
                    self.completions, base_prompt=base_prompt
                ).answer
                if upsample_prompt
                else base_prompt
            )
        image = self._pipeline(
            prompt=prompt_upsampler_response,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
        ).images[0]
        return base64_encode_image(image)
