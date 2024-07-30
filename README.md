# Prompt Upsampling for Diffusion Models

Prompting for current generation of text-to-image diffusion models is extremely brittle (such as Stable Diffusion 2.1 and Stable Diffusion XL), i.e, its difficult to create an optimal prompting strategy to reliably generate images of a certain quality and sometimes even reliably following the prompt to generate the image.

This project aims to implement the prompt upsampling strategy as mentioned in the technical report from OpenAI accompanying DALL-E 3; [Improving Image Generation with Better Captions](https://cdn.openai.com/papers/dall-e-3.pdf) and evaluate against datasets like DrawBench and Open Parti-Prompts. 