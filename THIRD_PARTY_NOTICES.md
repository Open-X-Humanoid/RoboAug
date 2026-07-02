# Third-Party Notices

This project depends on the following third-party open-source software. None of
it is vendored in this repository; each component is installed from its upstream
source and remains the property of its respective authors under its own license.
Please refer to the upstream repositories for the full license texts and terms.

## External model libraries (installed separately, see the README)

| Component | Upstream | License |
| --- | --- | --- |
| GroundingDINO | https://github.com/IDEA-Research/GroundingDINO | Apache-2.0 |
| Segment Anything 2 (SAM 2) | https://github.com/facebookresearch/sam2 | Apache-2.0 |

## Runtime dependencies (installed via `requirements.txt`)

| Component | Upstream | License |
| --- | --- | --- |
| Stable Diffusion 3 Medium | https://huggingface.co/stabilityai/stable-diffusion-3-medium | Stability AI Community License |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Apache-2.0 |
| diffusers | https://github.com/huggingface/diffusers | Apache-2.0 |
| transformers | https://github.com/huggingface/transformers | Apache-2.0 |
| CLIP | https://github.com/openai/CLIP | MIT |

## Related work

The overall approach builds on ideas from
[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

---

Model weights are **not** distributed with this repository; download them from
the upstream sources above and comply with their respective licenses. In
particular, note that the Stable Diffusion 3 Medium weights are governed by the
Stability AI Community License, which has its own usage terms — review them
before any commercial use.
