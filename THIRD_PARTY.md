# Third-party sources

- **Mobile-O**, Abdelrahman Shaker, Ahmed Heakl, Jaseel Muhammad, Ritesh Thawkar, Omkar Thawakar, Senmao Li, Hisham Cholakkal, Ian Reid, Eric P. Xing, Salman Khan and Fahad Shahbaz Khan. [Source](https://github.com/Amshaker/Mobile-O), source commit `91c255080a0130846fe5b5cd54ff5af3c05ab2b9`. The connector is imported unchanged during export. Mobile-O's [LICENSE.txt](https://github.com/Amshaker/Mobile-O/blob/main/LICENSE.txt) is **CC BY-NC-SA 4.0**. Its README requires written permission for commercial use. The Hugging Face card uses different license metadata; this distribution follows the repository's stated CC BY-NC-SA 4.0 terms. Converted and quantized Mobile-O materials, including this project's contributions to those adaptations, are distributed under that same license. The tokenizer and trained weights come from [Amshaker/Mobile-O-0.5B](https://huggingface.co/Amshaker/Mobile-O-0.5B); the manifest records the downloaded model revision. See the [paper](https://arxiv.org/abs/2602.20161).
- **SANA component configurations**, [Efficient-Large-Model/Sana_600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers). The transformer and VAE weights used here come from the Mobile-O checkpoint. The upstream SANA model supplies architecture and scheduler configurations.
- **BiRefNet lite 512 ONNX**, [studioludens/birefnet-lite-512](https://huggingface.co/studioludens/birefnet-lite-512), based on [ZhengPeng7/BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite). MIT. Uses the supplied FP16 ONNX export, with ImageNet input normalization and sigmoid output conversion. The manifest records its revision. The model's license is saved in `public/birefnet-LICENSE.txt`.
- **ONNX Runtime Web**, Microsoft, MIT. Version pinned in `package.json`; distributed runtime files come from the installed npm package.
- **Tokenizers.js**, Hugging Face, Apache-2.0. Used for browser tokenization; version pinned in `package.json`.
- **Matter.js**, Liam Brummitt, MIT. Version 0.20.0 provides browser rigid-body physics. The existing convex decomposition supplies its compound bodies without another decomposition dependency. [Source](https://github.com/liabru/matter-js). License saved in `public/matter-LICENSE.txt`.
- **Diffusers and Transformers**, Hugging Face, Apache-2.0; **PyTorch**, BSD-style license; **ONNX**, Apache-2.0. Used locally during preparation. Versions are in `requirements-export.txt`.

Keep original license and attribution notices with redistributed models and libraries. Nothing in this distribution grants commercial model-use permission or claims endorsement by upstream authors. Original Tricky Words application code has no separate reuse license selected; see `LICENSE.md` in the source repository.

The browser conversion quantizes Mobile-O's constant text, diffusion and decoder matrix and embedding weights to INT8, rewrites eligible pointwise convolutions as matrices, and stores large weights in external data files. Remaining weights and activations retain FP16, with original FP32 attention. It also batches classifier-free guidance, replaces diffusion Clip nodes with Max/Min and promotes affected BiRefNet padding operations to FP32. Generation remains at 512px with eight diffusion steps and guidance 1.5. These modifications do not change the upstream license or imply endorsement by the authors. The upstream license includes its disclaimer of warranties; retain the complete license texts supplied with the package.

## Mobile-O citation

```bibtex
@article{shaker2026mobileo,
  title={Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device},
  author={Shaker, Abdelrahman and Heakl, Ahmed and Muhammad, Jaseel and Thawkar, Ritesh and Thawakar, Omkar and Li, Senmao and Cholakkal, Hisham and Reid, Ian and Xing, Eric P. and Khan, Salman and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:2602.20161},
  year={2026}
}
```
