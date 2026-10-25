# Development and packaging

## Source layout

| Path | Purpose |
| --- | --- |
| `public/` | Browser game, generation pipeline, physics and diagnostics |
| `public/models/` | Committed manifests and tokenizer; untracked local model weights |
| `public/runtime/` | Generated browser runtime files, excluded from Git |
| `scripts/` | Runtime preparation, model conversion, packaging and local diagnostics |
| `tests/` | JavaScript behavior tests and optional Python model checks |
| `docs/images/` | Selected game UI screenshots |
| `dist/` | Generated player files and ZIP, excluded from Git |
| `artifacts/` | Local diagnostics and benchmark output, excluded from Git |

## Work on the game

Use Node.js 22 or later. Python 3.10 or later is needed only for the optional static preview server, ZIP packaging and model export tools.

```sh
npm ci --ignore-scripts
npm test
npm run prepare-runtime
```

`prepare-runtime` copies the pinned ONNX Runtime Web, Tokenizers.js and Matter.js assets from npm packages. Runtime binaries and `node_modules/` are not committed. `npm test` needs the committed manifests and tokenizer, but does not need ONNX weights or a GPU. The model-free suite should run on Windows, macOS and Linux; browser hardware support is a separate check.

Download `TrickyWords.zip` from the GitHub Release and copy the ten files inside its `models` folder into `public/models/`. Keep the manifests and tokenizer already present in the source checkout.

```sh
python server.py
```

Open `http://127.0.0.1:8765/`, select `public/models`, then load the AI models. The server binds to loopback and serves static files. It does not receive player prompts or run inference. Stop it with Ctrl+C. The player ZIP does not include this server or `start.cmd`.

The isolated `background-test.html` diagnostic reads server-hosted model files and is intended for development. The main game reads selected local files and does not fall back to downloading model weights.

## Required model pack

The folder picker requires the graph/data pairs referenced by the committed manifests:

```text
text-compact-v2.onnx
text-compact-v2.data
dit-compact-v2.onnx
dit-compact-v2.data
vae-q4-compact-v1.onnx
vae-q4-compact-v1.data
background-compact-v2.onnx
background-compact-v2.data
background-compact-cpu-v2.onnx
background-compact-cpu-v2.data
```

These files total about 1.075 GB. The active native model set is about 0.881 GB; the CPU background fallback makes the active set about 0.976 GB. These are file sizes, not runtime memory estimates. The picker checks filenames, duplicates and byte sizes. File-size validation is not cryptographic verification.

The original Hugging Face PyTorch/safetensors checkpoint cannot replace this ONNX pack. The manifests and tokenizer are application assets, so selecting a folder cannot override the model configuration.

## Build the player download

With prepared weights and runtimes in place:

```sh
npm run build-download
python scripts/package-download.py
```

The output is `dist/TrickyWords/` and `dist/TrickyWords.zip`. The ZIP adds the model weights and bundled notices to the built game. Packaging checks model byte sizes, the expected file list and ZIP CRCs before replacing the archive.

The HTML inlines its styles and starts feedback before loading four classic JavaScript files. Those files embed manifests, tokenizer data, calibration assets, runtime modules and WASM. Modules, workers and WASM use in-memory Blob URLs. The content security policy blocks HTTP fetches during player initialization and inference. External model data stays in user-selected File/Blob objects to avoid copying the entire pack into JavaScript arrays.

The generated folder is intended for direct browser opening. It has been tested over local HTTP in Windows Chromium. Direct `file://` opening, macOS and the final Firefox package remain unverified. Do not present those targets as tested. CPU background removal uses one thread without cross-origin isolation, including the intended local-file launch.

The download keeps explicit folder selection after each reopen or refresh. No local server, native launcher or browser security changes are part of the player package.

## Rebuild the model exports

This is optional and substantially heavier than rebuilding the game. Use an isolated Python environment and allow several GB of downloads and temporary files. The pinned CPU PyTorch dependency in `requirements-export.txt` matches the Windows preparation environment; other platforms may need their supported PyTorch wheel. Cross-platform model export has not been verified.

Create and activate a virtual environment using the commands for your operating system. Then:

```sh
python -m pip install -r requirements-export.txt
git clone https://github.com/Amshaker/Mobile-O.git vendor/Mobile-O
git -C vendor/Mobile-O checkout 91c255080a0130846fe5b5cd54ff5af3c05ab2b9
```

The exporters can overwrite local manifests and models. Before a full re-export, preserve the existing pack and start with an empty `public/models/` directory. This avoids carrying compact-export metadata into intermediate exports. Exporters download the upstream model revisions available at preparation time and record revisions in their manifests; a future export may not reproduce the release byte for byte.

```sh
python scripts/export_models.py
python -c "from pathlib import Path; Path('artifacts').mkdir(exist_ok=True)"
python scripts/export_background_fp32.py
python scripts/export_compact.py
python -m unittest discover -s tests -p 'test_model_exports.py'
```

The pipeline exports actual Mobile-O text conditioning, diffusion and VAE decoder weights. It rewrites diffusion Clip operations as equivalent Max/Min nodes and promotes affected background Pad operations to FP32 for browser shader compatibility. Compact conversion quantizes eligible constant matrix and embedding weights to INT4, lowers eligible pointwise convolutions to matrices, and stores large tensors in external `.data` files. Generation remains at 512 × 512, eight diffusion steps and guidance 1.5.

The Python suite needs intermediate original exports as well as compact files. It cannot run from the ten-file player pack alone. Some profiling scripts are Windows-specific and require additional tools such as `psutil`; they are not part of the player flow or normal CI.

After changing model assets, rerun both test suites and browser generation checks. Keep each graph with its matching data file. Do not export components concurrently: exporters update the same manifests.

## Verification and releases

The JavaScript suite covers word validation and history, tokenization parity, scheduling, model selection and loading, cancellation, calibration worker cleanup, masks, polygon geometry, stacking, mouse targets, startup feedback and stage state.

Real Windows Chromium checks have covered setup, generated pieces, Stop, mouse aiming and wheel rotation, A/D/Q/E, click/S/Space drop, typing without triggering game keys, and stage completion/reset. UI captures have been inspected in a 390-pixel layout fixture. These checks do not establish support for integrated GPUs, 8 GB RAM machines, every browser or direct-file origins.

Keep checkpoints, weights, generated runtimes, local reports and ZIPs out of Git. Commit the source, manifests/tokenizer, tests, docs and notices. Attach the complete ZIP and SHA-256 checksum file to GitHub Releases. The early preview is tagged `v0.1.0-alpha.1` and marked as a prerelease.

Retain [THIRD_PARTY.md](../THIRD_PARTY.md) and all copied licenses. Mobile-O adaptations are distributed under CC BY-NC-SA 4.0; publishing this source does not grant commercial model-use permission.
