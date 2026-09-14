# Kaggle MiniMax H3 package

This folder is a reusable Kaggle notebook package for a temporary, local ComfyUI-based MiniMax H3 video pipeline. The result files are deliberately kept under this folder's `results/` directory.

The pipeline accepts a natural-language prompt, character/scene/outfit/motion/audio references, optional first/last frames, duration, aspect ratio, resolution, seed, and quality mode. It selects Ref2VA, FL2VA, or T2VA without silently dropping incompatible conditions. Generated MP4 files are accompanied by JSON manifests containing the prompt, reference roles, workflow hash, model revision, backend decision, device map, safety limits, transfer measurements, telemetry, timing, validation, and failed attempts.

## Architecture decision

Two T4 devices are not treated as one pooled VRAM device. Before generation, the package evaluates four real multi-GPU routes:

| Route | Same generation? | Planned placement | T4 default decision |
| --- | --- | --- | --- |
| SGLang H3 | Yes, tensor/encoder parallel | DiT tensor parallel across both; encoder parallel; layerwise CPU offload for evicted components | Capacity/installation blocked until Kaggle telemetry proves otherwise |
| Diffusers automatic layer dispatch | Yes, if the H3 class loads | Automatically discovers the indexed transformer block sequence, assigns contiguous intact blocks across every visible GPU, then uses CPU/disk overflow tiers | Experimental; selected before the one-GPU fallback when optional dependencies and live safety headroom are present |
| Diffusers balanced | Intended, but requires map inspection | `device_map="balanced"` probe plus CPU group offload | Blocked by H3-specific capacity/contract checks |
| Diffusers explicit split | Yes if the modular pipeline loads and runs | text encoder on GPU 1; transformer and VAEs on GPU 0; CPU as third tier | Blocked by the documented 48 GiB-card int8 example envelope |
| ComfyUI H3 sequence parallel | Yes, if the optional custom node is installed | `MiniMaxH3SPUNETLoader`, `world_size=2`, sequence/head sharding | Blocked by its replicated DiT footprint on 15 GiB T4s |
| ComfyUI phase-aware H3 dispatch | Yes if the native model exposes dispatchable blocks and live verification passes | Qwen language layers across GPU0/GPU1 during conditioning; intact transformer blocks across GPU0/GPU1; transformer released; audio VAE GPU0 and video VAE GPU1; CPU/disk overflow tiers | Experimental custom-loader path; fails instead of silently downgrading when two GPUs are visible |

The native ComfyUI one-GPU path remains `comfyui_native_fallback` and is
explicitly labeled as a fallback. The direct notebook frontend now exposes
both T4s and uses the phase-aware custom sampler. It starts with `--lowvram`
but leaves `--cpu-vae` disabled so the decode nodes can place audio and video
VAEs on different devices. Set `KAGGLE_H3_PHASE_SHARDING=off` only when the
one-GPU fallback is intentional.

This conservative result is grounded in the published implementation envelopes, not an assumption that explicit dispatch is impossible. The official H3 ComfyUI integration is documented in [MiniMax H3's repository](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.md), the native node implementation is [here](https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy_extras/nodes_minimax_h3.py), and current official ComfyUI H3 model files are listed in [Comfy-Org's model README](https://huggingface.co/Comfy-Org/MiniMax-H3/raw/main/README.md). The SGLang route follows the [official H3 diffusion cookbook](https://raw.githubusercontent.com/sgl-project/sglang/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx). The explicit Diffusers component placement is based on the [official H3 pipeline documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3). The optional ComfyUI sequence-parallel node is [buqi-code/buqi-minimax-h3-multigpu](https://github.com/buqi-code/buqi-minimax-h3-multigpu).

The final Kaggle report must be read from the run manifest. It must not claim both GPUs were used merely because two devices were detected. A real dual-GPU claim requires all of the following in one generation record:

- both CUDA indices are visible and present in the actual device map;
- both indices show non-trivial utilization/memory activity in the telemetry window;
- direct GPU-to-GPU transfer timing is recorded and compared with CPU-staged timing;
- CPU headroom stays above 24 GiB, with 26 GiB as the target;
- each GPU stays below approximately 14 GiB used for the safety profile.

## Files

- `../kaggle_h3.zip`: clean upload bundle containing this package and `kaggle_h3.ipynb` without generated results, model weights, or Python caches.
- `../kaggle_h3.ipynb`: entrypoint notebook kept beside this package; upload/import it into Kaggle and run top to bottom.
- `workflows/kaggle_h3_ref2va.json`: reference-conditioned API-format graph.
- `workflows/kaggle_h3_fl2va.json`: first/last-frame API-format graph.
- `workflows/kaggle_h3_t2va.json`: text-to-video/audio API-format graph.
- `workflows/kaggle_h3_turbo_smoke.json`: Ref2VA 360p/16:9, 5-second, 4-step adapter smoke graph using the checked-in character and scene reference images.
- `custom_nodes/kaggle_h3_adapters.py`: the auto-resolved smoke reference loader, global conditioning implementation, legacy bounded Ref2VA wrapper, just-in-time diffusion auto-loader, explicit H3 text-encoder/VAE loaders, a conditioning-complete phase barrier, `H3 Adapter Stack`, phase-aware H3 sampler, and GPU0/GPU1 audio/video VAE decode nodes.
- `custom_nodes/kaggle_h3_conditioning.py`: the separate V3 entrypoint that registers the autogrowing `Kaggle H3 Conditioning` node; it is separate because ComfyUI prioritizes legacy mappings over a same-module V3 entrypoint.
- At runtime, the bootstrap copies only the node module and adapter catalog into `ComfyUI/custom_nodes`; dependency-light helpers are installed under the private `ComfyUI/kaggle_h3_support/` package so ComfyUI does not scan them as separate custom nodes.
- `src/kaggle_h3/model_manager.py`: pinned, one-variant-at-a-time H3 diffusion download and exact inactive-checkpoint removal used by the ComfyUI loader.
- `src/kaggle_h3/ref2va.py`: shared Ref2VA seconds, 17*k+5 frame alignment, and 32-pixel canvas-preset rules.
- `src/kaggle_h3/phase_runtime.py`: phase orchestration, automatic Qwen language-layer dispatch/release, Accelerate dispatch verification, GPU/RAM monitoring, transformer release, and per-VAE device placement.
- `custom_nodes/kaggle_h3_adapter_catalog.json`: pinned built-in adapter metadata and registration surface.
- `custom_nodes.lock`: resolved ComfyUI and optional multi-GPU custom-node revisions.
- `src/kaggle_h3/`: bootstrap, ComfyUI API, backend checks/workers, workflow builder, telemetry, pipeline, and media validation.
- `requirements-h3-layer-sharded.txt`: optional Diffusers/Accelerate stack for the experimental automatic transformer-block dispatcher.
- `tests/`: dependency-light local checks.
- `results/`: created on first run; MP4s, manifests, telemetry JSONL, keyframes, and benchmark summaries go here.

Each generated workflow node carries a visible ComfyUI `_meta.title` label. The
labels identify the H3 mode, GPU assignment, dispatch barrier, adapter/sampler
step count, and final output role, so imported graphs remain understandable
without relying on their internal node IDs.

## Kaggle setup

1. Import `kaggle_h3.ipynb` from this repository into a Kaggle Notebook. The first code cell defaults to `https://github.com/celunah/kaggle_h3.git`; optionally set `GITHUB_REF` to a branch, tag, or 40-character commit. The notebook clones or updates that checkout directly under `/kaggle/working/kaggle_h3_github`, so the large Kaggle Dataset upload is not needed. It leaves model weights out of GitHub and downloads them into `/kaggle/tmp`. Set `GITHUB_REPOSITORY` to an empty string to use the older Dataset mode: it refreshes the read-only mount into `/kaggle/working/kaggle_h3` before importing the package, preserving generated `results/` files. If an older nested package folder is still mounted beside it, Dataset mode selects the uniquely named `kaggle_h3` root automatically; `/kaggle/input` itself cannot be deleted from a notebook.
2. Enable Internet only if ComfyUI, Python packages, or public Hugging Face model files need to be downloaded. Public files do not require a token. If a gated mirror is used, create a Kaggle secret named `HF_TOKEN`; the notebook reads it without printing it. The dependency cell installs the optional layer-dispatch stack by default; set `INSTALL_LAYER_SHARDED_BACKEND = False` there for a ComfyUI-only session.
3. Run the dependency and hardware cells. They print every detected CUDA device, model capacity, CPU RAM, and all backend decisions, including the explicit fallback.
4. The notebook performs the hardware/backend preflight automatically. Its direct ComfyUI frontend exposes both visible T4s, sets `KAGGLE_H3_PHASE_SHARDING=auto`, and lets the custom H3 loader/barrier apply the phase-aware two-GPU path. The loader and sampler refuse to disguise a one-GPU map as sharded; set `KAGGLE_H3_PHASE_SHARDING=off` only for the explicit fallback.
5. The startup cell starts ComfyUI, clones the pinned checkout when needed, installs its `requirements.txt`, downloads only the shared H3 text/audio/video assets into `/kaggle/tmp/minimax-h3-models`, configures ComfyUI to read that external model tree, installs the H3 adapter node, and stages the smoke references into ComfyUI's `input/` directory. The supplied workflows select those staged files directly inside `Kaggle H3 Conditioning`; no separate reference-loader nodes are required. It does not pre-download a diffusion checkpoint, construct a workflow, or queue a request. Choose `FL2VA` or `Ref2VA` in the `Kaggle H3 | Diffusion Auto Loader` node; that node removes only the other known H3 diffusion filename and streams the selected pinned checkpoint when the workflow executes. `/kaggle/tmp` is scratch storage and must be repopulated after a new session.
6. ComfyUI is started on internal port `8188` with `0.0.0.0` binding for the notebook environment. When two GPUs are in the preflight plan, the launcher explicitly sets `CUDA_VISIBLE_DEVICES=0,1` and passes `--cuda-device 0,1`; it then checks `/system_stats` and stops before loading a workflow if ComfyUI exposes fewer than two devices. The startup output must therefore show both devices and roughly 29 GiB total VRAM, rather than only `cuda:0`. If it reports one device, stop the old ComfyUI process and rerun the updated startup cell; changing environment variables cannot change an already-running process. The notebook reloads the bootstrap module when that cell is rerun, so a full kernel reset is not required unless an older ComfyUI child remains alive. The notebook then prints the local runtime address and leaves public exposure optional. To test remote access, run the optional public-tunnel cell in the entrypoint; it restarts ComfyUI with `--enable-cors-header *` for the dynamic tunnel hostname, downloads `cloudflared` into `/kaggle/tmp`, starts a temporary tunnel to `http://127.0.0.1:8188`, and prints the generated URL if Kaggle permits it. This public mode disables ComfyUI's origin protection, so anyone with the URL can access the unauthenticated instance. The phase sampler prints an observed map and per-device peak memory during each generation.
7. Build or import an H3 graph in the web interface. For Turbo, use `H3 Adapter Stack` followed by `H3 Turbo Sampler`; select the matching 4-step or 8-step adapter and sampler configuration. The node downloads its selected LoRA into `ComfyUI/models/loras/` on first use.

Use `Kaggle H3 Conditioning` between the loaders and `Kaggle H3 | Transformer
Dispatch Barrier`. It contains the prompt, `Ref2VA`/`FL2VA` mode selector,
start/end frame upload/select controls, seconds, resolution preset, aspect
ratio, and 15 mixed-media reference rows. Each row can select/upload an image,
video, audio, or `None`; an inline video's soundtrack is paired automatically.
The typed image/video/audio sockets remain available for advanced graphs, but
do not use the same slot through both a file selector and a socket. In FL2VA
mode, use start/end frames and do not select Ref2VA references. In Ref2VA mode,
references are passed to ComfyUI's native `MiniMaxH3ReferenceToVideo`; optional
start/end images are applied through the native `MiniMaxH3AddGuide` contract.
The node converts seconds to H3's aligned frame count and prints the requested
and actual duration. Its `positive` output connects to the dispatch barrier's
`conditioning` input, and its `LATENT` output connects to
`latent_image`; the supplied workflows already contain these links. The
minimum is 39 frames (1.625 seconds), and the current project safety cap is 15
seconds. A nominal 720p/16:9 request becomes 1280x704 because
H3 requires a 32-pixel canvas grid; this is expected and matches native H3
behavior.

The older `Kaggle H3 | Ref2VA Conditioning (Seconds + Presets)` node remains
available for previously saved legacy graphs, but new workflows use the global
V3 node. Its mixed-media selector rows are materialized in API-format graphs
as pairs such as `reference_0=image` and `reference_0.file=...`; the advanced
typed socket groups retain dotted names such as `ref_images.ref_image_0`.

The web interface remains the ComfyUI workflow frontend. The direct graphs
use `KaggleH3TurboSampler` even for base H3 so transformer dispatch/release is
owned by one node. Native workflows now use `KaggleH3ShardedDiffusionLoader`,
which selects and downloads one H3 diffusion variant through ComfyUI's
configured model tree, then leaves the H3 model CPU-owned until
`KaggleH3PhaseDispatch` runs after
conditioning. That barrier applies and verifies the intact-block map across
GPU0/GPU1, while the sampler suppresses ComfyUI's subsequent GPU0-only reload.
`KaggleH3TextEncoderLoader` places the text encoder on GPU0 for conditioning;
it is evicted before transformer dispatch. Native H3 reference encoding can
temporarily call the video/audio VAEs before the barrier. After denoising,
`KaggleH3AudioVAEDecode` targets GPU0 and `KaggleH3VAEDecode` targets GPU1.
The separate Diffusers worker remains
available for the independent layer-sharding backend and still requires its
own real-generation verification.

The notebook uses `project_root / "results"`, so the expected artifact paths are:

```text
kaggle_h3/results/<run_id>_01.mp4
kaggle_h3/results/<run_id>.json
kaggle_h3/results/<run_id>/telemetry_attempt_01.jsonl
```

## Model and ComfyUI files

The default ComfyUI profile pins the `v0.34.0` ComfyUI release and downloads the shared video/audio VAE and Qwen text encoder from `Comfy-Org/MiniMax-H3`. The files are stored under `/kaggle/tmp/minimax-h3-models/models` so the 20 GiB persistent `/kaggle/working` output limit is not consumed. The diffusion checkpoint is intentionally deferred to the ComfyUI loader. Only one H3 diffusion checkpoint is present at a time; switching the loader selection removes the other exact known variant and precision files before downloading the replacement.

`Kaggle H3 | Diffusion Auto Loader` exposes `precision=auto`, `int8_convrot`, or `fp8_scaled`. INT8 ConvRot is the default performance selection, including the two-T4 Kaggle profile. FP8-scaled remains available as an explicit lower-VRAM option, and the environment variable `KAGGLE_H3_DIFFUSION_PRECISION=fp8_scaled` can override `auto`. The corresponding files are:

- `diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors` or `minimax_h3_ref2va_pruned_int8_convrot.safetensors` for Ref2VA;
- `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors` or `minimax_h3_fl2va_pruned_int8_convrot.safetensors` for FL2VA/T2VA;
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`;
- `vae/minimax_h3_video_vae_fp16.safetensors`;
- `vae/minimax_h3_audio_vae_fp32.safetensors`.

The T4 compatibility of a particular quantized kernel build is an empirical Kaggle check, not something this package declares in advance. The official model README documents alternative BF16, INT8 ConvRot, FP8-scaled, and NVFP4 assets; do not switch assets without recording the change in the manifest. `custom_nodes.lock` records the ComfyUI tag and the observed optional multi-GPU node commit used by the evaluator.

### H3 adapter stack and Turbo

The custom node exposes three ordered adapter slots, each with an independent
strength from `0.0` to `2.0`. Selectors come from the metadata catalog rather
than a hardcoded three-name list. The built-in entries are the official
LightX2V ComfyUI safetensors pinned to HF revision
`3ec17a324ced54151364f24f8b5fb6bf7e26414f`:

- FL2VA/T2VA Turbo 4-step v1.0 768p: 4 NFE, Euler, video/audio shifts `6/3`.
- FL2VA/T2VA Turbo 8-step v1.0 768p: 8 NFE, Euler, video/audio shifts `6/3`.
- Ref2VA Turbo 4-step v0.1: 4 NFE, Euler, video/audio shifts `12/3`.
- Ref2VA Turbo 8-step v1.0 768p: 8 NFE, Euler, video/audio shifts `6/3`.

`H3 Adapter Stack` detects the H3 variant from the incoming Comfy model,
resolves T2VA/FL2VA/Ref2VA from H3 conditioning metadata when the optional
mode input is `auto`, validates every selected adapter before applying any,
and applies them in slot order through ComfyUI's clone-based
`load_lora_for_models` path. The source checkpoint is never rewritten, LoRA
state is loaded on CPU and released after patching, and ComfyUI's model
offloading remains responsible for GPU residency.

Because ComfyUI `UNETLoader` can discard the source filename and FL2VA/Ref2VA
share an architecture, the node also exposes an advanced `model_variant`
override. Generated workflows set it from the selected H3 mode; manual graphs
should leave it at `auto` only when their loader carries equivalent metadata.
An ambiguous bare model fails with a compatibility error rather than guessing.

Turbo is not a label-only option. The stack outputs a typed
`H3_RUNTIME_CONFIG`; `H3 Turbo Sampler` consumes that output, applies the
matching H3 AV sigma shifts, creates the matching sigma count, and rejects a
wrong sampler or step count. A checkpoint explicitly marked as fused Turbo,
or a model that already records the same adapter, skips duplicate injection
and reports the reason. Unknown or incompatible adapters fail before download
or patching.

To register another adapter, append an object to
`custom_nodes/kaggle_h3_adapter_catalog.json` with at least:

```json
{
  "adapter_id": "mystic_motion_v1",
  "name": "Mystic motion v1",
  "repository": "owner/repository",
  "filename": "mystic_motion_v1.safetensors",
  "revision": "<40-character-HF-commit>",
  "model_variants": ["fl2va"],
  "conditioning_modes": ["T2VA", "FL2VA"],
  "recommended_strength": 1.0,
  "supported_steps": [4, 8],
  "fused": false,
  "license": "upstream license",
  "attribution": "upstream author",
  "kind": "motion",
  "schedules": {}
}
```

Use a real upstream repository, safe `.safetensors` filename, explicit
revision, supported variant/mode declarations, and license attribution. The
node downloads into the configured LoRA folder once and writes a small
`.h3.json` provenance sidecar. Never register an adapter merely because a
file happens to load: metadata and matched Comfy model patches are required.

## Backend-specific notes

### SGLang

The worker builder exposes the official two-GPU memory-mode shape (`--num-gpus 2`, `--tp-size 2`, Ulysses degree, encoder parallel, and DiT/text-encoder/VAE layerwise offload). Install the optional worker with `pip install "sglang[diffusion]" --prerelease=allow` when the Kaggle image and Python version support it. It uses the local OpenAI-compatible `/v1/videos` submit/status/content API. The official two-card H3 recipe was validated on 2x RTX 5090 32 GiB with roughly 384 GiB host RAM, so 2x T4 plus ordinary Kaggle host RAM is reported as unverified/capacity-blocked until a real run proves otherwise.

### Diffusers

The evaluator reports the generic Accelerate `balanced` probe, the automatic layer-sharded worker, and the explicit component split separately. Install `requirements-h3-layer-sharded.txt` when testing the new worker; the base requirements intentionally avoid pulling that optional stack into every ComfyUI-only session. The layer worker discovers the largest indexed H3 transformer block sequence, constructs a meta model to size it without a full CPU duplicate, creates a memory-budgeted hierarchical map, and loads complete residual blocks directly across every visible GPU. For the current 50-block H3 model it tries the proposed `blocks[0:23]` on GPU 0 and `blocks[23:50]` on GPU 1, with input projections/token refinement and the quantized output/final path on GPU 0. Other indexed block counts receive a balanced contiguous split. CPU RAM plus `/kaggle/tmp` disk are overflow tiers. It inspects the resulting parameter devices after loading and refuses to continue if the map collapses to one GPU. The conditioner is run first and released before denoiser materialization, avoiding a resident duplicate Qwen copy during denoising. The current worker does not claim an internal Qwen 21/29 layer split or a split VAE forward; it uses ComponentsManager for the temporary conditioner/VAE stages until those paths can be verified against the installed H3 implementation. Its live safety gate uses a 13 GiB/card residency budget, a 14 GiB process ceiling, and 26 GiB available host RAM; these are attempt thresholds, not claims that H3 is guaranteed to fit. The explicit component split remains available for comparison, while a generic `device_map="balanced"` is not treated as sufficient evidence by itself.

### ComfyUI sequence-parallel custom node

The workflow builder can emit `MiniMaxH3SPUNETLoader` and `MiniMaxH3SPVAEDecode` nodes with `world_size=2`. The optional node sequence-shards the computation but replicates the DiT weights, so its published footprint is not suitable for the default 15 GiB T4 safety budget. Installation is intentionally not automatic; use the node's own pinned instructions, then rerun the evaluator and benchmark.

### Native ComfyUI phase path and fallback

The explicit native path uses `KaggleH3ShardedDiffusionLoader` to keep the H3
model CPU-owned while `KaggleH3TextEncoderLoader` dispatches the Qwen language
layers contiguously across GPU0 and GPU1 during conditioning. The existing
workflow `device_id=0` means the primary boundary device; it does not disable
text-layer sharding. `KaggleH3PhaseDispatch` is a graph barrier: it runs after the H3
conditioning node, discovers the largest indexed block sequence, and applies
the 13 GiB/card residency and 26 GiB CPU-headroom map across both GPUs. For ComfyUI's
INT8 ConvRot H3 weights it does not call Accelerate's generic tensor hook:
that hook reconstructs `comfy_kitchen.QuantizedTensor` with an unsupported
`requires_grad` argument. Instead, GPU-budgeted blocks are moved through
ComfyUI's native quantized `_apply` path, CPU/disk-tier blocks remain
CPU-owned, and activation arguments are routed to the assigned GPU. The
quantized final/output path stays on the primary GPU so its internal dtype
conversion never crosses a live device boundary. The router uses deliberate
CUDA barriers at block outputs, activation transfers, final-layer handoff,
and ComfyUI's `cast_to`/`cast_to_gathered` quantized-weight copy boundaries;
copies retain ComfyUI's native stream/allocation path but force
`non_blocking=False` and synchronize the producer and consumer devices. The
launcher also disables `cudaMallocAsync` for this phase to avoid
allocator/stream races. The report records both the physical parameter map
and the actual execution map.
Finite-value diagnostics are opt-in and disabled by default because a
recursive `torch.isfinite` scan can synchronize and scan every dispatched
tensor. Set `KAGGLE_H3_FP_DIAGNOSTICS=1` before starting ComfyUI to enable
the fail-fast checks. This setting does not disable the explicit CUDA
synchronization, quantized-copy barriers, transfer barriers, or H3-safe
`res_multistep` history ownership fix.

Per-step diffusion telemetry is separately opt-in. Set
`KAGGLE_H3_DIFFUSION_TELEMETRY=1` to emit structured JSON records for each
`res_multistep` or Turbo `euler` step at `input`, `model_output`, `denoised`, `history`, and
`updated_latent`. Every record includes sigma/timestep, min, max, mean, std,
finite count, dtype, device, and tensor path. The model API exposes the
denoised prediction directly, so `denoised` is recorded as an explicit alias
of `model_output`; no extra transformation is performed. To persist a
machine-readable JSONL stream, also set
`KAGGLE_H3_DIFFUSION_TELEMETRY_PATH=/path/to/diffusion_telemetry.jsonl`.
Each floating-point record also includes its tensor shape and a
`sha256_raw_contiguous` checksum of the tensor bytes. The same telemetry
stream records VAE boundaries (`vae_video_in_source`, `vae_video_in`,
`vae_video_out`, `vae_video_output`, and the corresponding audio stages), so a
sampler latent can be compared with the exact values entering and leaving each
decoder. Checksums intentionally copy each recorded tensor to CPU after
synchronizing its CUDA device; this is diagnostic-only and should not be used
for production timing. Euler has no retained second-order history, so its
history record is explicitly marked absent.
The sampler preserves that map by filtering the already-dispatched H3 model
out of ComfyUI's subsequent global `load_models_gpu()` call. A lightweight
monitor logs peak used/allocated/reserved memory for both GPUs and minimum CPU
RAM during the phase. After the sample it removes dispatch hooks, asks the
ComfyUI patcher to return the model to CPU, and clears the cache before VAE
decode. The text-encoder and VAE phases have matching cleanup paths, including
failure cleanup, so a sampler or decode exception does not intentionally retain
their GPU pages.
The dedicated `Kaggle H3 | H3 Turbo Sampler` exposes `synchronize_mode`, which
defaults to `full` so existing workflows retain the proven behavior. `full`
keeps every device-wide block and per-step barrier. `safe` keeps every block,
transfer, quantized-copy, and cleanup barrier but limits the redundant
post-model sampler barrier to the primary GPU. `fast` keeps cross-device
activation barriers, synchronous quantized-weight copies, final-layer
handoff, and phase cleanup, while skipping same-device activation waits and
per-step full-device waits; it is the experimental performance mode and may
re-expose an asynchronous H3 kernel issue. The selected policy is recorded in
`H3_RUNTIME_CONFIG` and can be changed directly on the sampler node.
The explicit VAE loader/decoder nodes then target video at GPU1 and audio at
GPU0, while retaining CPU as the offload device. The dedicated sampler now
returns separate `video_latent` and `audio_latent` outputs. The H3 final AV
output is kept on GPU0; video is transferred directly GPU0 -> GPU1 only when
the video VAE consumes it, audio stays on GPU0, and decoded frames/waveform
are moved to CPU before `CreateVideo`/`SaveVideo`. This avoids CPU staging of a
packed AV NestedTensor and avoids retaining the unused stream in each decoder.
The audio decoder also reports and replaces non-finite VAE samples before the
AAC mux boundary, preventing PyAV's opaque `avcodec_send_frame()` failure.

This remains experimental because ComfyUI's ordinary `ModelPatcher` also
manages model loading. If Accelerate cannot coexist with the installed
ComfyUI build, the barrier/sampler raises a clear dispatch error rather than
silently using only GPU0. Setting `KAGGLE_H3_PHASE_SHARDING=off` selects the
one-GPU fallback, whose launch uses `--lowvram`, `--cpu-vae`, and CPU offload;
that fallback explicitly reports the secondary GPU as unused. The video
decoder still applies the dtype-safe, offload-aware cast without changing
checkpoint files.

`start_comfyui()` tees ComfyUI stdout/stderr to both `results/comfyui.log` and
the notebook console, so `[Kaggle H3][diffusion]` telemetry is visible live
while remaining available for later inspection. Set
`KAGGLE_H3_STREAM_COMFY_LOGS=0` before launch to disable console streaming
while keeping the file log.

The adapter node is currently a ComfyUI backend feature. The SGLang and
Diffusers worker scaffolds do not expose a compatible H3 LoRA contract, so a
request that selects Turbo/adapters is blocked on those workers instead of
silently generating with the base model. This keeps the two-T4 backend
investigation honest while retaining the true multi-GPU routes for requests
that they can actually execute.

## Validation and limitations

Local checks validate Python syntax, workflow shape, mode selection, H3 frame alignment, and manifest structure. `ffprobe` validates output streams, duration, frame rate, resolution, and audio presence when installed. Keyframes are extracted for review, but identity continuity, reference fidelity, and the Soul Spirit transition are intentionally reported as uncertain until a human or vision-model review is performed.

The adapter tests additionally validate pinned metadata, cache reuse, safe
download errors, variant/mode/step rejection, ordered strengths, fused-Turbo
deduplication, Comfy clone-based patching, runtime configuration, node
registration, and the low-resolution Turbo smoke graph. A local machine
without the Kaggle H3 weights cannot complete the real MP4 smoke generation;
that must be run in Kaggle and its manifest must be used for the final
injection/telemetry claim.

The local machine cannot manufacture Kaggle's two T4 result. Therefore the repository's local dry run is evidence that the orchestration is wired and honest, not evidence that H3 fits on two T4s. The final Kaggle manifest is the source of truth for: whether both GPUs were used for one H3 generation, which layers/components each device held, whether CPU RAM was used, peak memory on each T4, generation time versus the one-GPU baseline, and any backend limitations.
