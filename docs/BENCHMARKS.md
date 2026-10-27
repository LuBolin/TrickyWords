# Model precision benchmarks

Measured on 27 October 2026, Singapore time. Both GPUs belong to one Windows laptop with an Intel Core i9-13900HX and 64 GiB system RAM. These results do not certify a physical 16 GB or 8 GB laptop.

The release selects 8-bit constant matrix and embedding weights. Other weights and activations retain FP16, with original FP32 attention. Both precision modes use Mobile-O-0.5B revision `92b5ef6a90a770c631153c3c32be0bbe049e2a8d`, ONNX Runtime Web 1.30.0, 512 × 512 output, eight diffusion steps and guidance 1.5.

## Speed

Times include text conditioning, drawing, image decoding, background removal and collision polygons. They exclude one-time setup.

| GPU and browser | Runs per mode | 4-bit median | 8-bit median |
| --- | ---: | ---: | ---: |
| RTX 4060 Laptop, in-app Chromium 156 | 18 | 4.83 s | 4.27 s |
| Intel UHD, isolated headless Edge 156 | 6 | 34.84 s | 34.81 s |

RTX cases used slipper, teapot and penguin at seeds 12345, 23456 and 34567, twice per mode. Intel cases used the same three words at seed 12345, twice per mode. All 48 runs completed with cutouts and collision polygons. Setup compared GPU and CPU background runtimes and selected native WebGPU JSPI for both GPUs.

The Intel test selected its DXGI adapter for the whole isolated browser, then verified WebGPU vendor `intel`, architecture `gen-12lp`, FP16 support and the Intel UHD hardware renderer. It used driver 31.0.101.5187 with the browser sandbox enabled. The RTX driver was 32.0.16.2101.

Intel object times ranged from 34.64 to 34.89 seconds for 4-bit and 34.58 to 34.95 seconds for 8-bit. Its first setup took about three minutes in each mode. The 8-bit stage medians were 0.30 s for text, 16.18 s for drawing, 12.08 s for decoding, 6.13 s for background removal and 0.07 s for polygons.

The Intel timing difference is negligible. The lower RTX 8-bit median is an observation, not a proven precision-only speedup: its unchanged background stage also became faster between passes. Other applications, power state and driver caches were not fully controlled.

## Files and memory

| Measurement | 4-bit | 8-bit |
| --- | ---: | ---: |
| Complete model pack, including CPU background fallback | 1.075 GB | 1.677 GB |
| Active native model files | 0.881 GB | 1.484 GB |
| Intel private browser RAM peak during generation | 6.13 GB | 6.71 GB |
| Intel shared GPU allocation peak during generation | 3.81 GB | 4.36 GB |

GB values are decimal. The Intel memory sampler tracked only the fresh browser process tree, using Windows private working set and per-process GPU allocation counters at a median interval of 0.53 seconds. These are absolute sampled peaks and may miss short spikes. Private RAM and shared GPU allocations can overlap; adding them does not give total physical RAM use. System RAM capacity was not reduced to simulate another computer.

## Image fidelity

Three same-seed examples were compared against FP16 outputs. Average absolute pixel error was 0.06150 for 4-bit and 0.01423 for 8-bit, a 76.9% reduction. This measures closeness to FP16 in those examples, not a percentage improvement in semantic accuracy or player preference.

Intel UHD can run this pipeline, but its roughly 35-second object delay exceeds a five-second target. Faster integrated GPUs need their own measurements. Direct file opening, macOS and the final Firefox package remain unverified.
