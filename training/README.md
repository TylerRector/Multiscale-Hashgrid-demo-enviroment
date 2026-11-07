# Training

Python 3.11+; CPU training supported. From the demo root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r training/requirements.txt
.venv/bin/python training/run.py
./build/hashgrid_demo --model training/runs/demo_training/export
```

The script generates traced supervision, gathers/denoises targets, trains,
validates, and exports the GLSL shader and weights. `fix_p5` uses the bundled
model's training settings: 8,000 iterations, seed 1337. New results may differ
across library versions. `--profile micro --run smoke` runs a smaller pipeline;
its model has a different architecture from the bundled model.

Generated data stays in `training/runs/`; the bundled model is never overwritten.
The original training tables used schema version 2; new runs generate version 3.

A fresh micro tooling check completed 2,000 steps and passed export parity.
Its held-out PSNR was 8.97 dB versus a 9 dB target; this is a plumbing check,
not a replacement for the bundled model.

To save real training snapshots and generate the GIFs:

```sh
.venv/bin/python training/run.py --run learning_gif --snapshot-every 250
.venv/bin/python training/render_media.py --run learning_gif
```

The offline capture tool builds a temporary executable using the demo renderer.
Its camera path is checked against walls, floor, ceiling, the ball, and the pillar.
The training animation uses a fixed camera/light/exposure and actual saved weights;
only its small yellow counter is overlaid on the current model render.
