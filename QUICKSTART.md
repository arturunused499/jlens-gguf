# J‑Lens Quickstart

A friendly, hands‑on tour of **jlens‑gguf** — see and steer what a GGUF model
is "thinking" at every layer, live, through llama.cpp. No PyTorch, no cloud.

If you can run `llama-server`, you can run this.

---

## 1. What this is (in one picture)

```
   your browser (the visualizer)                 your chat app (optional)
            │                                            │  OpenAI API
            ▼                                            ▼
      jlens bridge  ──────────────▶  jlens-server  ◀───────────────
     (numpy lens math)              (llama.cpp + a live steering hook)
                                          │
                                     your GGUF model
```

- **jlens‑server** loads your GGUF with llama.cpp and can read *and edit* the
  residual stream at every layer while the model runs.
- **The bridge + web UI** turn those activations into the classic J‑Lens
  layer × position heatmap and let you steer, swap, and ablate concepts.
- It's also a **drop‑in `llama-server`** — point any app at it and steer the
  tokens *that app* generates.

---

## 2. Install (once)

```bash
git clone --recursive <repo-url> jlens-gguf && cd jlens-gguf

native/build.sh                       # builds llama.cpp (submodule) + jlens-server
python3 -m venv .venv && .venv/bin/pip install -e .
```

You need `gcc`, `cmake`, `python3`. That's it. (No GPU required — CPU works;
add `-ngl N` later if you have one.)

Grab a model if you don't have one, e.g.:

```bash
mkdir -p models && curl -L -o models/qwen2.5-1.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf
```

---

## 3. Sixty seconds to your first lens view

```bash
python -m jlens_gguf quickstart models/qwen2.5-1.5b-instruct-q8_0.gguf
```

This starts everything and opens `http://127.0.0.1:8090`. Type a prompt, press
**Run**. You'll see a grid: **columns are token positions, rows are layers**,
and each cell shows the top word the lens reads out of that activation.

> Already running a `llama-server`? Just do
> `python -m jlens_gguf quickstart --llama-server http://127.0.0.1:8080`
> and it inspects the same model.

For sharper early‑layer readouts, fit a lens first (a few minutes on CPU):

```bash
python -m jlens_gguf fit --model models/qwen2.5-1.5b-instruct-q8_0.gguf \
    --corpus wikitext:100 -o lenses/qwen1.5b.gguf
python -m jlens_gguf quickstart models/qwen2.5-1.5b-instruct-q8_0.gguf \
    --lens lenses/qwen1.5b.gguf
```

Without a lens you get the classic **logit lens** (still useful); with one you
get the **Jacobian lens** the paper describes.

---

## 4. Reading the view

| Panel | What it shows |
|---|---|
| **Heatmap** (center‑left) | top word per (position, layer). Hover for the full top‑10; the little red number is the word's rank. |
| **By Layer** | full readout down the stack at the selected position. |
| **By Pos** | readout across positions at the selected layer. |
| **Cell readout** (right) | top‑40 words + probabilities at the selected cell. |

**Try this:**
- Click any cell → it selects that (position, layer) and **pins** its word.
- Pinned words get a **rank‑vs‑layer chart** and a **rank heatmap** over the
  whole grid — you can literally watch a concept rise through the layers.
- Hold **Shift** and move the mouse to scrub; arrow keys move the selection.

---

## 5. The fun part — steering, swapping, ablating

Three buttons on the interventions bar. Each opens a token search; pick a word,
choose layers and positions, and the view re‑runs live. Cells whose top word
changed vs. the un‑steered baseline get an **orange corner marker**.

- **steer** `h += α·v̂ₜ` — *summon* a concept (positive α) or *suppress* it
  (negative α). Start around α = 2–4 over a mid‑layer band.
- **swap** — exchange two concepts' lens coordinates (the paper's concept
  patch). Best on a **single layer**.
- **ablate** — project a concept's direction out of the stream.

Then hit **generate** (right panel) to see the steered continuation next to the
baseline.

### Worked example (the two‑hop "boot" fact)

Prompt:

```
Fact: The capital of Japan is Tokyo.
Fact: The currency used in the country shaped like a boot is
```

Qwen‑1.5B answers *"the Japanese yen"* — it wrongly binds "currency" to the
salient **Japan** instead of the boot‑shaped country. In the heatmap you can
see ` yen` collapse to rank 0 in the last layers while ` euro` hovers at
rank 60–500 in the middle "workspace" layers — the runner‑up hypothesis.

Now add two **ablate** interventions (` yen` and ` Japanese`, layers 14–26) and
generate:

> **baseline:** `the Japanese yen.`  →  **ablated:** `the Euro.`

You just corrected a faulty chain of reasoning at the concept level — live, on a
quantized model.

---

## 6. Steer a real app (backend mode)

`jlens-server` speaks the OpenAI API and takes `llama-server`'s flags, so it's a
drop‑in backend. However you launch llama‑server, swap the binary:

```bash
native/jlens-server -m models/qwen2.5-1.5b-instruct-q8_0.gguf -c 8192 -ngl 99 \
    --host 0.0.0.0 --port 8080
```

Point your app's base URL at `http://<host>:8080/v1` (Open WebUI, SillyTavern,
an agent framework, your own `openai` client — anything that talks to
llama‑server). Then in the visualizer's **Live backend** panel:

1. Build some intervention chips.
2. **push interventions** → they now apply to *every* completion your app
   requests.
3. **load last chat** → pull the app's most recent turn into the heatmap and
   see the readouts behind what it said.
4. **clear** → back to a normal backend.

```python
# your app, unchanged — just a different base_url
import openai
client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")
client.chat.completions.create(model="m", messages=[
    {"role": "user", "content": "The currency of the country shaped like a boot is the"}])
# with " Euro" ablated live: "…is the Indonesian rupiah" instead of "…the Euro"
```

---

## 7. Cheat sheet

```bash
# visualize (autostarts the sidecar)
python -m jlens_gguf serve   --model M.gguf [--lens L.gguf]

# easiest launch (opens browser; can read a running llama-server)
python -m jlens_gguf quickstart [M.gguf | --llama-server URL] [--lens L.gguf]

# fit a lens (ridge regression; --gram-dtype float32 for big models)
python -m jlens_gguf fit --model M.gguf --corpus wikitext:100 -o L.gguf
python -m jlens_gguf fit --model M.gguf --corpus my_text.txt --layers 8,9,10 -o band.gguf

# convert the paper's exact PyTorch lens (no torch needed)
python -m jlens_gguf convert-pt lens.pt lens.gguf

# steer an app: run jlens-server as a drop-in llama-server
native/jlens-server -m M.gguf -c 8192 -ngl 99 --host 0.0.0.0 --port 8080

# inspect a lens file
python -m jlens_gguf inspect L.gguf
```

**UI keys:** click = select+pin · Shift+hover = scrub · ←→ position · ↑↓ layer ·
the `lens` checkbox toggles fitted‑lens vs raw logit‑lens.

---

## 8. Does it work on *my* model?

- **Any GGUF llama.cpp can load** — dense or **Mixture‑of‑Experts** (Qwen3‑MoE,
  Mixtral, DeepSeek, OLMoE, …). The lens only touches the `d_model`‑wide
  residual stream, so MoE routing/expert count doesn't change anything. If
  `/props` shows `l_out_ok: true`, you're good (it's true for all mainstream
  decoder architectures).
- **Quantized models are fine** — readout weights are dequantized to fp32 for
  the math; readouts match llama.cpp's own logits to ~1e‑3 even at Q4.
- **Big models:** fitting memory scales as `n_layers × d_model²` (the lens is
  independent of expert count). A ~200–400B MoE fits on a machine sized to run
  it; use `--gram-dtype float32` and/or fit a **band** of layers with
  `--layers` in a few passes and `merge`. The `fit` command prints its
  estimated footprint up front. The main interactive cost is the readout grid
  (`positions × d_model × vocab`); use a layer **stride** or shorter prompts on
  very large models.

---

## 9. If something's off

- **`l_out_ok: false`** in `/props` → that architecture doesn't expose the
  residual tensors the lens needs (rare). Everything else refuses loudly rather
  than guessing.
- **Grid feels slow** on a huge model → raise the layer stride, use a shorter
  prompt, or fit/inspect fewer layers.
- **Steered output degenerates** ("Italy Italy Italy…") → lower α or use fewer
  layers; strong steering over many layers compounds.
- **Model won't load** → it's a plain llama.cpp load; the same `-c`, `-ngl`,
  `-b` flags you'd give `llama-server` apply here.

Full reference, architecture notes, and the API are in
[`README.md`](README.md).
