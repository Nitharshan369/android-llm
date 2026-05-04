# Android On-Device LLM — SmolLM2-360M + llama.cpp

Runs a quantized LLM entirely on an Android device using native C++ inference. No internet. No API. The model loads from internal storage and generates text through llama.cpp via JNI.

A separate `Quantifier/` module runs the original FP16 model on desktop to produce reference outputs for validating what the quantized mobile version actually does.

---

## Model

| Property | Value |
|---|---|
| Model | `HuggingFaceTB/smollm2-360M-Instruct` |
| Desktop format | FP16 — HuggingFace Transformers |
| Mobile format | GGUF |
| Quantization | Q4_K_M |
| Inference engine | llama.cpp |

---

## What this project actually does

**Android side:** Takes a text prompt from the UI, passes it through JNI into a C++ inference loop powered by llama.cpp, and returns generated tokens back to the screen. The GGUF model runs fully on the device CPU — no network calls at any point.

**Quantifier side:** Runs the same model in FP16 precision on desktop using HuggingFace Transformers with deterministic sampling (`do_sample=False`). This gives a stable baseline to compare against what the quantized mobile version produces for the same prompt.

---

## Android inference pipeline

```
┌─────────────────────────────┐
│      Android UI             │
│   (user types a prompt)     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      MainActivity.kt        │
│   handles input + display   │
└──────────────┬──────────────┘
               │  JNI call
               ▼
┌─────────────────────────────┐
│      native-lib.cpp         │
│   loads model, tokenizes,   │
│   runs generation loop      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         llama.cpp           │
│   CPU inference, sampling,  │
│   decoding                  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   GGUF model (Q4_K_M)       │
│   smollm2-360m-instruct     │
│   on-device storage         │
└──────────────┬──────────────┘
               │
               ▼
       generated text → UI
```

### Component responsibilities

**`MainActivity.kt`** — entry point of the app. Handles all user interaction, sends the prompt to the native layer, and renders the returned text.

**`native-lib.cpp`** — JNI bridge. Loads the GGUF model file, tokenizes the input, runs the llama.cpp generation loop, and returns decoded output text to Kotlin.

**`llama.cpp` (git submodule)** — the inference engine. Handles the actual CPU-based forward pass, token sampling, and decoding.

---

## Quantifier pipeline

```
┌─────────────────────────────┐
│       prompt input          │
│  "Explain python in         │
│   programming"              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   run_model_smollm.py       │
│   HuggingFace Transformers  │
│   do_sample=False           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  smollm2-360M-Instruct      │
│  FP16 — full precision      │
│  desktop CPU/GPU            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      fp_output.txt          │
│  FP16 reference output      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    quantify_models.py       │
│  loads fp_output.txt and    │
│  q4_output.txt, computes    │
│  comparison metrics         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│            metrics output               │
│  Similarity Score    (SequenceMatcher)  │
│  Jaccard Similarity  (token sets)       │
│  Token Accuracy      (token overlap)    │
│  Length Difference   (token count)      │
│  FP / Q4 Unique Tokens                 │
└─────────────────────────────────────────┘
```

`q4_output.txt` holds the Android model's output for the same prompt and is fed into `quantify_models.py` alongside `fp_output.txt` for comparison.

### Quantifier files

| File | Purpose |
|---|---|
| `run_model_smollm.py` | Runs the FP16 model, writes output to `fp_output.txt` |
| `quantify_models.py` | Loads both output files, computes all comparison metrics |
| `fp_output.txt` | FP16 model output — the reference |
| `q4_output.txt` | Q4_K_M model output — from the Android run |
| `requirements.txt` | `transformers`, `torch`, `sentencepiece`, `accelerate` |

### Metrics computed by `quantify_models.py`

- **Similarity score** — character-level sequence similarity via `SequenceMatcher`
- **Jaccard similarity** — overlap of unique token sets between FP16 and Q4 outputs
- **Token accuracy** — proportion of tokens from FP16 output present in Q4 output
- **Length difference** — absolute difference in token count between the two outputs
- **Unique token counts** — vocabulary size of each output independently

---

## Repository structure

```
android-llm/
│
├── app/
│   └── src/main/
│       ├── cpp/
│       │   ├── llama.cpp/          ← git submodule
│       │   ├── native-lib.cpp      ← JNI bridge
│       │   └── CMakeLists.txt
│       ├── java/.../MainActivity.kt
│       └── res/
│
├── Quantifier/
│   ├── run_model_smollm.py         ← runs FP16 model
│   ├── quantify_models.py          ← computes comparison metrics
│   ├── fp_output.txt               ← FP16 reference output
│   ├── q4_output.txt               ← Q4_K_M model output
│   └── requirements.txt
│
├── .gitmodules
├── LICENSE
└── README.md
```

---

## Setup

### Clone

```bash
git clone --recurse-submodules https://github.com/Nitharshan369/android-llm.git
cd android-llm
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## Running the Android app

**Requirements**
- Android Studio (latest stable)
- Android SDK — API 36
- Android NDK + CMake
- Physical device recommended — emulator inference will be very slow

**Steps**

1. Open the project in Android Studio
2. Let Gradle sync complete
3. Build and run on device

The app will invoke `native-lib.cpp` via JNI on launch. The inference result appears in the UI.

---

## Model setup

The native layer expects the model at this path on-device:

```
/data/data/com.example.firstapplication/files/smollm2-360m-instruct-q4_k_m.gguf
```

**Push with ADB (fastest for development)**

```bash
adb push smollm2-360m-instruct-q4_k_m.gguf \
  /data/data/com.example.firstapplication/files/
```

**Alternatives**
- Bundle the model in `app/src/main/assets/` and copy to `filesDir` at runtime
- Download from HuggingFace Hub on first launch and write to `filesDir`

If the file is missing or the path doesn't match, inference will fail silently.

---

## Running the Quantifier

**Install dependencies**

```bash
pip install -r Quantifier/requirements.txt
```

**Step 1 — run the FP16 model**

```bash
python Quantifier/run_model_smollm.py
```

Loads `smollm2-360M-Instruct` in FP16, runs generation with `do_sample=False` on the prompt, and writes the output to `Quantifier/fp_output.txt`.

**Step 2 — run the comparison**

```bash
python Quantifier/quantify_models.py
```

Reads `fp_output.txt` and `q4_output.txt`, cleans both, and prints:

```
===== RESULTS =====
Similarity Score   : 0.XXXX
Jaccard Similarity : 0.XXXX
Token Accuracy     : 0.XXXX
Length Difference  : XX
FP Unique Tokens   : XX
Q4 Unique Tokens   : XX
```

`q4_output.txt` contains the Android model's output for the same prompt — update it with whatever your device generates before running the comparison.

---

## Observations

- Quantized model preserves the overall meaning of responses
- Output may be slightly shorter or less detailed than FP16
- Memory footprint is significantly lower than FP16
- Inference is feasible on mobile CPU — slow, but functional

---

## Limitations

- Model path is hardcoded in native code
- No automatic model download or setup
- Evaluation metrics are qualitative + lightweight (no BLEU, no embeddings)
- UI is minimal — single prompt/response, no chat history
- CPU-only — no GPU or NPU acceleration

---

## Tech stack

| Layer | Technology |
|---|---|
| Android UI | Kotlin |
| Native inference | C++ — NDK + JNI |
| Inference engine | llama.cpp |
| Model format | GGUF (Q4_K_M) |
| Desktop evaluation | Python + HuggingFace Transformers |
| Evaluation metrics | `quantify_models.py` — SequenceMatcher, Jaccard, token overlap |

---

## Full pipeline

```
FP16 model
    │
    ├─→ quantize → GGUF Q4_K_M → llama.cpp → JNI → Android execution
    │
    └─→ Python (Transformers) → FP16 reference output → compare
```
