# open-jev

`open-jev` is a browser-focused TypeScript library for **typed decisions**: one piece of text (the _state_) plus any number of typed questions go in, and one forward pass returns a calibrated probability distribution per question. Nothing is generated, so an answer is always one of the options you provided.

It runs open reproductions of the _shape_ of TypeSafe AI's [Jev "System One" model](https://typesafe.ai/blog/introducing-system-one-models-and-jev) via [`@huggingface/transformers` (Transformers.js)](https://huggingface.co/docs/transformers.js/en/index). Everything happens on-device: WebGPU when available, WebAssembly otherwise.

## Models

| Alias            | Repo                                                                                                                  | Base             | Weights (q4f16 / q4) | Notes                                                                                                                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kev-0.6b`       | [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX)                                   | Qwen3-0.6B-Base  | 0.34 GB / 0.38 GB    | Default. Small and fast. 8192-token context.                                                                                                                                           |
| `kev-4b`         | [onnx-community/kev-4b-ONNX](https://huggingface.co/onnx-community/kev-4b-ONNX)                                       | Qwen3-4B-Base    | 2.3 GB / 2.5 GB      | Most accurate. 8192-token context. Needs a capable GPU.                                                                                                                                |
| `open-jev`       | [onnx-community/open-jev-deberta-v3-large-ONNX](https://huggingface.co/onnx-community/open-jev-deberta-v3-large-ONNX) | DeBERTa-v3-large | 0.35 GB / 0.48 GB    | Also ships `fp16` (0.88 GB) and `fp32` (1.75 GB). 512-token context.                                                                                                                   |
| `gliner2-decide` | [onnx-community/GLiNER2.5-Decide-ONNX](https://huggingface.co/onnx-community/GLiNER2.5-Decide-ONNX)                   | DeBERTa-v3-large | 0.52 GB / 0.89 GB    | Fastino's [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), trained on 17 operational domains. Also ships `fp16` (0.87 GB) and `fp32` (1.74 GB). 512-token context. |

Pass the alias as `model`, or any Hugging Face repo id whose `config.json` carries an `open_jev`, `kev` or `gliner2` section. The encoding family is detected from that config.

## Install

```bash
npm install open-jev @huggingface/transformers
```

`@huggingface/transformers` is a peer dependency so your app and this library share one copy of Transformers.js and its model cache.

## Quick start

```ts
import { OpenJev, choice, score, noul } from "open-jev";

const info = await OpenJev.info({ dtype: "q4f16" });
console.log(info.isCached, info.downloadSize, info.device, info.dtype);

const jev = await OpenJev.load({
  model: "kev-0.6b", // default; or "kev-4b", "open-jev", "gliner2-decide"
  dtype: "q4f16",
  onProgress: ({ progress }) =>
    console.log(`Model download: ${Math.round(progress * 100)}%`),
});

const state =
  "I was charged twice for the same order and nobody answers my emails. I want my money back now.";

const [area, sentiment, refund] = await jev.decide(state, [
  choice("Which product area is the message about?", [
    "fees & charges",
    "refund & dispute",
    "card",
    "other",
  ]),
  score("How positive is the sentiment of this message?", [
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
  ]),
  noul("The customer is asking for a refund."),
]);

area.choice; // "fees & charges" | "refund & dispute" | "card" | "other"
area.confidence; // 0.59
sentiment.level; // "very negative" | ... | "very positive"
sentiment.score; // 0.91 (expected level index, may fall between levels)
refund.answer; // true
refund.probability; // 0.88 (p(yes))
```

Answers are fully typed: option literals flow from the question into the answer, so `area.choice` is a union of exactly the strings you passed.

## Keyed questions

Pass an object instead of an array and get the answers back under the same keys:

```ts
const answers = await jev.decide(state, {
  area: choice("Which product area?", [
    "fees & charges",
    "refund & dispute",
    "other",
  ]),
  refund: noul("The customer is asking for a refund."),
});

answers.area.choice; // "fees & charges" | "refund & dispute" | "other"
answers.refund.answer; // boolean
```

## Question types

| Builder                                        | Type     | Answer                                                         |
| ---------------------------------------------- | -------- | -------------------------------------------------------------- |
| `choice(instructions, options, descriptions?)` | `choice` | Pick one option. Descriptions render as `option: description`. |
| `score(instructions, levels)`                  | `score`  | Rate on an ordered scale of levels (first = lowest).           |
| `noul(statement)`                              | `noul`   | Does the statement hold for the state? (yes/no)                |

Limits per model:

| Model            | `choice` options | `score` levels |
| ---------------- | ---------------- | -------------- |
| `open-jev`       | 2 to 255         | 2 to 10        |
| `kev-*`          | 1 to 255         | 2 to 255       |
| `gliner2-decide` | 2 to 64          | 2 to 10        |

The builders are optional sugar. Plain objects work too:

```ts
await jev.decide(state, [
  {
    type: "choice",
    instructions: "Which team should handle this?",
    options: ["billing", "shipping", "other"],
    descriptions: { billing: "Charges, invoices, payment problems" },
  },
  {
    type: "score",
    instructions: "How urgent?",
    options: ["low", "medium", "high"],
  },
  { type: "noul", instructions: "The customer is angry." },
]);
```

### Answer shapes

```ts
type ChoiceAnswer<O> = {
  type: "choice";
  choice: O; // option with the highest probability
  confidence: number; // probability of `choice`
  probabilities: Record<O, number>;
};

type ScoreAnswer<L> = {
  type: "score";
  score: number; // expected level index, 0..levels-1
  normalized: number; // score rescaled to 0..1
  level: L; // level label closest to `score`
  confidence: number; // highest single-level probability
  probabilities: Record<L, number>;
};

type NoulAnswer = {
  type: "noul";
  answer: boolean; // probability >= 0.5
  probability: number; // p(yes)
  confidence: number; // max(p(yes), p(no))
};
```

## API

### `OpenJev.load(options?): Promise<OpenJev>`

Downloads (or reads from cache) the tokenizer and model and resolves to a ready instance. All options are optional:

- `model` (default `"kev-0.6b"`)
  - `"kev-0.6b"`, `"kev-4b"`, `"open-jev"`, `"gliner2-decide"`, or a Hugging Face repo id / path Transformers.js understands.
- `dtype` (default `"auto"`)
  - `fp32`, `fp16`, `q4` or `q4f16` (the kev models only ship `q4` and `q4f16`).
  - `auto` picks the model's best WebGPU variant (`q4f16` for kev, `fp16` for open-jev) when `shader-f16` is supported, `q4` everywhere else.
- `device` (default `"auto"`)
  - `webgpu`, `wasm`, or `cpu` (Node.js).
  - `auto` picks `webgpu` when available, `cpu` in Node.js, otherwise `wasm`.
- `onProgress`
  - Called with `{ progress, loaded, total }` while files download. `progress` is `0..1`, `loaded` and `total` are bytes. Only fires when the rounded value changes.
- `maxLength` (default `512` for open-jev, `8192` for kev)
  - Context limit. For open-jev the whole sequence; for kev the state plus one question branch.
- `temperature`, `maxStateTokens`, `truncation`
  - Defaults for `decide()`, see below.

### `OpenJev.info(options?): Promise<OpenJevInfo>`

Returns model cache/download metadata for a configuration (`model`, `device`, `dtype`) without loading anything.

- `isCached`: whether every required file is in the browser cache.
- `downloadSize`: sum of all required file sizes (bytes).
- `files`: the files Transformers.js will fetch.
- `model`, `family`, `device`, `dtype`: the resolved runtime.

### `jev.decide(state, questions, options?)`

One forward pass, returns typed answers. Per-call options override the defaults given to `load()`:

- `temperature` (default: open-jev's calibrated `1.05`, `1` for kev)
  - Softmax temperature applied to each question's logits.
- `maxStateTokens` (default `256` for open-jev, `8192` for kev)
  - Token budget for the state. It is cut further if the questions would not fit in `maxLength`.
- `truncation` (default `"cut"`)
  - `"cut"` drops trailing state tokens, `"error"` throws when the state does not fit.

`decide()` throws if the questions alone exceed `maxLength`. Concurrent calls are queued and run one after another.

### `jev.countTokens(text): number`

Number of tokens `text` occupies, without markers. Use it to check a state against `maxStateTokens` up front.

### `jev.runtime`

The `{ model, family, device, dtype }` that were loaded. `MODELS` maps each alias to its repo id.

### `jev.dispose(): Promise<void>`

Releases the ONNX session. Pending `decide()` calls finish first; the instance cannot be used afterwards.

## How it works

All families read the state once and score every option of every question in a single pass. The library builds the model-specific sequence and reads the right logits back.

**open-jev** (DeBERTa-v3-large):

```
[CLS] [STATE] state [Q] instructions [OPT] option_1 [OPT] option_2 … [Q] … [SEP]
```

together with a span-slot tensor (`seg`) and per-pair slot ids (`pair_q`, `pair_opt`). The graph returns one logit per (question, option) pair.

**kev** (Qwen3):

```
<state> state <q> instructions <opt> option_1 </opt> <opt> option_2 </opt> … <decide> <q> …
```

The graph takes only `input_ids` and `attention_mask`, derives a block-causal mask from the delimiters so each question sees the state and itself only, and returns one logit per token. The library reads the value at every option's `</opt>` position. Caller text is escaped (`<|name|>` becomes `<¦name¦>`) so it can never forge a delimiter.

**gliner2** (GLiNER2.5-Decide, DeBERTa-v3-large):

```
( [P] instructions ( [L] option_1 [L] option_2 … ) ) [SEP_STRUCT] ( [P] … ) [SEP_TEXT] word word …
```

This is the classification path of Fastino's [GLiNER2](https://github.com/fastino-ai/GLiNER2) processor, reproduced token for token. The state is lowercased, split into words with the processor's regex and tokenized one word at a time (and given a terminal `.` if it has none); the instructions and options keep their case. `choice` descriptions are appended to the instructions as ` [DESCRIPTION] option: description` rather than to the option. The graph takes `marker_positions`, the index of every `[L]` token, and returns one logit per marker from the model's own 1024→2048→1 head. The ONNX conversion matches the Python library to about 1e-7 at fp32 and 3e-4 at fp16; the q4 variants can flip a close call.

In all cases a temperature-scaled softmax within each question's group is that question's distribution. `noul` questions use the fixed options `["no", "yes"]` the models were trained with.

## Development

```bash
pnpm install
pnpm dev
```

`pnpm dev` builds the library once, then runs the library watch build and the example app's Vite dev server side by side. Use `pnpm dev:lib` or `pnpm dev:example` to run only one of them.

Build for publish:

```bash
pnpm build
```

Type-check:

```bash
pnpm typecheck
```

Publish to npm (checks login, bumps the version, type-checks, builds, publishes):

```bash
pnpm release <otp> [patch|minor|major]
```

## Example app

A minimal vanilla demo is included at `examples/simple`.

```bash
cd examples/simple
pnpm install --ignore-workspace
pnpm dev
```

## Notes

- Designed for browser environments; works in Node.js with `device: "cpu"` (or `auto`).
- The models are English only. open-jev was trained on three public domains (banking support, movie reviews, Wikipedia yes/no), the kev models on ten. Questions outside these domains work but are less accurate; measure before relying on them. See the model cards linked above for numbers and limitations.
- Loading a kev model logs one Transformers.js warning ("assuming encoder-only architecture"). It is expected: the graph is a custom pointer-head export, not a text generator.
- Model weights are Apache-2.0, this library is MIT.
