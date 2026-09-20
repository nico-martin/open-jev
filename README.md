# open-jev

`open-jev` is a browser-focused TypeScript library for **typed decisions**: one piece of text (the _state_) plus any number of typed questions go in, and one forward pass returns a calibrated probability distribution per question. Nothing is generated, so an answer is always one of the options you provided.

It runs [open-jev-deberta-v3-large](https://huggingface.co/onnx-community/open-jev-deberta-v3-large-ONNX), an open reproduction of the _shape_ of TypeSafe AI's [Jev "System One" model](https://typesafe.ai/blog/introducing-system-one-models-and-jev), via [`@huggingface/transformers` (Transformers.js)](https://huggingface.co/docs/transformers.js/en/index). Everything happens on-device: WebGPU when available, WebAssembly otherwise.

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

| Builder                         | Type     | Answer                                                       |
| ------------------------------- | -------- | ------------------------------------------------------------ |
| `choice(instructions, options)` | `choice` | Pick one of 2 to 255 options.                                |
| `score(instructions, levels)`   | `score`  | Rate on an ordered scale of 2 to 10 levels (first = lowest). |
| `noul(statement)`               | `noul`   | Does the statement hold for the state? (yes/no)              |

The builders are optional sugar. Plain objects work too:

```ts
await jev.decide(state, [
  {
    type: "choice",
    instructions: "Which product area?",
    options: ["card", "other"],
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

- `model` (default `onnx-community/open-jev-deberta-v3-large-ONNX`)
  - Hugging Face repo id or any path Transformers.js understands.
- `dtype` (default `"auto"`)
  - `fp16` (0.88 GB), `fp32` (1.75 GB), `q4` (0.48 GB) or `q4f16` (0.35 GB).
  - `auto` picks `fp16` on WebGPU with `shader-f16` support, `q4` everywhere else.
- `device` (default `"auto"`)
  - `webgpu`, `wasm`, or `cpu` (Node.js).
  - `auto` picks `webgpu` when available, `cpu` in Node.js, otherwise `wasm`.
- `onProgress`
  - Called with `{ progress, loaded, total }` while files download. `progress` is `0..1`, `loaded` and `total` are bytes. Only fires when the rounded value changes.
- `maxLength` (default `512`)
  - Total sequence limit of the model.
- `temperature`, `maxStateTokens`, `truncation`
  - Defaults for `decide()`, see below.

### `OpenJev.info(options?): Promise<OpenJevInfo>`

Returns model cache/download metadata for a configuration (`model`, `device`, `dtype`) without loading anything.

- `isCached`: whether every required file is in the browser cache.
- `downloadSize`: sum of all required file sizes (bytes).
- `files`: the files Transformers.js will fetch.
- `device`, `dtype`: the resolved runtime.

### `jev.decide(state, questions, options?)`

One forward pass, returns typed answers. Per-call options override the defaults given to `load()`:

- `temperature` (default: the model's calibrated `1.05`)
  - Softmax temperature applied to each question's logits.
- `maxStateTokens` (default `256`)
  - Token budget for the state. It is cut further if the questions would not fit in `maxLength`.
- `truncation` (default `"cut"`)
  - `"cut"` drops trailing state tokens, `"error"` throws when the state does not fit.

`decide()` throws if the questions alone exceed `maxLength`. Concurrent calls are queued and run one after another.

### `jev.countTokens(text): number`

Number of tokens `text` occupies, without markers. Use it to check a state against `maxStateTokens` up front.

### `jev.runtime`

The `{ device, dtype }` that were loaded.

### `jev.dispose(): Promise<void>`

Releases the ONNX session. Pending `decide()` calls finish first; the instance cannot be used afterwards.

## How it works

The library builds the single sequence the model expects:

```
[CLS] [STATE] state [Q] instructions [OPT] option_1 [OPT] option_2 … [Q] … [SEP]
```

together with the span-slot tensor (`seg`) and per-pair slot ids (`pair_q`, `pair_opt`). The graph returns one logit per (question, option) pair; a temperature-scaled softmax within each question's group is that question's distribution. `noul` questions use the fixed options `["no", "yes"]` the model was trained with.

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
pnpm run publish <otp> [patch|minor|major]
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
- Context is 512 tokens in total, the state is cut to 256 tokens by default.
- The model is English only and was trained on three public domains (banking support, movie reviews, Wikipedia yes/no). Questions outside these domains work but are less accurate; measure before relying on them. See the [model card](https://huggingface.co/onnx-community/open-jev-deberta-v3-large-ONNX) for numbers and limitations.
- Model weights are Apache-2.0, this library is MIT.
