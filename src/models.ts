import { AutoModel, PreTrainedModel } from "@huggingface/transformers";
import type {
  PretrainedConfig,
  PreTrainedTokenizer,
  Tensor,
} from "@huggingface/transformers";
import {
  encodeKevSequence,
  encodeSequence,
  type EncodedSequence,
  type TokenizedQuestion,
} from "./encoding";
import type { QuestionLimits } from "./questions";
import type { ModelAlias, ModelFamily, OpenJevDtype } from "./types";

/** Hugging Face repos behind the built-in aliases. */
export const MODELS: Record<ModelAlias, string> = {
  "open-jev": "onnx-community/open-jev-deberta-v3-large-ONNX",
  "kev-0.6b": "onnx-community/kev-0.6b-ONNX",
  "kev-4b": "onnx-community/kev-4b-ONNX",
};

export const DEFAULT_MODEL: ModelAlias = "kev-0.6b";

export function resolveModelId(model: string | undefined): string {
  const id = model ?? DEFAULT_MODEL;
  return (MODELS as Record<string, string>)[id] ?? id;
}

export type JevModel = {
  (inputs: Record<string, Tensor>): Promise<{ logits: Tensor }>;
  dispose?: () => Promise<unknown>;
};

export type LoadModelOptions = {
  config: PretrainedConfig;
  dtype: OpenJevDtype;
  device: string;
  progress_callback: (info: unknown) => void;
};

export type FamilyDefaults = {
  temperature: number;
  maxStateTokens: number;
  maxLength: number;
};

/** Everything that differs between the model families. */
export type FamilyAdapter = {
  name: ModelFamily;
  /** Best variant on WebGPU with `shader-f16`; `q4` is used elsewhere. */
  webgpuDtype: OpenJevDtype;
  limits: QuestionLimits;
  defaults: FamilyDefaults;
  loadModel(modelId: string, options: LoadModelOptions): Promise<JevModel>;
  /** Resolve marker/delimiter token ids once the tokenizer is available. */
  prepare(tokenizer: PreTrainedTokenizer): void;
  /** Applied to every caller-provided text before tokenization. */
  escape(text: string): string;
  encode(params: {
    state: number[];
    questions: TokenizedQuestion[];
    maxStateTokens: number;
    maxLength: number;
  }): EncodedSequence;
};

type ConfigJson = Record<string, unknown>;

export function detectFamily(config: PretrainedConfig): FamilyAdapter {
  const json = config as unknown as ConfigJson;
  if (json.kev && typeof json.kev === "object") {
    return kevFamily(json.kev as ConfigJson);
  }
  if (json.open_jev && typeof json.open_jev === "object") {
    return openJevFamily(json.open_jev as ConfigJson);
  }
  throw new Error(
    `Unsupported model: config.json has neither an "open_jev" nor a "kev" section. Use one of ${Object.keys(MODELS).join(", ")} or a compatible ONNX conversion.`,
  );
}

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function encodeSingle(tokenizer: PreTrainedTokenizer, text: string): number {
  const { input_ids } = tokenizer(text, { add_special_tokens: false }) as {
    input_ids: Tensor;
  };
  const ids = Array.from(input_ids.data as ArrayLike<bigint | number>, Number);
  if (ids.length !== 1) {
    throw new Error(`Tokenizer does not know the marker token ${text}.`);
  }
  return ids[0];
}

function openJevFamily(section: ConfigJson): FamilyAdapter {
  let markers: Parameters<typeof encodeSequence>[0]["markers"] | null = null;

  return {
    name: "open-jev",
    webgpuDtype: "fp16",
    limits: {
      minChoiceOptions: 2,
      maxChoiceOptions: 255,
      minScoreLevels: 2,
      maxScoreLevels: 10,
    },
    defaults: {
      temperature: num(section.temperature, 1.05),
      maxStateTokens: num(section.max_state_tokens, 256),
      maxLength: num(section.max_len, 512),
    },
    loadModel: async (modelId, options) =>
      (await AutoModel.from_pretrained(modelId, {
        config: options.config,
        dtype: options.dtype,
        device: options.device as "webgpu",
        progress_callback: options.progress_callback,
      })) as unknown as JevModel,
    prepare(tokenizer) {
      markers = {
        cls: encodeSingle(tokenizer, "[CLS]"),
        sep: encodeSingle(tokenizer, "[SEP]"),
        state: encodeSingle(tokenizer, "[STATE]"),
        q: encodeSingle(tokenizer, "[Q]"),
        opt: encodeSingle(tokenizer, "[OPT]"),
      };
    },
    escape: (text) => text,
    encode(params) {
      if (!markers) {
        throw new Error("Model family not prepared.");
      }
      return encodeSequence({ ...params, markers });
    },
  };
}

function kevFamily(section: ConfigJson): FamilyAdapter {
  const ids = (section.delimiter_ids ?? {}) as Record<string, unknown>;
  const names = (section.delimiters ?? {}) as Record<string, unknown>;
  let delimiters: Parameters<typeof encodeKevSequence>[0]["delimiters"] | null =
    null;

  const resolve = (
    tokenizer: PreTrainedTokenizer,
    key: string,
    fallback: string,
  ): number => {
    const id = ids[key];
    if (typeof id === "number") {
      return id;
    }
    const name =
      typeof names[key] === "string" ? (names[key] as string) : fallback;
    return encodeSingle(tokenizer, name);
  };

  return {
    name: "kev",
    webgpuDtype: "q4f16",
    limits: {
      minChoiceOptions: 1,
      maxChoiceOptions: num(section.max_options, 255),
      minScoreLevels: 2,
      maxScoreLevels: num(section.max_options, 255),
    },
    defaults: {
      temperature: 1,
      maxStateTokens: num(section.max_state_tokens, 8192),
      maxLength: num(section.max_branch_tokens, 8192),
    },
    // Transformers.js maps qwen3 to its text-generation class, which expects a
    // KV-cache graph. The base class takes the single-session path that feeds
    // every graph input by name. It logs an "assuming encoder-only" warning.
    loadModel: async (modelId, options) =>
      (await PreTrainedModel.from_pretrained(modelId, {
        config: options.config,
        dtype: options.dtype,
        device: options.device as "webgpu",
        progress_callback: options.progress_callback,
      })) as unknown as JevModel,
    prepare(tokenizer) {
      delimiters = {
        state: resolve(tokenizer, "state", "<|fim_prefix|>"),
        question: resolve(tokenizer, "question", "<|fim_middle|>"),
        optionStart: resolve(tokenizer, "option_start", "<|box_start|>"),
        optionEnd: resolve(tokenizer, "option_end", "<|box_end|>"),
        decide: resolve(tokenizer, "decide", "<|fim_suffix|>"),
      };
    },
    // Caller text can never produce a delimiter: <|name|> -> <¦name¦>
    escape: (text) => text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>"),
    encode(params) {
      if (!delimiters) {
        throw new Error("Model family not prepared.");
      }
      return encodeKevSequence({ ...params, delimiters });
    },
  };
}
