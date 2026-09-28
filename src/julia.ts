import {
  AutoModel,
  ModelRegistry,
  PretrainedConfig,
  Tensor,
} from "@huggingface/transformers";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import type { FamilyAdapter, JevModel } from "./models";
import type { Question } from "./types";

/**
 * Julia 1 by Supersonic Labs (mmBERT-small with a decision head).
 *
 * Unlike the other families, Julia scores each question in its own sequence,
 * question first and state last:
 *
 *   <bos> {type} question: {question} <eos> <mask> {opt 1} … <mask> {opt n} <eos> {state} <eos>
 *
 * The graph takes the position of every `<mask>` (`marker_pos`), which of
 * them are real (`marker_mask`) and the question type (`qtype`), and returns
 * one logit per option. All questions of a `decide()` call go through the
 * model as one batch. This mirrors `julia/data.py` of the original checkpoint.
 */

const QTYPES = { choice: 0, score: 1, noul: 2 } as const;

/** Answer texts Julia's typed API uses for `noul` (index 1 = true). */
const NOUL_OPTION_TEXTS = ["false", "true"] as const;

/** Limits of the original model: 2–20 options, at most 48 tokens each. */
const MAX_OPTIONS = 20;
const MAX_OPTION_TOKENS = 48;
/**
 * Room for the question and its options at the front of every sequence.
 * The model's own runtime and benchmark use 512; the budget only decides
 * when a question is too long, not how it is encoded.
 */
const HEAD_LENGTH = 512;

/** Repos known to ship the Julia 1 graph without a `config.json`. */
export const JULIA_REPOS = ["SupersonicLabs/Julia-1-ONNX"];

/** The published export keeps the fp32 graph at the repo root. */
const MODEL_FILES = ["model.onnx", "model.onnx.data"];
export const JULIA_FILES = [
  "tokenizer.json",
  "tokenizer_config.json",
  ...MODEL_FILES,
];

type Special = {
  bos: number;
  eos: number;
  mask: number;
  pad: number;
  space: number;
};

type Sequence = {
  ids: number[];
  markers: number[];
  qtype: number;
  stateTokens: number;
  stateTruncated: boolean;
};

export function juliaConfig(): PretrainedConfig {
  // The export has no config.json; the graph is an mmBERT (ModernBERT)
  // encoder, which Transformers.js runs through its single-session path.
  return new PretrainedConfig({ model_type: "modernbert" } as never);
}

export function juliaFamily(): FamilyAdapter {
  let tokenizer: PreTrainedTokenizer | null = null;
  let special: Special | null = null;

  const raw = (text: string): number[] => {
    const { input_ids } = tokenizer!(text, { add_special_tokens: false }) as {
      input_ids: Tensor;
    };
    return Array.from(input_ids.data as ArrayLike<bigint | number>, Number);
  };

  const single = (text: string): number => {
    const ids = raw(text);
    if (ids.length !== 1) {
      throw new Error(`Tokenizer does not know the token ${text}.`);
    }
    return ids[0];
  };

  /**
   * Tokenize like the Rust `tokenizers` library the model was trained with.
   * Its Metaspace pre-tokenizer (split, always prepend) starts a new piece at
   * every space, so "a  b" is `▁a ▁ ▁b`. Transformers.js merges the run into
   * `▁▁ b`, which changed predictions on texts with double spaces. Encoding
   * one space-free word at a time avoids that path.
   */
  const encode = (input: string): number[] => {
    // Caller text can never produce the option marker (the original's non-strict mode).
    const text = input.replace(/<mask>/g, " ");
    if (text === "") {
      return [];
    }
    const spaced = text.startsWith(" ") ? text : ` ${text}`;
    const ids: number[] = [];
    for (const piece of spaced.split(/(?= )/)) {
      const word = piece.slice(1);
      if (word === "") {
        ids.push(special!.space);
      } else {
        ids.push(...raw(word));
      }
    }
    return ids;
  };

  const optionTexts = (question: Question): string[] => {
    if (question.type === "noul") {
      // Julia was trained with a description of each outcome; without one it
      // falls back to the literal words (80.5% vs 65.2% on its noul benchmark).
      const [no, yes] = NOUL_OPTION_TEXTS;
      return [
        question.descriptions?.false ?? no,
        question.descriptions?.true ?? yes,
      ];
    }
    // Julia's typed API renders a described choice as its description alone.
    if (question.type === "choice" && question.descriptions) {
      const descriptions = question.descriptions as Record<
        string,
        string | undefined
      >;
      return question.options.map((option) => descriptions[option] || option);
    }
    return [...question.options];
  };

  const sequence = (
    question: Question,
    state: number[],
    maxLength: number,
    maxStateTokens: number,
  ): Sequence => {
    const { bos, eos, mask } = special!;
    const head = encode(`${question.type} question: ${question.instructions}`);
    const options = optionTexts(question).map((text, index) => {
      const ids = encode(` ${text}`);
      if (ids.length > MAX_OPTION_TOKENS) {
        throw new Error(
          `Option ${index + 1} of "${question.instructions}" is ${ids.length} tokens; Julia takes at most ${MAX_OPTION_TOKENS}.`,
        );
      }
      return [mask, ...ids];
    });
    const budget =
      HEAD_LENGTH - options.reduce((sum, option) => sum + option.length, 0);
    if (head.length > budget) {
      throw new Error(
        `Question "${question.instructions}" and its options do not fit Julia's ${HEAD_LENGTH}-token head; shorten them.`,
      );
    }

    const ids = [bos, ...head, eos];
    const markers: number[] = [];
    for (const option of options) {
      markers.push(ids.length);
      ids.push(...option);
    }
    ids.push(eos);

    const room = Math.min(maxStateTokens, maxLength - ids.length - 1);
    if (room < 0) {
      throw new Error(
        "Question and options leave no room for the state; raise maxLength.",
      );
    }
    const kept = state.slice(0, room);
    ids.push(...kept, eos);
    return {
      ids,
      markers,
      qtype: QTYPES[question.type],
      stateTokens: kept.length,
      stateTruncated: kept.length < state.length,
    };
  };

  return {
    name: "julia",
    webgpuDtype: "fp32",
    fallbackDtype: "fp32",
    dtypes: ["fp32"],
    limits: {
      minChoiceOptions: 2,
      maxChoiceOptions: MAX_OPTIONS,
      minScoreLevels: 2,
      maxScoreLevels: MAX_OPTIONS,
    },
    defaults: {
      temperature: 1,
      // The state gets whatever the question leaves of the sequence.
      maxStateTokens: 8192,
      // The browser export's default; mmBERT itself takes up to 8192.
      maxLength: 1024,
    },
    files: () => JULIA_FILES,
    loadModel: async (modelId, options) => {
      // Transformers.js sizes its combined progress from the standard
      // onnx/ layout, which this repo does not use, so it would only emit
      // per-file events. Sum them here into the progress_total events the
      // other families get.
      const files: Record<string, { loaded: number; total: number }> = {};
      await Promise.all(
        MODEL_FILES.map(async (file) => {
          const meta = await ModelRegistry.get_file_metadata(modelId, file);
          files[file] = { loaded: 0, total: meta.size ?? 0 };
        }),
      );
      const progress = (info: {
        status: string;
        file?: string;
        loaded?: number;
        total?: number;
      }): void => {
        const file = info.file?.replace(/^\//, "");
        if (info.status !== "progress" || !file || !(file in files)) {
          return;
        }
        files[file] = {
          loaded: info.loaded ?? 0,
          total: info.total || files[file].total,
        };
        const loaded = Object.values(files).reduce(
          (sum, f) => sum + f.loaded,
          0,
        );
        const total = Object.values(files).reduce((sum, f) => sum + f.total, 0);
        options.progress_callback({
          status: "progress_total",
          progress: total > 0 ? (loaded / total) * 100 : 0,
          loaded,
          total,
        });
      };
      return (await AutoModel.from_pretrained(modelId, {
        config: options.config,
        dtype: options.dtype,
        device: options.device as "webgpu",
        progress_callback: progress,
        subfolder: "",
        model_file_name: "model",
        // The weights file is named as the graph references it.
        session_options: {
          externalData: [{ path: "model.onnx.data", data: "model.onnx.data" }],
        },
      } as never)) as unknown as JevModel;
    },
    prepare(loaded) {
      tokenizer = loaded;
      const t = loaded as unknown as Record<string, string | undefined>;
      special = {
        bos: single(t.cls_token ?? t.bos_token ?? "<bos>"),
        eos: single(t.sep_token ?? t.eos_token ?? "<eos>"),
        mask: single(t.mask_token ?? "<mask>"),
        pad: single(t.pad_token ?? "<pad>"),
        space: raw(" ")[0],
      };
    },
    escape: (text) => text.replace(/<mask>/g, " "),
    encode() {
      throw new Error(
        "Julia scores each question in its own sequence; use decide().",
      );
    },
    countTokens: (text) => encode(text).length,
    async decide({ model, state, questions, maxLength, maxStateTokens }) {
      if (!special) {
        throw new Error("Model family not prepared.");
      }
      const stateIds = encode(state);
      const items = questions.map((question) =>
        sequence(question, stateIds, maxLength, maxStateTokens),
      );

      // One batch, padded like the original collator: to a multiple of 8
      // with <pad>, masked out by the attention mask.
      const batch = items.length;
      const length =
        Math.ceil(Math.max(...items.map((item) => item.ids.length)) / 8) * 8;
      const count = Math.max(...items.map((item) => item.markers.length));
      const ids = new BigInt64Array(batch * length).fill(BigInt(special.pad));
      const attention = new BigInt64Array(batch * length);
      const positions = new BigInt64Array(batch * count);
      const markerMask = new Uint8Array(batch * count);
      const qtype = new BigInt64Array(batch);
      items.forEach((item, row) => {
        item.ids.forEach((id, column) => {
          ids[row * length + column] = BigInt(id);
          attention[row * length + column] = 1n;
        });
        item.markers.forEach((position, column) => {
          positions[row * count + column] = BigInt(position);
          markerMask[row * count + column] = 1;
        });
        qtype[row] = BigInt(item.qtype);
      });

      const { logits } = await model({
        input_ids: new Tensor("int64", ids, [batch, length]),
        attention_mask: new Tensor("int64", attention, [batch, length]),
        marker_pos: new Tensor("int64", positions, [batch, count]),
        marker_mask: new Tensor("bool", markerMask, [batch, count]),
        qtype: new Tensor("int64", qtype, [batch]),
      });
      const values = Array.from(logits.to("float32").data as ArrayLike<number>);

      return {
        logits: items.map((item, row) =>
          values.slice(row * count, row * count + item.markers.length),
        ),
        stateTokens: items.length
          ? Math.min(...items.map((item) => item.stateTokens))
          : 0,
        stateTruncated: items.some((item) => item.stateTruncated),
      };
    },
  };
}
