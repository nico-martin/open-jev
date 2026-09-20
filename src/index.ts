import {
  AutoModel,
  AutoTokenizer,
  ModelRegistry,
  Tensor,
} from "@huggingface/transformers";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import { decodeAnswer } from "./answers";
import {
  encodeSequence,
  type MarkerIds,
  type TokenizedQuestion,
} from "./encoding";
import { questionOptions, validateQuestion } from "./questions";
import type {
  Answer,
  AnswersFor,
  DecideOptions,
  LoadProgress,
  ModelProgressCallback,
  OpenJevInfo,
  OpenJevOptions,
  OpenJevRuntime,
  Question,
  Questions,
} from "./types";
import { normalizeError } from "./utils/errors";
import { clamp } from "./utils/math";
import { resolveRuntime } from "./utils/runtime";

export { choice, noul, score } from "./questions";
export type {
  Answer,
  AnswerFor,
  AnswersFor,
  ChoiceAnswer,
  ChoiceQuestion,
  DecideOptions,
  LoadProgress,
  NoulAnswer,
  NoulQuestion,
  OpenJevDevice,
  OpenJevDtype,
  OpenJevInfo,
  OpenJevOptions,
  OpenJevRuntime,
  Question,
  Questions,
  ScoreAnswer,
  ScoreQuestion,
} from "./types";

const DEFAULT_MODEL_ID = "onnx-community/open-jev-deberta-v3-large-ONNX";
const DEFAULT_TEMPERATURE = 1.05;
const DEFAULT_MAX_STATE_TOKENS = 256;
const DEFAULT_MAX_LENGTH = 512;

type JevModel = {
  (inputs: Record<string, Tensor>): Promise<{ logits: Tensor }>;
  config?: { open_jev?: { temperature?: number } };
  dispose?: () => Promise<unknown>;
};

type LoadedParts = {
  model: JevModel;
  tokenizer: PreTrainedTokenizer;
  markers: MarkerIds;
  runtime: OpenJevRuntime;
  defaults: Required<DecideOptions>;
  maxLength: number;
};

/**
 * Typed decisions in the browser with the open-jev model.
 *
 * One `state` (any text) plus any number of typed questions go in, one
 * forward pass returns a calibrated probability distribution per question.
 * Nothing is generated, so answers are always one of the options you gave.
 *
 * Create an instance with `OpenJev.load()`.
 */
export class OpenJev {
  /** The backend and weight variant that were loaded. */
  readonly runtime: OpenJevRuntime;

  private readonly model: JevModel;
  private readonly tokenizer: PreTrainedTokenizer;
  private readonly markers: MarkerIds;
  private readonly defaults: Required<DecideOptions>;
  private readonly maxLength: number;

  private queue: Promise<unknown> = Promise.resolve();
  private disposed = false;

  private constructor(parts: LoadedParts) {
    this.model = parts.model;
    this.tokenizer = parts.tokenizer;
    this.markers = parts.markers;
    this.runtime = parts.runtime;
    this.defaults = parts.defaults;
    this.maxLength = parts.maxLength;
  }

  /**
   * Download (or read from cache) the tokenizer and model and return a ready
   * instance.
   *
   * Supported options:
   * - `model`: Hugging Face repo id (default `onnx-community/open-jev-deberta-v3-large-ONNX`).
   * - `dtype`: `fp32 | fp16 | q4 | q4f16 | auto` (default `auto`).
   * - `device`: `webgpu | wasm | cpu | auto` (default `auto`).
   * - `onProgress`: download progress callback.
   * - `temperature`, `maxStateTokens`, `truncation`: defaults for `decide()`.
   * - `maxLength`: total sequence limit (default `512`).
   */
  static async load(options: OpenJevOptions = {}): Promise<OpenJev> {
    const modelId = options.model ?? DEFAULT_MODEL_ID;
    const runtime = await resolveRuntime(options);
    const onProgress = options.onProgress;

    let lastProgress = -1;
    const progressCallback: ModelProgressCallback = (info): void => {
      if (!onProgress || info.status !== "progress_total") {
        return;
      }

      const progress = Math.round(clamp(info.progress / 100, 0, 1) * 100) / 100;
      if (progress === lastProgress) {
        return;
      }

      lastProgress = progress;
      onProgress({ progress, loaded: info.loaded, total: info.total });
    };

    const tokenizer = await AutoTokenizer.from_pretrained(modelId);

    const encodeMarker = (marker: string): number => {
      const ids = encode(tokenizer, marker);
      if (ids.length !== 1) {
        throw new Error(
          `Tokenizer of "${modelId}" does not know the marker token ${marker}.`,
        );
      }
      return ids[0];
    };

    const markers: MarkerIds = {
      cls: encodeMarker("[CLS]"),
      sep: encodeMarker("[SEP]"),
      state: encodeMarker("[STATE]"),
      q: encodeMarker("[Q]"),
      opt: encodeMarker("[OPT]"),
    };

    const model = (await AutoModel.from_pretrained(modelId, {
      dtype: runtime.dtype,
      device: runtime.device,
      progress_callback: progressCallback,
    })) as unknown as JevModel;

    const defaults: Required<DecideOptions> = {
      temperature:
        options.temperature ??
        model.config?.open_jev?.temperature ??
        DEFAULT_TEMPERATURE,
      maxStateTokens: options.maxStateTokens ?? DEFAULT_MAX_STATE_TOKENS,
      truncation: options.truncation ?? "cut",
    };

    return new OpenJev({
      model,
      tokenizer,
      markers,
      runtime,
      defaults,
      maxLength: Math.max(
        8,
        Math.floor(options.maxLength ?? DEFAULT_MAX_LENGTH),
      ),
    });
  }

  /**
   * Get model metadata for a configuration without loading it.
   *
   * - `isCached`: whether every required file is present in the browser cache.
   * - `downloadSize`: total size in bytes of the files that will be fetched.
   * - `files`: the file list, `device` and `dtype`: the resolved runtime.
   */
  static async info(
    options: Pick<OpenJevOptions, "model" | "device" | "dtype"> = {},
  ): Promise<OpenJevInfo> {
    const modelId = options.model ?? DEFAULT_MODEL_ID;
    const runtime = await resolveRuntime(options);

    const files = await ModelRegistry.get_files(modelId, {
      dtype: runtime.dtype,
      device: runtime.device,
      include_tokenizer: true,
      include_processor: false,
    });

    const [isCached, metadata] = await Promise.all([
      ModelRegistry.is_cached(modelId, {
        dtype: runtime.dtype,
        device: runtime.device,
      }),
      Promise.all(
        files.map((file) => ModelRegistry.get_file_metadata(modelId, file)),
      ),
    ]);

    const downloadSize = metadata.reduce(
      (sum, meta) => sum + (meta.size ?? 0),
      0,
    );

    return { ...runtime, isCached, downloadSize, files };
  }

  /**
   * Answer typed questions about one state in a single forward pass.
   *
   * Pass questions as an array (answers come back as a tuple in the same
   * order) or as an object (answers come back under the same keys).
   */
  async decide<const Qs extends Questions>(
    state: string,
    questions: Qs,
    options: DecideOptions = {},
  ): Promise<AnswersFor<Qs>> {
    this.assertNotDisposed();

    if (typeof state !== "string") {
      throw new Error("OpenJev.decide() expects the state to be a string.");
    }

    const isList = Array.isArray(questions);
    const keys = isList
      ? (questions as readonly Question[]).map((_, index) => String(index))
      : Object.keys(questions);
    const list = isList
      ? [...(questions as readonly Question[])]
      : Object.values(questions as Readonly<Record<string, Question>>);

    if (list.length === 0) {
      throw new Error("OpenJev.decide() needs at least one question.");
    }

    list.forEach((question, index) =>
      validateQuestion(question, isList ? `#${index}` : `"${keys[index]}"`),
    );

    const settings: Required<DecideOptions> = {
      temperature: Math.max(
        1e-6,
        options.temperature ?? this.defaults.temperature,
      ),
      maxStateTokens: Math.max(
        0,
        Math.floor(options.maxStateTokens ?? this.defaults.maxStateTokens),
      ),
      truncation: options.truncation ?? this.defaults.truncation,
    };

    const answers = await this.enqueue(() => this.run(state, list, settings));

    if (isList) {
      return answers as unknown as AnswersFor<Qs>;
    }

    const record: Record<string, Answer> = {};
    keys.forEach((key, index) => {
      record[key] = answers[index];
    });
    return record as unknown as AnswersFor<Qs>;
  }

  /**
   * Number of tokens `text` occupies in the sequence (without markers).
   * Useful to check a state against `maxStateTokens` before deciding.
   */
  countTokens(text: string): number {
    this.assertNotDisposed();
    return encode(this.tokenizer, text).length;
  }

  /**
   * Release the ONNX session. Pending `decide()` calls finish first; the
   * instance cannot be used afterwards.
   */
  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }

    this.disposed = true;

    try {
      await this.queue;
    } catch {
      // A pending decision failed; nothing left to wait for.
    }

    await this.model.dispose?.();
  }

  private async run(
    state: string,
    questions: Question[],
    settings: Required<DecideOptions>,
  ): Promise<Answer[]> {
    try {
      const tokenized: TokenizedQuestion[] = questions.map((question) => ({
        instructions: encode(this.tokenizer, question.instructions),
        options: questionOptions(question).map((option) =>
          encode(this.tokenizer, option),
        ),
      }));

      const encoded = encodeSequence({
        state: encode(this.tokenizer, state),
        questions: tokenized,
        markers: this.markers,
        maxStateTokens: settings.maxStateTokens,
        maxLength: this.maxLength,
      });

      if (encoded.stateTruncated && settings.truncation === "error") {
        throw new Error(
          `State was cut to ${encoded.stateTokens} tokens (limit ${settings.maxStateTokens}, ${this.maxLength} in total). Shorten the state or the questions, or set truncation to "cut".`,
        );
      }

      const { logits } = await this.model({
        input_ids: int64(encoded.inputIds),
        attention_mask: int64(encoded.inputIds.map(() => 1)),
        seg: int64(encoded.seg),
        pair_q: int64(encoded.pairQ),
        pair_opt: int64(encoded.pairOpt),
      });

      const scores = Array.from(logits.to("float32").data as ArrayLike<number>);

      return questions.map((question, index) =>
        decodeAnswer(
          question,
          encoded.groups[index].map((pair) => scores[pair]),
          settings.temperature,
        ),
      );
    } catch (error) {
      throw normalizeError(error);
    }
  }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    const result = this.queue.then(task, task);
    this.queue = result.catch(() => undefined);
    return result;
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error("OpenJev instance has been disposed.");
    }
  }
}

function encode(tokenizer: PreTrainedTokenizer, text: string): number[] {
  const { input_ids } = tokenizer(text, { add_special_tokens: false }) as {
    input_ids: Tensor;
  };
  return Array.from(input_ids.data as ArrayLike<bigint | number>, Number);
}

function int64(values: number[]): Tensor {
  return new Tensor("int64", BigInt64Array.from(values, BigInt), [
    1,
    values.length,
  ]);
}
