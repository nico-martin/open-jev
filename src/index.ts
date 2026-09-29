import {
  AutoConfig,
  AutoTokenizer,
  ModelRegistry,
  Tensor,
} from "@huggingface/transformers";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import { decodeAnswer } from "./answers";
import type { TokenizedQuestion } from "./encoding";
import {
  detectFamily,
  MODELS,
  resolveModelId,
  type FamilyAdapter,
  type JevModel,
} from "./models";
import { questionOptionTexts, validateQuestion } from "./questions";
import type {
  Answer,
  AnswersFor,
  DecideOptions,
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

export { MODELS } from "./models";
export { choice, noul, score } from "./questions";
export type {
  Answer,
  AnswerFor,
  AnswersFor,
  ChoiceAnswer,
  ChoiceQuestion,
  DecideOptions,
  LoadProgress,
  ModelAlias,
  ModelFamily,
  ModelId,
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

type LoadedParts = {
  model: JevModel;
  tokenizer: PreTrainedTokenizer;
  family: FamilyAdapter;
  runtime: OpenJevRuntime;
  defaults: Required<DecideOptions>;
  maxLength: number;
};

/**
 * Typed decisions in the browser with Jev-shaped models.
 *
 * One `state` (any text) plus any number of typed questions go in, one
 * forward pass returns a calibrated probability distribution per question.
 * Nothing is generated, so answers are always one of the options you gave.
 *
 * Create an instance with `OpenJev.load()`. Built-in models: `kev-0.6b`
 * (default) and `kev-4b` (Qwen3), `open-jev` (DeBERTa-v3-large) and
 * `gliner2-decide` (GLiNER2.5-Decide, DeBERTa-v3-large).
 */
export class OpenJev {
  /** The model, family, backend and weight variant that were loaded. */
  readonly runtime: OpenJevRuntime;

  private readonly model: JevModel;
  private readonly tokenizer: PreTrainedTokenizer;
  private readonly family: FamilyAdapter;
  private readonly defaults: Required<DecideOptions>;
  private readonly maxLength: number;

  private queue: Promise<unknown> = Promise.resolve();
  private disposed = false;

  private constructor(parts: LoadedParts) {
    this.model = parts.model;
    this.tokenizer = parts.tokenizer;
    this.family = parts.family;
    this.runtime = parts.runtime;
    this.defaults = parts.defaults;
    this.maxLength = parts.maxLength;
  }

  /**
   * Download (or read from cache) the tokenizer and model and return a ready
   * instance.
   *
   * Supported options:
   * - `model`: `kev-0.6b` (default), `kev-4b`, `open-jev`, `gliner2-decide` or a Hugging Face repo id.
   * - `dtype`: `fp32 | fp16 | q4 | q4f16 | auto` (default `auto`).
   * - `device`: `webgpu | wasm | cpu | auto` (default `auto`).
   * - `onProgress`: download progress callback.
   * - `temperature`, `maxStateTokens`, `truncation`: defaults for `decide()`.
   * - `maxLength`: context limit (model-specific default).
   */
  static async load(options: OpenJevOptions = {}): Promise<OpenJev> {
    const modelId = resolveModelId(options.model);
    const config = await AutoConfig.from_pretrained(modelId);
    const family = detectFamily(config);
    const runtime = await resolveRuntime(options, family.webgpuDtype);
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
    family.prepare(tokenizer);

    const model = await family.loadModel(modelId, {
      config,
      dtype: runtime.dtype,
      device: runtime.device,
      progress_callback: progressCallback as (info: unknown) => void,
    });

    const defaults: Required<DecideOptions> = {
      temperature: options.temperature ?? family.defaults.temperature,
      maxStateTokens: options.maxStateTokens ?? family.defaults.maxStateTokens,
      truncation: options.truncation ?? "cut",
    };

    return new OpenJev({
      model,
      tokenizer,
      family,
      runtime: { model: modelId, family: family.name, ...runtime },
      defaults,
      maxLength: Math.max(
        8,
        Math.floor(options.maxLength ?? family.defaults.maxLength),
      ),
    });
  }

  /**
   * Get model metadata for a configuration without loading it.
   *
   * - `isCached`: whether every required file is present in the browser cache.
   * - `downloadSize`: total size in bytes of the files that will be fetched.
   * - `files`: the file list; `model`, `family`, `device`, `dtype`: the resolved runtime.
   */
  static async info(
    options: Pick<OpenJevOptions, "model" | "device" | "dtype"> = {},
  ): Promise<OpenJevInfo> {
    const modelId = resolveModelId(options.model);
    const config = await AutoConfig.from_pretrained(modelId);
    const family = detectFamily(config);
    const runtime = await resolveRuntime(options, family.webgpuDtype);

    const files = await ModelRegistry.get_files(modelId, {
      config,
      dtype: runtime.dtype,
      device: runtime.device,
      include_tokenizer: true,
      include_processor: false,
    });

    const [isCached, metadata] = await Promise.all([
      ModelRegistry.is_cached(modelId, {
        config,
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

    return {
      model: modelId,
      family: family.name,
      ...runtime,
      isCached,
      downloadSize,
      files,
    };
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
      validateQuestion(
        question,
        isList ? `#${index}` : `"${keys[index]}"`,
        this.family.limits,
      ),
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
    if (this.family.tokenize) {
      return this.family.tokenize({
        state: text,
        questions: [],
        encode: (value) => this.encode(value),
      }).state.length;
    }
    return this.encode(text).length;
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
      const encode = (text: string): number[] => this.encode(text);
      const { state: stateIds, questions: tokenized } = this.family.tokenize
        ? this.family.tokenize({ state, questions, encode })
        : {
            state: encode(state),
            questions: questions.map((question): TokenizedQuestion => ({
              instructions: encode(question.instructions),
              options: questionOptionTexts(question).map(encode),
            })),
          };

      const encoded = this.family.encode({
        state: stateIds,
        questions: tokenized,
        maxStateTokens: settings.maxStateTokens,
        maxLength: this.maxLength,
      });

      if (encoded.stateTruncated && settings.truncation === "error") {
        throw new Error(
          `State was cut to ${encoded.stateTokens} tokens (limit ${settings.maxStateTokens}, ${this.maxLength} in total). Shorten the state or the questions, or set truncation to "cut".`,
        );
      }

      const inputs: Record<string, Tensor> = {
        input_ids: int64(encoded.inputIds),
        attention_mask: int64(encoded.inputIds.map(() => 1)),
      };
      for (const [name, values] of Object.entries(encoded.extraInputs)) {
        inputs[name] = int64(values);
      }

      const { logits } = await this.model(inputs);
      const scores = Array.from(logits.to("float32").data as ArrayLike<number>);

      return questions.map((question, index) =>
        decodeAnswer(
          question,
          encoded.groups[index].map((position) => scores[position]),
          settings.temperature,
        ),
      );
    } catch (error) {
      throw normalizeError(error);
    }
  }

  /** Tokenize caller text (escaped per family, no special tokens). */
  private encode(text: string): number[] {
    const { input_ids } = this.tokenizer(this.family.escape(text), {
      add_special_tokens: false,
    }) as { input_ids: Tensor };
    return Array.from(input_ids.data as ArrayLike<bigint | number>, Number);
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

function int64(values: number[]): Tensor {
  return new Tensor("int64", BigInt64Array.from(values, BigInt), [
    1,
    values.length,
  ]);
}
