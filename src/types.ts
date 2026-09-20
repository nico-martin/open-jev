import type { ProgressInfo } from "@huggingface/transformers";

/** ONNX weight variants published for the open-jev model. */
export type OpenJevDtype = "fp32" | "fp16" | "q4" | "q4f16";

/** Execution backends: `webgpu`/`wasm` in the browser, `cpu` in Node.js. */
export type OpenJevDevice = "webgpu" | "wasm" | "cpu";

/** Options that control how a single `decide()` call is encoded and scored. */
export type DecideOptions = {
  /**
   * Softmax temperature. Defaults to the calibrated value shipped with the
   * model (`1.05`).
   */
  temperature?: number;
  /**
   * Maximum number of state tokens kept before the questions (default `256`).
   */
  maxStateTokens?: number;
  /**
   * What to do when the state does not fit: `"cut"` (default) drops trailing
   * state tokens, `"error"` throws instead.
   */
  truncation?: "cut" | "error";
};

export type OpenJevOptions = DecideOptions & {
  /**
   * Hugging Face model id (or a local/self-hosted path understood by
   * Transformers.js). Defaults to `onnx-community/open-jev-deberta-v3-large-ONNX`.
   */
  model?: string;
  /**
   * Weight variant. `"auto"` (default) picks `fp16` on WebGPU with
   * `shader-f16` support and `q4` everywhere else.
   */
  dtype?: OpenJevDtype | "auto";
  /**
   * Backend. `"auto"` (default) picks `webgpu` when available, `cpu` in
   * Node.js, else `wasm`.
   */
  device?: OpenJevDevice | "auto";
  /** Total sequence length limit (default `512`). */
  maxLength?: number;
  /** Called with download progress while files are fetched. */
  onProgress?: (progress: LoadProgress) => void;
};

export type OpenJevRuntime = {
  device: OpenJevDevice;
  dtype: OpenJevDtype;
};

export type OpenJevInfo = OpenJevRuntime & {
  /** Whether every required file is present in the browser cache. */
  isCached: boolean;
  /** Sum of all required file sizes in bytes (model weights + tokenizer + config). */
  downloadSize: number;
  /** Remote files Transformers.js will fetch for this configuration. */
  files: string[];
};

export type LoadProgress = {
  /** Fraction of bytes fetched so far (`0..1`). */
  progress: number;
  /** Bytes fetched so far. */
  loaded: number;
  /** Total bytes to fetch. */
  total: number;
};

/** Pick one option out of up to 255. */
export type ChoiceQuestion<O extends string = string> = {
  type: "choice";
  instructions: string;
  options: readonly O[];
};

/** Rate on an ordered scale of 2 to 10 levels (first = lowest). */
export type ScoreQuestion<L extends string = string> = {
  type: "score";
  instructions: string;
  options: readonly L[];
};

/** Yes/no statement about the state. */
export type NoulQuestion = {
  type: "noul";
  instructions: string;
};

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

export type ChoiceAnswer<O extends string = string> = {
  type: "choice";
  /** The option with the highest probability. */
  choice: O;
  /** Probability of `choice` (`0..1`). */
  confidence: number;
  /** Full distribution over the options. */
  probabilities: Record<O, number>;
};

export type ScoreAnswer<L extends string = string> = {
  type: "score";
  /** Expected level index (`0..levels-1`, may fall between levels). */
  score: number;
  /** `score` rescaled to `0..1`. */
  normalized: number;
  /** Level label closest to `score`. */
  level: L;
  /** Highest single-level probability (`0..1`). */
  confidence: number;
  /** Full distribution over the levels. */
  probabilities: Record<L, number>;
};

export type NoulAnswer = {
  type: "noul";
  /** `true` when `probability >= 0.5`. */
  answer: boolean;
  /** Probability that the statement holds, p(yes). */
  probability: number;
  /** `max(p(yes), p(no))`. */
  confidence: number;
};

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export type AnswerFor<Q extends Question> =
  Q extends ChoiceQuestion<infer O>
    ? ChoiceAnswer<O>
    : Q extends ScoreQuestion<infer L>
      ? ScoreAnswer<L>
      : Q extends NoulQuestion
        ? NoulAnswer
        : never;

export type QuestionList = readonly Question[];
export type QuestionMap = Readonly<Record<string, Question>>;
export type Questions = QuestionList | QuestionMap;

export type AnswersFor<Qs extends Questions> = Qs extends QuestionList
  ? {
      -readonly [K in keyof Qs]: Qs[K] extends Question
        ? AnswerFor<Qs[K]>
        : Qs[K];
    }
  : Qs extends QuestionMap
    ? {
        -readonly [K in keyof Qs]: Qs[K] extends Question
          ? AnswerFor<Qs[K]>
          : never;
      }
    : never;

export type ModelProgressCallback = (info: ProgressInfo) => void;
