import type {
  ChoiceQuestion,
  NoulQuestion,
  Question,
  ScoreQuestion,
} from "./types";

export const MAX_CHOICE_OPTIONS = 255;
export const MIN_SCORE_LEVELS = 2;
export const MAX_SCORE_LEVELS = 10;

/** Options the model was trained with for `noul` questions (index 1 = yes). */
export const NOUL_OPTIONS = ["no", "yes"] as const;

/** Build a `choice` question: pick one of the given options. */
export function choice<const O extends string>(
  instructions: string,
  options: readonly O[],
): ChoiceQuestion<O> {
  return { type: "choice", instructions, options };
}

/** Build a `score` question: rate on an ordered scale (first = lowest). */
export function score<const L extends string>(
  instructions: string,
  levels: readonly L[],
): ScoreQuestion<L> {
  return { type: "score", instructions, options: levels };
}

/** Build a `noul` question: does the statement hold for the state? */
export function noul(statement: string): NoulQuestion {
  return { type: "noul", instructions: statement };
}

/** Options for a question as they are fed to the model. */
export function questionOptions(question: Question): readonly string[] {
  return question.type === "noul" ? NOUL_OPTIONS : question.options;
}

export function validateQuestion(question: Question, label: string): void {
  if (!question || typeof question !== "object") {
    throw new Error(`Question ${label} must be an object.`);
  }

  if (
    typeof question.instructions !== "string" ||
    question.instructions.trim() === ""
  ) {
    throw new Error(`Question ${label} needs non-empty instructions.`);
  }

  if (question.type === "noul") {
    return;
  }

  if (question.type !== "choice" && question.type !== "score") {
    throw new Error(
      `Question ${label} has unknown type "${String(
        (question as { type: unknown }).type,
      )}". Expected "choice", "score" or "noul".`,
    );
  }

  const options = question.options;
  if (!Array.isArray(options)) {
    throw new Error(`Question ${label} needs an options array.`);
  }

  const min = question.type === "choice" ? 2 : MIN_SCORE_LEVELS;
  const max =
    question.type === "choice" ? MAX_CHOICE_OPTIONS : MAX_SCORE_LEVELS;
  if (options.length < min || options.length > max) {
    throw new Error(
      `Question ${label} (${question.type}) needs between ${min} and ${max} options, got ${options.length}.`,
    );
  }

  const seen = new Set<string>();
  for (const option of options) {
    if (typeof option !== "string" || option.trim() === "") {
      throw new Error(`Question ${label} has an empty or non-string option.`);
    }
    if (seen.has(option)) {
      throw new Error(`Question ${label} has a duplicate option "${option}".`);
    }
    seen.add(option);
  }
}
