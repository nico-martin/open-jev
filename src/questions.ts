import type {
  ChoiceQuestion,
  NoulQuestion,
  Question,
  ScoreQuestion,
} from "./types";

/** Options the models were trained with for `noul` questions (index 1 = yes). */
export const NOUL_OPTIONS = ["no", "yes"] as const;

export type QuestionLimits = {
  minChoiceOptions: number;
  maxChoiceOptions: number;
  minScoreLevels: number;
  maxScoreLevels: number;
};

/** Build a `choice` question: pick one of the given options. */
export function choice<const O extends string>(
  instructions: string,
  options: readonly O[],
  descriptions?: Partial<Record<O, string>>,
): ChoiceQuestion<O> {
  return descriptions
    ? { type: "choice", instructions, options, descriptions }
    : { type: "choice", instructions, options };
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

/** Option labels of a question (keys of the answer distribution). */
export function questionLabels(question: Question): readonly string[] {
  return question.type === "noul" ? NOUL_OPTIONS : question.options;
}

/** Option texts as they are fed to the model (`name: description` when given). */
export function questionOptionTexts(question: Question): string[] {
  if (question.type === "noul") {
    return [...NOUL_OPTIONS];
  }

  if (question.type === "choice" && question.descriptions) {
    const descriptions = question.descriptions as Record<string, string>;
    return question.options.map((option) => {
      const description = descriptions[option];
      return description ? `${option}: ${description}` : option;
    });
  }

  return [...question.options];
}

export function validateQuestion(
  question: Question,
  label: string,
  limits: QuestionLimits,
): void {
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

  const min =
    question.type === "choice"
      ? limits.minChoiceOptions
      : limits.minScoreLevels;
  const max =
    question.type === "choice"
      ? limits.maxChoiceOptions
      : limits.maxScoreLevels;
  if (options.length < min || options.length > max) {
    throw new Error(
      `Question ${label} (${question.type}) needs between ${min} and ${max} options for this model, got ${options.length}.`,
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
