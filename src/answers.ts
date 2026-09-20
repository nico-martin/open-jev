import { NOUL_OPTIONS } from "./questions";
import type {
  Answer,
  ChoiceAnswer,
  NoulAnswer,
  Question,
  ScoreAnswer,
} from "./types";
import { softmax } from "./utils/math";

/** Turn the raw logits of one question's option group into a typed answer. */
export function decodeAnswer(
  question: Question,
  logits: number[],
  temperature: number,
): Answer {
  const probabilities = softmax(logits, temperature);

  switch (question.type) {
    case "choice":
      return decodeChoice(question.options, probabilities);
    case "score":
      return decodeScore(question.options, probabilities);
    case "noul":
      return decodeNoul(probabilities);
  }
}

function decodeChoice(
  options: readonly string[],
  probabilities: number[],
): ChoiceAnswer {
  const best = argmax(probabilities);
  return {
    type: "choice",
    choice: options[best],
    confidence: probabilities[best],
    probabilities: toRecord(options, probabilities),
  };
}

function decodeScore(
  levels: readonly string[],
  probabilities: number[],
): ScoreAnswer {
  const expected = probabilities.reduce((sum, p, index) => sum + p * index, 0);
  const best = argmax(probabilities);
  const last = levels.length - 1;
  const nearest = Math.min(last, Math.max(0, Math.round(expected)));

  return {
    type: "score",
    score: expected,
    normalized: last > 0 ? expected / last : 0,
    level: levels[nearest],
    confidence: probabilities[best],
    probabilities: toRecord(levels, probabilities),
  };
}

function decodeNoul(probabilities: number[]): NoulAnswer {
  const yesIndex = NOUL_OPTIONS.indexOf("yes");
  const probability = probabilities[yesIndex];
  return {
    type: "noul",
    answer: probability >= 0.5,
    probability,
    confidence: Math.max(probability, 1 - probability),
  };
}

function argmax(values: number[]): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) {
      best = index;
    }
  }
  return best;
}

function toRecord(
  keys: readonly string[],
  values: number[],
): Record<string, number> {
  const record: Record<string, number> = {};
  keys.forEach((key, index) => {
    record[key] = values[index];
  });
  return record;
}
