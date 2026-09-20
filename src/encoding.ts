/**
 * Builds the single-sequence input the open-jev graph expects:
 *
 *   [CLS] [STATE] state [Q] instructions [OPT] option_1 [OPT] option_2 … [Q] … [SEP]
 *
 * plus the span-slot tensor (`seg`) and the per-pair slot ids (`pair_q`,
 * `pair_opt`). See the model card for the layout.
 */

export type MarkerIds = {
  cls: number;
  sep: number;
  state: number;
  q: number;
  opt: number;
};

export type TokenizedQuestion = {
  instructions: number[];
  options: number[][];
};

export type EncodedSequence = {
  inputIds: number[];
  seg: number[];
  pairQ: number[];
  pairOpt: number[];
  /** Pair indices per question, in question order. */
  groups: number[][];
  /** Number of state tokens that made it into the sequence. */
  stateTokens: number;
  /** Whether the state had to be cut. */
  stateTruncated: boolean;
};

export type EncodeParams = {
  state: number[];
  questions: TokenizedQuestion[];
  markers: MarkerIds;
  maxStateTokens: number;
  maxLength: number;
};

export function encodeSequence(params: EncodeParams): EncodedSequence {
  const { state, questions, markers, maxStateTokens, maxLength } = params;

  // [CLS] [STATE] … [SEP]
  let fixedTokens = 3;
  for (const question of questions) {
    fixedTokens += 1 + question.instructions.length;
    for (const option of question.options) {
      fixedTokens += 1 + option.length;
    }
  }

  const budget = maxLength - fixedTokens;
  if (budget < 0) {
    throw new Error(
      `Questions need ${fixedTokens} tokens which exceeds the ${maxLength} token context. Shorten the instructions or options, or ask fewer questions per call.`,
    );
  }

  const stateLimit = Math.min(maxStateTokens, budget);
  const stateTokens = state.slice(0, stateLimit);

  const inputIds: number[] = [markers.cls, markers.state, ...stateTokens];
  const seg: number[] = inputIds.map(() => -1);
  const pairQ: number[] = [];
  const pairOpt: number[] = [];
  const groups: number[][] = [];

  const totalPairs = questions.reduce((n, q) => n + q.options.length, 0);

  questions.forEach((question, questionIndex) => {
    const questionSlot = totalPairs + questionIndex;

    inputIds.push(markers.q, ...question.instructions);
    seg.push(-1, ...question.instructions.map(() => questionSlot));

    const group: number[] = [];
    for (const option of question.options) {
      const pairIndex = pairOpt.length;
      inputIds.push(markers.opt, ...option);
      seg.push(-1, ...option.map(() => pairIndex));
      pairQ.push(questionSlot);
      pairOpt.push(pairIndex);
      group.push(pairIndex);
    }
    groups.push(group);
  });

  inputIds.push(markers.sep);
  seg.push(-1);

  return {
    inputIds,
    seg,
    pairQ,
    pairOpt,
    groups,
    stateTokens: stateTokens.length,
    stateTruncated: stateTokens.length < state.length,
  };
}
