/**
 * Sequence builders for the two model families.
 *
 * open-jev (DeBERTa):
 *   [CLS] [STATE] state [Q] instructions [OPT] option_1 [OPT] option_2 … [Q] … [SEP]
 *   plus the span-slot tensor (`seg`) and per-pair slot ids (`pair_q`, `pair_opt`).
 *
 * kev (Qwen3):
 *   <state> state <q> instructions <opt> option_1 </opt> <opt> option_2 </opt> … <decide> <q> …
 *   only `input_ids` and `attention_mask`; the graph derives the block-causal
 *   mask from the delimiters and returns one logit per token.
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
  /** Extra int64 inputs besides `input_ids` and `attention_mask`. */
  extraInputs: Record<string, number[]>;
  /** Per question: indices into the flat `logits` output, one per option. */
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
    extraInputs: { seg, pair_q: pairQ, pair_opt: pairOpt },
    groups,
    stateTokens: stateTokens.length,
    stateTruncated: stateTokens.length < state.length,
  };
}

export type KevDelimiterIds = {
  state: number;
  question: number;
  optionStart: number;
  optionEnd: number;
  decide: number;
};

export type KevEncodeParams = {
  state: number[];
  questions: TokenizedQuestion[];
  delimiters: KevDelimiterIds;
  maxStateTokens: number;
  /** Limit for the state plus one question branch. */
  maxLength: number;
};

export function encodeKevSequence(params: KevEncodeParams): EncodedSequence {
  const { state, questions, delimiters, maxStateTokens, maxLength } = params;

  const branches = questions.map((question) => {
    const branch = [delimiters.question, ...question.instructions];
    const ends: number[] = [];
    for (const option of question.options) {
      branch.push(delimiters.optionStart, ...option, delimiters.optionEnd);
      ends.push(branch.length - 1);
    }
    branch.push(delimiters.decide);
    return { branch, ends };
  });

  const longestBranch = Math.max(...branches.map((b) => b.branch.length));
  // The <state> delimiter counts towards the branch budget.
  const budget = maxLength - 1 - longestBranch;
  if (budget < 0) {
    throw new Error(
      `A question needs ${longestBranch + 1} tokens which exceeds the ${maxLength} token context. Shorten the instructions or options.`,
    );
  }

  const stateLimit = Math.min(maxStateTokens, budget);
  const stateTokens = state.slice(0, stateLimit);

  const inputIds: number[] = [delimiters.state, ...stateTokens];
  const groups: number[][] = [];

  for (const { branch, ends } of branches) {
    const base = inputIds.length;
    inputIds.push(...branch);
    groups.push(ends.map((end) => base + end));
  }

  return {
    inputIds,
    extraInputs: {},
    groups,
    stateTokens: stateTokens.length,
    stateTruncated: stateTokens.length < state.length,
  };
}

/**
 * gliner2 (GLiNER2.5-Decide, DeBERTa-v3-large):
 *   ( [P] prompt ( [L] option_1 [L] option_2 … ) ) [SEP_STRUCT] ( [P] … ) [SEP_TEXT] word word …
 *   plus `marker_positions`, the index of every [L] token; the graph returns
 *   one logit per marker. `prompt` is the instructions with option descriptions
 *   appended as ` [DESCRIPTION] option: description`; the state is lowercased
 *   and tokenized word by word (see `gliner2Family`).
 */

export type Gliner2MarkerIds = {
  p: number;
  l: number;
  sepStruct: number;
  sepText: number;
  /** Token ids of "(" and ")" as standalone words. */
  open: number[];
  close: number[];
};

export type Gliner2EncodeParams = {
  state: number[];
  questions: TokenizedQuestion[];
  markers: Gliner2MarkerIds;
  maxStateTokens: number;
  maxLength: number;
};

export function encodeGliner2Sequence(
  params: Gliner2EncodeParams,
): EncodedSequence {
  const { state, questions, markers, maxStateTokens, maxLength } = params;

  const inputIds: number[] = [];
  const markerPositions: number[] = [];
  const groups: number[][] = [];

  questions.forEach((question, questionIndex) => {
    if (questionIndex > 0) {
      inputIds.push(markers.sepStruct);
    }
    inputIds.push(...markers.open, markers.p, ...question.instructions);
    inputIds.push(...markers.open);
    const group: number[] = [];
    for (const option of question.options) {
      group.push(markerPositions.length);
      markerPositions.push(inputIds.length);
      inputIds.push(markers.l, ...option);
    }
    groups.push(group);
    inputIds.push(...markers.close, ...markers.close);
  });

  inputIds.push(markers.sepText);

  const budget = maxLength - inputIds.length;
  if (budget < 0) {
    throw new Error(
      `Questions need ${inputIds.length} tokens which exceeds the ${maxLength} token context. Shorten the instructions or options, or ask fewer questions per call.`,
    );
  }

  const stateLimit = Math.min(maxStateTokens, budget);
  const stateTokens = state.slice(0, stateLimit);
  inputIds.push(...stateTokens);

  return {
    inputIds,
    extraInputs: { marker_positions: markerPositions },
    groups,
    stateTokens: stateTokens.length,
    stateTruncated: stateTokens.length < state.length,
  };
}
