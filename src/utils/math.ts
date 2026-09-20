export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Numerically stable softmax with temperature scaling. */
export function softmax(logits: number[], temperature = 1): number[] {
  if (logits.length === 0) {
    return [];
  }

  const max = Math.max(...logits);
  const scaled = logits.map((x) => Math.exp((x - max) / temperature));
  const sum = scaled.reduce((a, b) => a + b, 0);
  return scaled.map((x) => x / sum);
}
