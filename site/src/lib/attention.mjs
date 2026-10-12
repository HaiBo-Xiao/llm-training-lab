// 数值与 scripts/04_inspect_attention.py 的固定教学示例一致。
export const Q = [[1, 0], [0, 1], [1, 1]];
export const K = [[1, 0], [0, 1], [1, 1]];

export function calculateAttention(causal = true, futureValue = 2) {
  const values = [[1, 0], [0, 1], [futureValue, futureValue]];
  const scores = Q.map(query => K.map(key =>
    query.reduce((sum, value, index) => sum + value * key[index], 0) / Math.sqrt(query.length),
  ));
  const masked = scores.map((row, i) => row.map((value, j) => causal && j > i ? -Infinity : value));
  const weights = masked.map(row => {
    const largest = Math.max(...row);
    const exps = row.map(value => Math.exp(value - largest));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map(value => value / total);
  });
  const output = weights.map(row => values[0].map((_, dimension) =>
    row.reduce((sum, weight, position) => sum + weight * values[position][dimension], 0),
  ));
  return { scores, masked, weights, values, output };
}
