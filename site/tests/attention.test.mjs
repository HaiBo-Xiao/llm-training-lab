import test from 'node:test';
import assert from 'node:assert/strict';
import { calculateAttention } from '../src/lib/attention.mjs';

function near(actual, expected, tolerance = 1e-5) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} 与参考值 ${expected} 不一致`);
}

test('固定示例与注意力练习的参考数值一致', () => {
  const { weights, output } = calculateAttention();
  const expectedWeights = [[1, 0, 0], [.330238, .669762, 0], [.248255, .248255, .503490]];
  weights.forEach((row, i) => row.forEach((value, j) => near(value, expectedWeights[i][j])));
  [[1, 0], [.330238, .669762], [1.255235, 1.255235]].forEach((row, i) => row.forEach((value, j) => near(output[i][j], value)));
});

test('改变未来 V 不影响前两个位置，当前位置仍可变化', () => {
  const before = calculateAttention(true, 2);
  const after = calculateAttention(true, 100);
  assert.deepEqual(before.output.slice(0, 2), after.output.slice(0, 2));
  assert.notDeepEqual(before.output[2], after.output[2]);
  assert.deepEqual(before.weights, after.weights);
});

test('关闭因果遮罩会允许未来信息影响前面位置', () => {
  const before = calculateAttention(false, 2);
  const after = calculateAttention(false, 100);
  assert.notDeepEqual(before.output[0], after.output[0]);
  assert.notDeepEqual(before.output[1], after.output[1]);
  for (const row of before.weights) near(row.reduce((sum, value) => sum + value, 0), 1);
});
