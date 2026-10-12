import { useEffect, useId, useState, type CSSProperties } from 'react';
import { calculateAttention } from '../lib/attention.mjs';
import { useReady } from '../lib/useReady';

const steps = [
  { label: '计算分数', short: 'QKᵀ / √D', description: '每个 Q 与所有 K 做点积，再除以 √2。此时，未来位置仍有分数。' },
  { label: '应用遮罩', short: 'Causal mask', description: '把未来位置的分数替换成负无穷；当前位置与过去的位置保持原值。' },
  { label: '归一化权重', short: 'Softmax', description: '对每一行做 softmax。可见位置的权重和为 1，未来位置的权重为 0。' },
  { label: '汇总信息', short: 'Weights × V', description: '每一行权重分别乘上各位置的 V，再相加，得到这个位置的输出向量。' },
];

export default function AttentionLab() {
  const ready = useReady();
  const [step, setStep] = useState(2);
  const [row, setRow] = useState(1);
  const [causal, setCausal] = useState(true);
  const [future, setFuture] = useState(2);
  const [playing, setPlaying] = useState(false);
  const id = useId();
  const data = calculateAttention(causal, future);
  const baseline = calculateAttention(causal, 2);
  const matrix = step === 0 ? data.scores : step === 1 ? data.masked : data.weights;
  const current = steps[step];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => {
      if (step === 3) setPlaying(false);
      else setStep(value => value + 1);
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [playing, step]);

  function selectStep(value: number) { setStep(value); setPlaying(false); }
  function reset() { setStep(2); setRow(1); setCausal(true); setFuture(2); setPlaying(false); }

  return <div className="attention-lab lab" aria-label="因果注意力交互实验" inert={!ready} aria-busy={!ready}>
    <div className="lab-heading"><span><i className="live-dot" />交互实验</span><span className="mono">固定 Q / K · T = 3 · D = 2</span></div>
    <div className="lab-steps" role="group" aria-label="计算步骤">{steps.map((item, index) => <button key={item.label} aria-pressed={step === index} onClick={() => selectStep(index)}><span className="mono">0{index + 1}</span><span>{item.label}</span></button>)}</div>
    <div className="lab-stage-head"><div><span className="eyebrow">STEP 0{step + 1}</span><h3>{current.label}</h3></div><span className="formula-chip mono">{current.short}</span></div>
    <p className="lab-description" aria-live="polite">{!causal && step === 1 ? '遮罩已关闭，未来位置的分数也被保留。可以和开启遮罩时对照。' : !causal && step > 1 ? '遮罩已关闭，每个位置都能读取未来。此时上三角权重也会参与输出。' : current.description}</p>
    <div className="attention-workspace">
      <div className="attention-matrix-area"><div className="matrix-label">列：被读取的位置</div><table className="attention-matrix" aria-label={`${current.label}矩阵`}><thead><tr><th><span className="sr-only">查询位置</span></th>{[0,1,2].map(j => <th key={j} scope="col">{j}</th>)}</tr></thead><tbody>{matrix.map((values,i) => <tr key={i} className={row === i ? 'selected-matrix-row' : ''}><th scope="row"><button aria-label={`选择查询位置 ${i}`} aria-pressed={row === i} onClick={() => setRow(i)}>{i}<span aria-hidden="true">→</span></button></th>{values.map((value,j) => {
        const blocked = causal && step > 0 && j > i;
        const opacity = Number.isFinite(value) ? .06 + (step < 2 ? value / Math.SQRT2 : value) * .36 : 0;
        return <td key={j}><button onClick={() => setRow(i)} aria-label={`查询 ${i} 读取位置 ${j}：${Number.isFinite(value) ? value.toFixed(3) : '负无穷'}${blocked ? '，已遮罩' : ''}`} className={`attention-cell ${blocked ? 'blocked' : ''}`} style={{ '--heat': opacity } as CSSProperties}><span>{Number.isFinite(value) ? value.toFixed(3) : '−∞'}</span></button></td>;
      })}</tr>)}</tbody></table><div className="matrix-legend"><span><i className="legend-swatch" />可读取</span><span><i className="legend-swatch hatched" />未来位置</span><span>点击行号选择位置</span></div></div>
      <div className="attention-results"><div className="result-label">{step === 3 ? '加权后的输出' : `聚焦查询位置 ${row}`}</div>{step === 3 ? <div className="output-vectors">{data.output.map((vector,i) => { const changed = vector.some((value,d) => Math.abs(value - baseline.output[i][d]) > .0001); return <div key={i} data-output-row={i} className={`output-vector ${row === i ? 'selected-output' : ''} ${changed ? 'changed-output' : ''}`}><span>位置 {i}</span><strong className="mono">[{vector.map(x => x.toFixed(3)).join(', ')}]</strong><small>{future === 2 ? '原始结果' : changed ? '输出变化' : '保持不变'}</small></div>; })}</div> : <div className="row-explanation"><span className="big-position mono">0{row}</span><p>{step === 0 ? '与全部 3 个键向量比较' : causal ? `可以读取位置 ${Array.from({length:row+1},(_,i)=>i).join('、')}` : '可读取全部 3 个位置'}</p><div className="row-values mono">[{matrix[row].map(value => Number.isFinite(value) ? value.toFixed(3) : '−∞').join(', ')}]</div><small>{step === 2 ? `这一行的权重和 = ${data.weights[row].reduce((sum,x)=>sum+x,0).toFixed(3)}` : step === 1 && causal ? `已屏蔽 ${2-row} 个未来位置` : '同一行对应同一个查询'}</small></div>}</div>
    </div>
    <div className="lab-playback"><button className="lab-button" onClick={() => { if (playing) setPlaying(false); else { setStep(0); setPlaying(true); } }}><span aria-hidden="true">{playing ? 'Ⅱ' : '▷'}</span>{playing ? '暂停播放' : '播放完整过程'}</button><div><button className="lab-small-button" aria-label="上一步" disabled={step === 0} onClick={() => selectStep(step-1)}>←</button><span className="mono">{step + 1} / 4</span><button className="lab-small-button" aria-label="下一步" disabled={step === 3} onClick={() => selectStep(step+1)}>→</button></div></div>
    <div className="lab-experiment"><div className="experiment-control-heading"><span>试着改变一个条件</span><button className="reset-button" onClick={reset}>重置</button></div><label className="mask-control"><input type="checkbox" checked={causal} onChange={event => { setCausal(event.target.checked); setPlaying(false); }} /><span>启用因果遮罩</span><small>{causal ? '每个位置只能读取自己和过去' : '对照：允许读取未来'}</small></label><div className="range-control"><label htmlFor={id}>位置 2 的值向量 <span className="mono">V₂ = [{future}, {future}]</span></label><input id={id} aria-label="位置 2 的 V 值" type="range" min="2" max="100" step="1" value={future} onChange={event => { setFuture(Number(event.target.value)); setStep(3); setPlaying(false); }} /><div className="range-ticks"><span>2 · 原始值</span><span>100 · 大幅改变</span></div></div><p className="experiment-insight" aria-live="polite">{future === 2 ? '把滑块向右拖动，观察前两个位置的输出。' : causal ? '位置 0、1 的输出不变。因果遮罩阻止了未来 V 的影响。' : '前两个位置的输出也变了。关闭遮罩后，未来 V 参与了计算。'}</p></div>
    <div className="lab-source">教学示例 · 数值来自 <a href="/source/attention/">04_inspect_attention.py ↗</a></div>
  </div>;
}
