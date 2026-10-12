import { useState } from 'react';
import { calculateAttention } from '../lib/attention.mjs';

export default function HeroAttention() {
  const [position, setPosition] = useState(1);
  const { weights } = calculateAttention();
  return <div className="hero-figure" aria-label="交互式因果注意力示意">
    <div className="figure-top"><span><i className="live-dot" />交互笔记 No. 04</span><span className="mono">T = 3 · D = 2</span></div>
    <div className="hero-token-row">{['位置 0', '位置 1', '位置 2'].map((label, i) => <button key={label} aria-pressed={position === i} onClick={() => setPosition(i)}><span className="mono">0{i}</span><span>{label}</span></button>)}</div>
    <div className="hero-connectors" aria-hidden="true"><svg viewBox="0 0 300 58" preserveAspectRatio="none">{[50,150,250].map((x,i) => <path key={x} d={`M${x} 0 V15 Q${x} 30 150 30 V58`} className={i <= position ? 'visible-edge' : ''} />)}<circle cx="150" cy="48" r="3" /></svg></div>
    <div className="hero-matrix-wrap"><div className="matrix-bracket left" /><div className="hero-matrix">{weights.flatMap((row,i) => row.map((value,j) => <span key={`${i}-${j}`} className={`hero-cell ${j > i ? 'masked' : ''} ${i === position ? 'active-row' : ''}`} style={j <= i ? { '--cell-opacity': i === position ? .14 + value * .55 : .055 } as React.CSSProperties : undefined}>{j > i ? <span className="masked-dash">—</span> : value.toFixed(2)}</span>))}</div><div className="matrix-bracket right" /><div className="matrix-axis-label">注意力权重</div></div>
    <div className="figure-caption" aria-live="polite"><span className="figure-dot" />位置 {position} 可以读取：{Array.from({ length: position + 1 }, (_, i) => i).join('、')}<span className="caption-tail">未来被遮罩</span></div>
    <a className="figure-link" href="/learn/attention/">展开这一次计算 <span>↗</span></a>
  </div>;
}
