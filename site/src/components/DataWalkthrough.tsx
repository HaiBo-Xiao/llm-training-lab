import { useState, type CSSProperties } from 'react';
import { useReady } from '../lib/useReady';

export function TokenLab() {
  const ready = useReady();
  const [text, setText] = useState('小猫吃鱼。');
  const [position, setPosition] = useState(1);
  const vocab = ['。', '吃', '小', '猫', '鱼'];
  const chars = Array.from(text);
  const invalid = chars.filter(char => !vocab.includes(char));
  const valid = chars.length >= 2 && chars.length <= 12 && invalid.length === 0;
  const selected = Math.min(position, chars.length-2);
  return <div className="lab mini-lab" inert={!ready} aria-busy={!ready}><div className="lab-heading"><span><i className="live-dot" />字符与目标</span><span className="mono">CHARACTER-LEVEL</span></div><div className="mini-lab-body"><label className="field-label" htmlFor="token-input">输入一段练习词表中的文字</label><input id="token-input" className="text-input" value={text} maxLength={12} onChange={event => { setText(event.target.value); setPosition(0); }} /><div className="vocab-strip">{vocab.map((char,id) => <span key={char}>{char}<small className="mono">{id}</small></span>)}</div>{!valid ? <p role="alert" className="input-error">请输入 2～12 个字符，只使用「。吃小猫鱼」中的字符。</p> : <><div className="token-line"><span className="token-line-label mono">x</span><div className="token-tiles">{chars.slice(0,-1).map((char,i) => <button key={i} className={i === selected ? 'active-token' : ''} aria-label={`查看位置 ${i} 的预测任务`} aria-pressed={i === selected} onClick={()=>setPosition(i)}><span>{char}</span><small className="mono">{vocab.indexOf(char)}</small></button>)}</div></div><div className="token-line"><span className="token-line-label mono">y</span><div className="token-tiles">{chars.slice(1).map((char,i) => <div key={i} className={i === selected ? 'target-token' : ''}><span>{char}</span><small className="mono">{vocab.indexOf(char)}</small></div>)}</div></div><div className="mini-result" aria-live="polite"><span>位置 {selected} 的预测任务</span><strong>「{chars.slice(0,selected+1).join('')}」 <span>→</span> 「{chars[selected+1]}」</strong></div></>}</div><div className="lab-source">固定词表来自原始练习 · 点击输入位置查看对应目标</div></div>;
}

export function BatchLab() {
  const ready = useReady();
  const [start,setStart] = useState(0);
  const ids = Array.from({length:13},(_,i)=>i+10);
  const rows = [0,1].map(b=>ids.slice((start+b)*3,(start+b)*3+4));
  return <div className="lab mini-lab" inert={!ready} aria-busy={!ready}><div className="lab-heading"><span><i className="live-dot" />切分与组批</span><span className="mono">B = 2 · T = 3</span></div><div className="mini-lab-body"><div className="field-label">原始 ID 序列</div><div className="sequence-strip">{ids.map((id,i)=><span key={id} className={i>=start*3 && i<=start*3+6 ? 'sequence-selected' : ''}>{id}</span>)}</div><label className="batch-select">起始样本编号 k₀<select value={start} onChange={event=>setStart(Number(event.target.value))}><option value={0}>0 · 从 ID 10 开始</option><option value={1}>1 · 从 ID 13 开始</option><option value={2}>2 · 从 ID 16 开始</option></select></label><div className="batch-arrays"><div><span className="field-label">输入 xb</span>{rows.map((row,i)=><div key={i} className="array-row"><small>样本 {start+i}</small><span className="mono">[ {row.slice(0,-1).join(', ')} ]</span></div>)}</div><div><span className="field-label">目标 yb</span>{rows.map((row,i)=><div key={i} className="array-row target-array"><small>样本 {start+i}</small><span className="mono">[ {row.slice(1).join(', ')} ]</span></div>)}</div></div><div className="mini-result" aria-live="polite"><span>两条样本共用边界 ID {rows[0][3]}</span><strong className="mono">shape = [2, 3]</strong></div></div><div className="lab-source">独立手算示例 · 原练习实际配置为 B = 4、T = 8</div></div>;
}

export function ForwardLab() {
  const ready = useReady();
  const [id,setId] = useState(1);
  const vocab = ['。','小','猫','鱼'];
  const table = [[0,0],[1,0],[0,1],[1,1]];
  const weight = [[0,1,2,-1],[0,-1,1,2]];
  const h = table[id];
  const logits = vocab.map((_,i)=>h[0]*weight[0][i]+h[1]*weight[1][i]);
  const pred = logits.indexOf(Math.max(...logits));
  return <div className="lab mini-lab" inert={!ready} aria-busy={!ready}><div className="lab-heading"><span><i className="live-dot" />一次前向计算</span><span className="mono">D = 2 · V = 4</span></div><div className="mini-lab-body"><div className="field-label">选择一个输入字符</div><div className="forward-tokens">{vocab.map((char,i)=><button key={char} aria-pressed={id===i} onClick={()=>setId(i)}><span>{char}</span><small className="mono">ID {i}</small></button>)}</div><div className="embedding-strip"><span>取出 E 的第 {id} 行</span><span aria-hidden="true">→</span><strong className="mono">h = [{h.join(', ')}]</strong></div><div className="logit-chart" aria-label="每个候选字符的 logits">{vocab.map((char,i)=><div className="logit-row" key={char}><span>{char}</span><div className="logit-track"><span className={logits[i]<0?'negative-logit':''} style={{left:logits[i]<0?`${25+logits[i]*25}%`:'25%',width:`${Math.abs(logits[i])*25}%`} as CSSProperties}/></div><strong className="mono">{logits[i].toFixed(1)}</strong></div>)}</div><div className="mini-result" aria-live="polite"><span>{id===0?'分数相同时，argmax 返回第一个最大值':'取候选分数最大的 ID'}</span><strong>预测：{vocab[pred]} <span className="mono">/ ID {pred}</span></strong></div></div><div className="lab-source">固定 E、W 来自练习 · 这里显示分数，尚未转换为概率</div></div>;
}
