import { useEffect, useState } from 'react';

// 静态正文立即可读；组件接管前暂缓输入，避免首次输入被初始化覆盖。
export function useReady(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  return ready;
}
