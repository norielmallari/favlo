import { useEffect, useState } from 'react';
import { Platform } from 'react-native';

/**
 * Pixels the on-screen keyboard covers at the bottom of the layout viewport.
 * Chrome/Firefox with ``interactive-widget=resizes-content`` already shrink the
 * layout; this hook covers iOS Safari and any browser that only shrinks the
 * visual viewport. Pinch zoom is ignored.
 */
export function useKeyboardInset(): number {
  const [inset, setInset] = useState(0);

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof window === 'undefined') {
      return;
    }
    const vv = window.visualViewport;
    if (vv == null) {
      return;
    }
    let frame = 0;
    const update = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (vv.scale > 1.01) {
          setInset(0);
          return;
        }
        const layoutH = document.documentElement.clientHeight;
        setInset(Math.max(0, Math.round(layoutH - vv.height - vv.offsetTop)));
      });
    };
    vv.addEventListener('resize', update);
    vv.addEventListener('scroll', update);
    update();
    return () => {
      vv.removeEventListener('resize', update);
      vv.removeEventListener('scroll', update);
      cancelAnimationFrame(frame);
    };
  }, []);

  return inset;
}
