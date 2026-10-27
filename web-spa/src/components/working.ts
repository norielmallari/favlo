/** A light band sweeping across the Working label. Reduced motion leaves the word still. */
export function workingShimmer(base: string): string {
  return `
    @keyframes favlo-working-shimmer {
      from { background-position: -100% center; }
      to { background-position: 200% center; }
    }
    [data-testid="working"] {
      background-image: linear-gradient(
        90deg,
        ${base} 0%,
        ${base} 40%,
        color-mix(in srgb, ${base} 25%, white) 50%,
        ${base} 60%,
        ${base} 100%
      );
      background-size: 200% 100%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      color: transparent;
      animation: favlo-working-shimmer 2s linear infinite;
    }
    @media (prefers-reduced-motion: reduce) {
      [data-testid="working"] {
        animation: none;
        background-image: none;
        -webkit-text-fill-color: ${base};
        color: ${base};
      }
    }
  `;
}
