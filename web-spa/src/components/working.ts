/** Soft activity pulse for the Working chip. Reduced motion leaves the dot still. */
export function workingActivity(accent: string): string {
  return `
    @keyframes favlo-working-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.35; }
    }
    [data-testid="working-dot"] {
      background-color: ${accent};
      animation: favlo-working-pulse 2.4s ease-in-out infinite;
    }
    @media (prefers-reduced-motion: reduce) {
      [data-testid="working-dot"] {
        animation: none;
        opacity: 1;
      }
    }
  `;
}
