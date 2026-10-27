/** Theme colors for the conversation scrollbar. */
export function transcriptScrollbar(thumb: string, track: string, hover: string): string {
  return `
    [data-testid="transcript"] {
      scrollbar-width: thin;
      scrollbar-color: ${thumb} ${track};
    }
    [data-testid="transcript"]::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    [data-testid="transcript"]::-webkit-scrollbar-track {
      background: ${track};
    }
    [data-testid="transcript"]::-webkit-scrollbar-thumb {
      background-color: ${thumb};
      border: 2px solid ${track};
      border-radius: 999px;
    }
    [data-testid="transcript"]::-webkit-scrollbar-thumb:hover {
      background-color: ${hover};
    }
  `;
}
