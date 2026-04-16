import { useState } from "react";

interface Props {
  onInjected?: (condition: string) => void;
}

export function WeatherControls({ onInjected }: Props) {
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function inject(condition: "rainy" | "sunny") {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/weather", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ condition }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setLast(condition);
      onInjected?.(condition);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="bg-gray-800 rounded p-3">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-2">
        Weather Controls
      </h2>
      <div className="flex gap-2">
        <button
          disabled={busy}
          onClick={() => inject("rainy")}
          className="flex-1 py-1.5 rounded text-xs font-semibold bg-blue-700 hover:bg-blue-600 disabled:opacity-50 text-white transition-colors"
        >
          Inject Rain
        </button>
        <button
          disabled={busy}
          onClick={() => inject("sunny")}
          className="flex-1 py-1.5 rounded text-xs font-semibold bg-yellow-600 hover:bg-yellow-500 disabled:opacity-50 text-white transition-colors"
        >
          Inject Sun
        </button>
      </div>
      {last && (
        <p className="mt-1 text-xs text-gray-400">
          Last injected: <span className="text-gray-200">{last}</span>
        </p>
      )}
      {error && <p className="mt-1 text-xs text-red-400">{error}</p>}
    </div>
  );
}
