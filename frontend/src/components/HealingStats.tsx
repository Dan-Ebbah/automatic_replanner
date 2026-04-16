import { useState, useEffect } from "react";
import type { HealingRecord } from "../types";

interface Props {
  refreshTick: number;
}

export function HealingStats({ refreshTick }: Props) {
  const [record, setRecord] = useState<HealingRecord | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch("/api/healing")
      .then((r) => r.json())
      .then((data: HealingRecord) => setRecord(data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [refreshTick]);

  return (
    <div className="bg-gray-800 rounded p-3 text-xs">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-2">
        Healing Stats
      </h2>
      {loading && <p className="text-gray-500 italic">Loading…</p>}
      {!loading && (!record || !record.available) && (
        <p className="text-gray-500 italic">No healing run yet</p>
      )}
      {!loading && record?.available && (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1">
          <dt className="text-gray-500">Status</dt>
          <dd
            className={
              record.status === "success" ? "text-green-400 font-bold" : "text-red-400 font-bold"
            }
          >
            {record.status}
          </dd>

          <dt className="text-gray-500">Location</dt>
          <dd className="text-gray-200">{record.location}</dd>

          <dt className="text-gray-500">New Weather</dt>
          <dd className="text-gray-200">{record.new_weather}</dd>

          <dt className="text-gray-500">Affected Days</dt>
          <dd className="text-gray-200">
            {record.affected_days?.length
              ? record.affected_days.join(", ")
              : "—"}
          </dd>

          <dt className="text-gray-500">Attempts</dt>
          <dd className="text-gray-200">{record.attempts}</dd>

          <dt className="text-gray-500">Latency</dt>
          <dd className="text-gray-200">{record.latency_ms}ms</dd>

          <dt className="text-gray-500 col-span-2 mt-1 text-gray-600 font-mono">
            {record.run_id}
          </dt>
        </dl>
      )}
    </div>
  );
}
