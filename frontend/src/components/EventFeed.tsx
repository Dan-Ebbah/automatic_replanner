import type { WsEvent } from "../types";

const TYPE_STYLES: Record<string, string> = {
  plan_updated: "bg-blue-900 border-blue-500 text-blue-200",
  weather_alert: "bg-yellow-900 border-yellow-500 text-yellow-200",
  replan_started: "bg-purple-900 border-purple-500 text-purple-200",
  replan_complete: "bg-green-900 border-green-500 text-green-200",
  plan_built: "bg-teal-900 border-teal-500 text-teal-200",
  info: "bg-gray-800 border-gray-600 text-gray-300",
};

function formatPayload(event: WsEvent): string {
  switch (event.type) {
    case "plan_updated":
      return `Triggered by ${event.payload.trigger} · ${event.payload.step_count} steps`;
    case "weather_alert":
      return `${event.payload.location} → ${event.payload.new_weather}${event.payload.note ? " (no trip yet)" : ""}`;
    case "replan_started":
      return `Replanning ${event.payload.location} for ${event.payload.weather}…`;
    case "replan_complete":
      return `${event.payload.status.toUpperCase()} · ${event.payload.affected_days.length} day(s) · ${event.payload.latency_ms}ms · ${event.payload.attempts} attempt(s)`;
    case "plan_built":
      return `Day ${event.payload.day_number} (${event.payload.city}) · ${event.payload.action_count} actions`;
    case "info":
      return event.payload.message;
    default:
      return JSON.stringify((event as WsEvent).payload);
  }
}

function formatTime(iso: string): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return iso;
  }
}

interface Props {
  events: WsEvent[];
}

export function EventFeed({ events }: Props) {
  return (
    <div className="flex flex-col h-full">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-2">
        Event Feed
      </h2>
      <div className="flex-1 overflow-y-auto space-y-1 pr-1">
        {events.length === 0 && (
          <p className="text-gray-500 text-sm italic">Waiting for events…</p>
        )}
        {events.map((ev, i) => {
          const style = TYPE_STYLES[ev.type] ?? TYPE_STYLES.info;
          return (
            <div
              key={i}
              className={`border-l-2 rounded px-2 py-1 text-xs ${style}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono font-bold">{ev.type}</span>
                <span className="text-gray-400 shrink-0">{formatTime(ev.timestamp)}</span>
              </div>
              <div className="mt-0.5 opacity-80">{formatPayload(ev)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
