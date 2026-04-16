import type { ItineraryResponse, DayPlanResponse, ActivityAction } from "../types";

function getActionNames(actions: ActivityAction[]): Set<string> {
  return new Set(actions.map((a) => a.name));
}

function DayTable({
  day,
  snapshotDay,
}: {
  day: DayPlanResponse;
  snapshotDay?: DayPlanResponse;
}) {
  const currentNames = getActionNames(day.actions);
  const snapshotNames = snapshotDay ? getActionNames(snapshotDay.actions) : new Set<string>();

  const added = snapshotDay
    ? new Set([...currentNames].filter((n) => !snapshotNames.has(n)))
    : new Set<string>();
  const removed = snapshotDay
    ? new Set([...snapshotNames].filter((n) => !currentNames.has(n)))
    : new Set<string>();

  const allActions: ActivityAction[] = snapshotDay
    ? [
        ...day.actions,
        ...(snapshotDay.actions.filter((a) => !currentNames.has(a.name))),
      ]
    : day.actions;

  return (
    <div className="mb-4">
      <h3 className="text-xs font-semibold text-gray-400 mb-1">
        Day {day.day_number} · {day.city}{" "}
        <span className="font-normal opacity-60">({day.weather})</span>
      </h3>
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="text-gray-500 border-b border-gray-700">
            <th className="text-left py-0.5 pr-2">Activity</th>
            <th className="text-right py-0.5 pr-2">Duration</th>
            <th className="text-right py-0.5">Weather</th>
          </tr>
        </thead>
        <tbody>
          {allActions.map((action, i) => {
            const isAdded = added.has(action.name);
            const isRemoved = removed.has(action.name);
            let rowClass = "border-b border-gray-800 ";
            if (isAdded) rowClass += "text-green-400";
            else if (isRemoved) rowClass += "text-red-400 line-through opacity-60";
            else rowClass += "text-gray-200";

            return (
              <tr key={i} className={rowClass}>
                <td className="py-0.5 pr-2">
                  {action.name.replace(/_/g, " ")}
                  {isAdded && <span className="ml-1 text-green-500 text-xs">+</span>}
                  {isRemoved && <span className="ml-1 text-red-500 text-xs">−</span>}
                </td>
                <td className="text-right py-0.5 pr-2">{action.duration_minutes}m</td>
                <td className="text-right py-0.5">{action.weather_requirement}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

interface Props {
  currentPlan: ItineraryResponse | null;
  snapshotPlan: ItineraryResponse | null;
  replanInFlight: boolean;
}

export function DayPlanPanel({ currentPlan, snapshotPlan, replanInFlight }: Props) {
  if (!currentPlan || !currentPlan.trip_planned) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm italic">
        No itinerary yet
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-2 flex items-center gap-2">
        Day Plans
        {replanInFlight && (
          <span className="text-purple-400 text-xs animate-pulse">replanning…</span>
        )}
      </h2>
      <div className="flex-1 overflow-y-auto">
        {currentPlan.days.map((day) => {
          const snapshotDay = snapshotPlan?.days.find(
            (d) => d.day_number === day.day_number
          );
          return (
            <DayTable key={day.day_number} day={day} snapshotDay={snapshotDay} />
          );
        })}
      </div>
    </div>
  );
}
