import { useState, useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";

import { useWebSocket } from "./hooks/useWebSocket";
import { useItinerary } from "./hooks/useItinerary";

import { EventFeed } from "./components/EventFeed";
import { PlanGraph } from "./components/PlanGraph";
import { DayPlanPanel } from "./components/DayPlanPanel";
import { HealingStats } from "./components/HealingStats";
import { WeatherControls } from "./components/WeatherControls";

const STATUS_DOT: Record<string, string> = {
  open: "bg-green-500",
  connecting: "bg-yellow-500 animate-pulse",
  closed: "bg-red-500",
  error: "bg-red-500",
};

export default function App() {
  const { events, status } = useWebSocket();
  const latestEvent = events[0];

  const { currentPlan, snapshotPlan, replanInFlight } = useItinerary(latestEvent);

  const [healingRefreshTick, setHealingRefreshTick] = useState(0);

  // Bump healing stats whenever a fresh replan_complete arrives
  useEffect(() => {
    if (latestEvent?.type === "replan_complete") {
      setHealingRefreshTick((t) => t + 1);
    }
  }, [latestEvent]);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700 shrink-0">
        <h1 className="text-sm font-bold tracking-widest text-gray-100 uppercase">
          AEGIS Dashboard
        </h1>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className={`w-2 h-2 rounded-full ${STATUS_DOT[status]}`} />
          WebSocket {status}
        </div>
      </header>

      {/* Main 3-column layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left column — Event Feed */}
        <aside className="w-72 shrink-0 flex flex-col p-3 border-r border-gray-700 overflow-hidden">
          <EventFeed events={events} />
        </aside>

        {/* Center column — Planning Graph */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <div className="p-3 pb-0 shrink-0">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest">
              Planning Graph
              {replanInFlight && (
                <span className="ml-2 text-purple-400 text-xs animate-pulse">
                  animating…
                </span>
              )}
            </h2>
          </div>
          <div className="flex-1 overflow-hidden">
            <ReactFlowProvider>
              <PlanGraph plan={currentPlan} replanInFlight={replanInFlight} />
            </ReactFlowProvider>
          </div>
        </main>

        {/* Right column — Day Plans + Stats + Controls */}
        <aside className="w-80 shrink-0 flex flex-col gap-3 p-3 border-l border-gray-700 overflow-hidden">
          <div className="flex-1 overflow-hidden">
            <DayPlanPanel
              currentPlan={currentPlan}
              snapshotPlan={snapshotPlan}
              replanInFlight={replanInFlight}
            />
          </div>
          <HealingStats refreshTick={healingRefreshTick} />
          <WeatherControls />
        </aside>
      </div>
    </div>
  );
}
