import { useState, useEffect, useCallback, useRef } from "react";
import type { ItineraryResponse, WsEvent } from "../types";

const POLL_MS = 5000;

export function useItinerary(latestEvent: WsEvent | undefined) {
  const [currentPlan, setCurrentPlan] = useState<ItineraryResponse | null>(null);
  const [snapshotPlan, setSnapshotPlan] = useState<ItineraryResponse | null>(null);
  const [replanInFlight, setReplanInFlight] = useState(false);
  const lastEventTypeRef = useRef<string | undefined>(undefined);

  const fetchItinerary = useCallback(async () => {
    try {
      const res = await fetch("/api/itinerary");
      if (res.ok) {
        const data: ItineraryResponse = await res.json();
        setCurrentPlan(data);
      }
    } catch {
      // network error — ignore, will retry
    }
  }, []);

  // Poll regularly
  useEffect(() => {
    fetchItinerary();
    const id = setInterval(fetchItinerary, POLL_MS);
    return () => clearInterval(id);
  }, [fetchItinerary]);

  // React to WS events for snapshot + replan state
  useEffect(() => {
    if (!latestEvent) return;
    if (latestEvent.type === lastEventTypeRef.current && latestEvent.timestamp === lastEventTypeRef.current) return;

    if (latestEvent.type === "replan_started") {
      setSnapshotPlan(currentPlan);
      setReplanInFlight(true);
    } else if (latestEvent.type === "replan_complete") {
      setReplanInFlight(false);
      // Fetch fresh plan after a short delay for the backend to commit changes
      setTimeout(fetchItinerary, 500);
    }
    lastEventTypeRef.current = latestEvent.type;
  }, [latestEvent, currentPlan, fetchItinerary]);

  return { currentPlan, snapshotPlan, replanInFlight, refetch: fetchItinerary };
}
