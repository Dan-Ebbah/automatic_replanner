// ---- WebSocket event types ------------------------------------------------

export interface WsEventBase {
  type: string;
  timestamp: string;
}

export interface PlanUpdatedEvent extends WsEventBase {
  type: "plan_updated";
  payload: { trigger: string; step_count: number };
}

export interface WeatherAlertEvent extends WsEventBase {
  type: "weather_alert";
  payload: { location: string; new_weather: string; note?: string };
}

export interface ReplanStartedEvent extends WsEventBase {
  type: "replan_started";
  payload: { location: string; weather: string };
}

export interface ReplanCompleteEvent extends WsEventBase {
  type: "replan_complete";
  payload: {
    location: string;
    new_weather: string;
    affected_days: number[];
    status: "success" | "failed";
    attempts: number;
    latency_ms: number;
  };
}

export interface PlanBuiltEvent extends WsEventBase {
  type: "plan_built";
  payload: { day_number: number; city: string; action_count: number };
}

export interface InfoEvent extends WsEventBase {
  type: "info";
  payload: { message: string };
}

export type WsEvent =
  | PlanUpdatedEvent
  | WeatherAlertEvent
  | ReplanStartedEvent
  | ReplanCompleteEvent
  | PlanBuiltEvent
  | InfoEvent;

// ---- Itinerary types -------------------------------------------------------

export interface ActivityAction {
  name: string;
  duration_minutes: number;
  weather_requirement: "any" | "sunny" | "rainy";
}

export interface DayPlanResponse {
  day_number: number;
  city: string;
  weather: string;
  actions: ActivityAction[];
}

export interface ItineraryResponse {
  trip_planned: boolean;
  days: DayPlanResponse[];
}

// ---- Healing record -------------------------------------------------------

export interface HealingRecord {
  available: boolean;
  run_id?: string;
  location?: string;
  new_weather?: string;
  status?: "success" | "failed";
  attempts?: number;
  affected_days?: number[];
  latency_ms?: number;
}
