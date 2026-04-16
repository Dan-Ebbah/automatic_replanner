import { useEffect, useRef, useState, useCallback } from "react";
import type { WsEvent } from "../types";

export type WsStatus = "connecting" | "open" | "closed" | "error";

function buildWsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;
  return `${protocol}://${host}/api/ws`;
}

export function useWebSocket() {
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [status, setStatus] = useState<WsStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    setStatus("connecting");
    const ws = new WebSocket(buildWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setStatus("open");
    };

    ws.onmessage = (ev) => {
      if (!mountedRef.current) return;
      try {
        const parsed: WsEvent = JSON.parse(ev.data as string);
        setEvents((prev) => [parsed, ...prev].slice(0, 200));
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setStatus("error");
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setStatus("closed");
      reconnectTimer.current = setTimeout(connect, 3000);
    };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connect]);

  return { events, status };
}
