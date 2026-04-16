import { useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
} from "@xyflow/react";
import type { ItineraryResponse } from "../types";

const WEATHER_BORDER: Record<string, string> = {
  any: "#22c55e",    // green
  sunny: "#eab308",  // yellow
  rainy: "#3b82f6",  // blue
};

const NODE_W = 160;
const NODE_H = 56;
const COL_GAP = 220;
const ROW_GAP = 80;

interface Props {
  plan: ItineraryResponse | null;
  replanInFlight: boolean;
}

export function PlanGraph({ plan, replanInFlight }: Props) {
  const { nodes, edges } = useMemo(() => {
    if (!plan || !plan.trip_planned) return { nodes: [], edges: [] };

    const nodes: Node[] = [];
    const edges: Edge[] = [];

    plan.days.forEach((day, colIdx) => {
      // Day header node
      const headerId = `day-${day.day_number}-header`;
      nodes.push({
        id: headerId,
        position: { x: colIdx * COL_GAP, y: 0 },
        data: { label: `Day ${day.day_number}\n${day.city}` },
        style: {
          width: NODE_W,
          height: NODE_H,
          background: "#1e293b",
          border: "1px solid #475569",
          color: "#94a3b8",
          fontSize: 11,
          borderRadius: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center" as const,
          whiteSpace: "pre-line" as const,
        },
      });

      let prevId = headerId;
      day.actions.forEach((action, rowIdx) => {
        const nodeId = `day-${day.day_number}-action-${rowIdx}`;
        const border = WEATHER_BORDER[action.weather_requirement] ?? WEATHER_BORDER.any;
        nodes.push({
          id: nodeId,
          position: { x: colIdx * COL_GAP, y: (rowIdx + 1) * ROW_GAP },
          data: {
            label: action.name.replace(/_/g, " "),
          },
          style: {
            width: NODE_W,
            height: NODE_H,
            background: "#0f172a",
            border: `2px solid ${border}`,
            color: "#e2e8f0",
            fontSize: 10,
            borderRadius: 6,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            textAlign: "center" as const,
          },
        });

        edges.push({
          id: `e-${prevId}-${nodeId}`,
          source: prevId,
          target: nodeId,
          animated: replanInFlight,
          style: { stroke: "#475569" },
        });
        prevId = nodeId;
      });
    });

    return { nodes, edges };
  }, [plan, replanInFlight]);

  if (!plan || !plan.trip_planned) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm italic">
        No plan yet — send "plan my trip" in chat
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#1e293b" gap={20} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}
