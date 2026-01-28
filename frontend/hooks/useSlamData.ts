"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Vector2 = [number, number];

type PoseMatrix = number[][];

type HomographyPoints = {
  current: Vector2[];
  reference: Vector2[];
};

type GraphEdge = {
  from: number;
  to: number;
  type: "odometry" | "loop";
};

type TrajectoryNode = {
  id: number;
  pose: PoseMatrix;
};

type SlamMetrics = {
  reprojectionError: number;
  graphDensity: number;
  latencyMs: number;
};

type SlamMessage = {
  pose_matrix?: PoseMatrix;
  homography_pts?: HomographyPoints;
  graph_edges?: GraphEdge[];
  raw_trajectory?: TrajectoryNode[];
  optimized_trajectory?: TrajectoryNode[];
  metrics?: Partial<SlamMetrics>;
  keyframes?: string[];
};

type SlamState = {
  connected: boolean;
  lastMessageAt?: number;
  poseMatrix: PoseMatrix;
  homography: HomographyPoints;
  graphEdges: GraphEdge[];
  rawTrajectory: TrajectoryNode[];
  optimizedTrajectory: TrajectoryNode[];
  metrics: SlamMetrics;
  metricsHistory: SlamMetrics[];
  keyframes: string[];
};

const emptyPose: PoseMatrix = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

const defaultHomography: HomographyPoints = { current: [], reference: [] };

const defaultMetrics: SlamMetrics = {
  reprojectionError: 0,
  graphDensity: 0,
  latencyMs: 0,
};

const MAX_METRIC_SAMPLES = 120;

export function useSlamData(wsUrl?: string): SlamState {
  const [connected, setConnected] = useState(false);
  const [poseMatrix, setPoseMatrix] = useState<PoseMatrix>(emptyPose);
  const [homography, setHomography] = useState<HomographyPoints>(defaultHomography);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);
  const [rawTrajectory, setRawTrajectory] = useState<TrajectoryNode[]>([]);
  const [optimizedTrajectory, setOptimizedTrajectory] = useState<TrajectoryNode[]>([]);
  const [metrics, setMetrics] = useState<SlamMetrics>(defaultMetrics);
  const [metricsHistory, setMetricsHistory] = useState<SlamMetrics[]>([]);
  const [keyframes, setKeyframes] = useState<string[]>([]);
  const lastMessageAtRef = useRef<number | undefined>(undefined);
  const metricsRef = useRef<SlamMetrics>(defaultMetrics);
  const pendingFrameRef = useRef<number | null>(null);
  const latestMessageRef = useRef<SlamMessage | null>(null);

  useEffect(() => {
    const socketUrl = wsUrl ?? process.env.NEXT_PUBLIC_SLAM_WS_URL ?? "ws://localhost:8000/ws";
    const socket = new WebSocket(socketUrl);

    const flushLatest = () => {
      pendingFrameRef.current = null;
      const latest = latestMessageRef.current;
      if (!latest) {
        return;
      }
      if (latest.pose_matrix) {
        setPoseMatrix(latest.pose_matrix);
      }
      if (latest.homography_pts) {
        setHomography(latest.homography_pts);
      }
      if (latest.graph_edges) {
        setGraphEdges(latest.graph_edges);
      }
      if (latest.raw_trajectory) {
        setRawTrajectory(latest.raw_trajectory);
      }
      if (latest.optimized_trajectory) {
        setOptimizedTrajectory(latest.optimized_trajectory);
      }
      if (latest.metrics) {
        const currentMetrics = metricsRef.current;
        const nextMetrics: SlamMetrics = {
          reprojectionError: latest.metrics.reprojectionError ?? currentMetrics.reprojectionError,
          graphDensity: latest.metrics.graphDensity ?? currentMetrics.graphDensity,
          latencyMs: latest.metrics.latencyMs ?? currentMetrics.latencyMs,
        };
        metricsRef.current = nextMetrics;
        setMetrics(nextMetrics);
        setMetricsHistory((prev) => {
          const updated = [...prev, nextMetrics].slice(-MAX_METRIC_SAMPLES);
          return updated;
        });
      }
      if (latest.keyframes) {
        setKeyframes(latest.keyframes);
      }
      lastMessageAtRef.current = Date.now();
    };

    socket.addEventListener("open", () => setConnected(true));
    socket.addEventListener("close", () => setConnected(false));
    socket.addEventListener("message", (event) => {
      let parsed: SlamMessage;
      try {
        parsed = JSON.parse(event.data) as SlamMessage;
      } catch (error) {
        console.warn("Failed to parse SLAM message", error);
        return;
      }
      latestMessageRef.current = parsed;
      if (pendingFrameRef.current === null) {
        pendingFrameRef.current = window.requestAnimationFrame(flushLatest);
      }
    });

    return () => {
      if (pendingFrameRef.current !== null) {
        window.cancelAnimationFrame(pendingFrameRef.current);
      }
      socket.close();
    };
  }, [wsUrl]);

  const lastMessageAt = lastMessageAtRef.current;

  return useMemo(
    () => ({
      connected,
      lastMessageAt,
      poseMatrix,
      homography,
      graphEdges,
      rawTrajectory,
      optimizedTrajectory,
      metrics,
      metricsHistory,
      keyframes,
    }),
    [
      connected,
      lastMessageAt,
      poseMatrix,
      homography,
      graphEdges,
      rawTrajectory,
      optimizedTrajectory,
      metrics,
      metricsHistory,
      keyframes,
    ],
  );
}
