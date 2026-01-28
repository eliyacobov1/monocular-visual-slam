"use client";

import { Canvas } from "@react-three/fiber";
import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";
import { useSlamData } from "../../hooks/useSlamData";

type Vec3 = [number, number, number];

type TrajectoryPoint = {
  id: number;
  position: Vec3;
};

const chartColors = {
  reprojection: "#a855f7",
  density: "#22d3ee",
  latency: "#f97316",
};

const toTrajectoryPoints = (nodes: { id: number; pose: number[][] }[]): TrajectoryPoint[] =>
  nodes.map((node) => ({
    id: node.id,
    position: [node.pose?.[0]?.[3] ?? 0, node.pose?.[1]?.[3] ?? 0, node.pose?.[2]?.[3] ?? 0],
  }));

const TrajectoryLine = ({ points, color }: { points: Vec3[]; color: string }) => {
  const positions = useMemo(() => points.flat(), [points]);
  if (positions.length < 6) {
    return null;
  }

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[new Float32Array(positions), 3]}
        />
      </bufferGeometry>
      <lineBasicMaterial args={[{ color, linewidth: 2 }]} />
    </line>
  );
};

const TrajectoryNodes = ({ points, color }: { points: TrajectoryPoint[]; color: string }) => (
  <group>
    {points.map((point) => (
      <group key={point.id} position={point.position}>
        <axesHelper args={[0.15]} />
        <mesh>
          <sphereGeometry args={[0.025, 12, 12]} />
          <meshStandardMaterial args={[{ color, emissive: color }]} />
        </mesh>
      </group>
    ))}
  </group>
);

export default function Dashboard() {
  const {
    connected,
    homography,
    graphEdges,
    rawTrajectory,
    optimizedTrajectory,
    metrics,
    metricsHistory,
    keyframes,
  } = useSlamData();

  const rawPoints = useMemo(() => toTrajectoryPoints(rawTrajectory), [rawTrajectory]);
  const optimizedPoints = useMemo(() => toTrajectoryPoints(optimizedTrajectory), [optimizedTrajectory]);

  const chartData = useMemo(
    () =>
      metricsHistory.map((entry, index) => ({
        frame: index,
        reprojection: entry.reprojectionError,
        density: entry.graphDensity,
        latency: entry.latencyMs,
      })),
    [metricsHistory],
  );

  const matchPairs = useMemo(() => {
    const pairs = homography.current.map((point, index) => ({
      current: point,
      reference: homography.reference[index],
    }));
    return pairs.filter((pair) => pair.reference);
  }, [homography]);

  const homographyBounds = useMemo(() => {
    const points = [...homography.current, ...homography.reference];
    const maxX = Math.max(1, ...points.map((point) => point[0]));
    const maxY = Math.max(1, ...points.map((point) => point[1]));
    return { width: maxX, height: maxY };
  }, [homography]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-6 py-6">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">Monocular SLAM</p>
            <h1 className="text-3xl font-semibold text-slate-50">Homography + Pose Graph Dashboard</h1>
          </div>
          <div className="flex items-center gap-3 rounded-full border border-cyan-500/40 bg-slate-900/60 px-4 py-2 text-sm">
            <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400" : "bg-rose-500"}`} />
            <span className="font-medium text-slate-200">
              {connected ? "Live stream connected" : "Awaiting WebSocket"}
            </span>
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
          <div className="grid gap-4">
            <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 p-4 shadow-[0_0_25px_rgba(34,211,238,0.15)]">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-cyan-200">Dual-View Camera Panel</h2>
                <span className="text-xs uppercase text-slate-400">Homography matches</span>
              </div>
              <div className="relative grid gap-4 md:grid-cols-2">
                {[
                  { label: "Reference Keyframe", points: homography.reference },
                  { label: "Current Frame", points: homography.current },
                ].map((panel) => (
                  <div
                    key={panel.label}
                    className="relative aspect-video overflow-hidden rounded-xl border border-slate-700 bg-slate-900/70"
                  >
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.18),_transparent_60%)]" />
                    <div className="absolute inset-x-0 top-0 flex items-center justify-between bg-slate-900/80 px-3 py-2 text-xs uppercase tracking-[0.25em] text-slate-400">
                      {panel.label}
                    </div>
                    <svg className="absolute inset-0 h-full w-full">
                      {panel.points.map((point, index) => (
                        <circle
                          key={`${panel.label}-pt-${index}`}
                          cx={point[0]}
                          cy={point[1]}
                          r={3}
                          fill="rgba(56,189,248,0.8)"
                        />
                      ))}
                    </svg>
                  </div>
                ))}
                <svg
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  viewBox={`0 0 ${homographyBounds.width * 2} ${homographyBounds.height}`}
                  preserveAspectRatio="none"
                >
                  {matchPairs.map((pair, index) => (
                    <line
                      key={`pair-${index}`}
                      x1={pair.reference[0]}
                      y1={pair.reference[1]}
                      x2={pair.current[0] + homographyBounds.width}
                      y2={pair.current[1]}
                      stroke="rgba(34,197,94,0.85)"
                      strokeWidth={1.5}
                      vectorEffect="non-scaling-stroke"
                    />
                  ))}
                </svg>
              </div>
            </div>

            <div className="rounded-2xl border border-purple-500/30 bg-slate-950 p-4 shadow-[0_0_25px_rgba(168,85,247,0.15)]">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-purple-200">Optimization Metrics</h2>
                <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Realtime</div>
              </div>
              <div className="grid gap-4 lg:grid-cols-3">
                {[
                  { label: "Reprojection Error", value: metrics.reprojectionError.toFixed(3), unit: "px" },
                  { label: "Graph Density", value: metrics.graphDensity.toFixed(2), unit: "edges" },
                  { label: "System Latency", value: metrics.latencyMs.toFixed(1), unit: "ms" },
                ].map((metric) => (
                  <div
                    key={metric.label}
                    className="rounded-xl border border-slate-700 bg-slate-900/70 px-4 py-3"
                  >
                    <p className="text-xs uppercase tracking-[0.25em] text-slate-400">{metric.label}</p>
                    <div className="mt-2 flex items-baseline gap-2">
                      <span className="text-2xl font-semibold text-slate-100">{metric.value}</span>
                      <span className="text-xs text-slate-400">{metric.unit}</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <XAxis dataKey="frame" hide />
                    <YAxis hide />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(15,23,42,0.9)",
                        border: "1px solid rgba(148,163,184,0.2)",
                        borderRadius: "12px",
                      }}
                      labelStyle={{ color: "#e2e8f0" }}
                    />
                    <Line type="monotone" dataKey="reprojection" stroke={chartColors.reprojection} dot={false} />
                    <Line type="monotone" dataKey="density" stroke={chartColors.density} dot={false} />
                    <Line type="monotone" dataKey="latency" stroke={chartColors.latency} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <div className="rounded-2xl border border-emerald-500/30 bg-slate-950 p-4 shadow-[0_0_25px_rgba(34,197,94,0.2)]">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-emerald-200">3D Pose Graph View</h2>
                <span className="text-xs uppercase tracking-[0.3em] text-slate-400">Trajectory</span>
              </div>
              <div className="h-80 rounded-xl border border-slate-800 bg-slate-900/60">
                <Canvas camera={{ position: [1.8, 1.2, 1.8], fov: 55 }}>
                  <ambientLight intensity={0.5} />
                  <pointLight position={[4, 4, 4]} intensity={1.2} />
                  <gridHelper args={[4, 8, "#1e293b", "#0f172a"]} />
                  <TrajectoryLine
                    points={rawPoints.map((point) => point.position)}
                    color="#ef4444"
                  />
                  <TrajectoryLine
                    points={optimizedPoints.map((point) => point.position)}
                    color="#22c55e"
                  />
                  <TrajectoryNodes points={rawPoints} color="#ef4444" />
                  <TrajectoryNodes points={optimizedPoints} color="#22c55e" />
                </Canvas>
              </div>
              <div className="mt-3 flex flex-wrap gap-3 text-xs uppercase tracking-[0.25em] text-slate-400">
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-red-500" /> Raw Odometry
                </span>
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-400" /> Optimized
                </span>
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-cyan-400" /> {graphEdges.length} edges
                </span>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-slate-200">Keyframe Gallery</h2>
                <span className="text-xs uppercase tracking-[0.3em] text-slate-400">Recent</span>
              </div>
              <div className="flex gap-3 overflow-x-auto pb-2">
                {keyframes.length === 0 ? (
                  <div className="flex h-24 w-full items-center justify-center rounded-xl border border-dashed border-slate-700 text-sm text-slate-400">
                    Keyframes will appear here as the graph grows.
                  </div>
                ) : (
                  keyframes.map((frame, index) => (
                    <div
                      key={`${frame}-${index}`}
                      className="flex h-24 min-w-[140px] flex-col items-center justify-center rounded-xl border border-slate-700 bg-slate-950/60"
                    >
                      <div className="text-xs uppercase tracking-[0.3em] text-slate-500">KF {index + 1}</div>
                      <div className="mt-2 text-[10px] text-slate-400">{frame}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
