const connectionEl = document.getElementById("connection-status");
const trackingStatusEl = document.getElementById("tracking-status");
const progressBarEl = document.getElementById("progress-bar");
const progressTextEl = document.getElementById("progress-text");
const frameIdEl = document.getElementById("frame-id");
const fpsEl = document.getElementById("fps");
const featuresEl = document.getElementById("features");
const matchesEl = document.getElementById("matches");
const inliersEl = document.getElementById("inliers");
const inlierRatioEl = document.getElementById("inlier-ratio");
const positionEl = document.getElementById("position");
const yprEl = document.getElementById("ypr");
const logEl = document.getElementById("log");
const canvas = document.getElementById("trajectory");
const ctx = canvas.getContext("2d");

const trajectory = [];
let lastPayload = null;

function setConnection(text, ok) {
  connectionEl.textContent = text;
  connectionEl.style.background = ok ? "rgba(34, 197, 94, 0.2)" : "rgba(239, 68, 68, 0.2)";
}

function updateTrajectory() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (trajectory.length < 2) {
    return;
  }
  const xs = trajectory.map((p) => p[0]);
  const ys = trajectory.map((p) => p[2]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padding = 12;
  const scaleX = (canvas.width - padding * 2) / (maxX - minX || 1);
  const scaleY = (canvas.height - padding * 2) / (maxY - minY || 1);

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  trajectory.forEach(([x, , z], idx) => {
    const px = padding + (x - minX) * scaleX;
    const py = canvas.height - padding - (z - minY) * scaleY;
    if (idx === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  });
  ctx.stroke();

  const [x, , z] = trajectory[trajectory.length - 1];
  const px = padding + (x - minX) * scaleX;
  const py = canvas.height - padding - (z - minY) * scaleY;
  ctx.fillStyle = "#dc2626";
  ctx.beginPath();
  ctx.arc(px, py, 4, 0, Math.PI * 2);
  ctx.fill();
}

function updateUI(payload) {
  lastPayload = payload;
  const progress = Math.round(payload.progress * 100);
  progressBarEl.style.width = `${progress}%`;
  progressTextEl.textContent = `${progress}% · ${payload.frame_id} / ${payload.total_frames || "—"} frames`;
  frameIdEl.textContent = payload.frame_id;
  fpsEl.textContent = payload.fps.toFixed(1);
  featuresEl.textContent = payload.features;
  matchesEl.textContent = payload.matches;
  inliersEl.textContent = payload.inliers;
  inlierRatioEl.textContent = payload.inlier_ratio.toFixed(2);
  positionEl.textContent = payload.position.map((v) => v.toFixed(2)).join(", ");
  yprEl.textContent = payload.yaw_pitch_roll.map((v) => v.toFixed(1)).join("° / ") + "°";

  trackingStatusEl.textContent = payload.status;
  trackingStatusEl.className = `status ${payload.status_level}`;

  trajectory.push(payload.position);
  if (trajectory.length > 500) {
    trajectory.shift();
  }
  updateTrajectory();

  logEl.innerHTML = "";
  payload.logs.forEach((entry) => {
    const li = document.createElement("li");
    li.textContent = entry;
    logEl.appendChild(li);
  });
}

function connect() {
  const wsUrl = window.SLAM_WS_URL || `ws://${window.location.hostname}:8000`;
  const socket = new WebSocket(wsUrl);

  socket.addEventListener("open", () => {
    setConnection("Live stream connected", true);
  });

  socket.addEventListener("close", () => {
    setConnection("Disconnected - retrying", false);
    setTimeout(connect, 1000);
  });

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "frame") {
      updateUI(message.payload);
    }
  });
}

setConnection("Connecting…", false);
connect();
