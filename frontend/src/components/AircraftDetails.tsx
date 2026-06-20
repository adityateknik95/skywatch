import type { Aircraft } from "../types";

interface Props {
  aircraft: Aircraft | undefined; // live data for the selected aircraft (this cycle)
  selected: string;
  injectedAttack: string | null; // non-null if this aircraft is the injected demo target
  stickyReason: string | null; // last reason the injected target was flagged with
}

// Plain-English explanation for each flag reason code.
const REASON_TEXT: Record<string, string> = {
  "physics:speed": "Moving faster than physically possible.",
  "physics:accel": "Speed jumped faster than any aircraft can accelerate.",
  "physics:turn": "Turning faster than physically possible.",
  "physics:climb": "Climbing or descending faster than physically possible.",
  "physics:velpos": "Reported speed doesn't match how far it actually moved.",
  "physics:altdiff": "Barometric and GPS altitude disagree (altitude spoof).",
  ml: "Motion doesn't match the model's learned normal pattern.",
};

function explainReason(reason: string | null | undefined): string {
  if (!reason) return "";
  return REASON_TEXT[reason] ?? reason;
}

function fmt(v: number | null | undefined, digits = 0, unit = ""): string {
  return v == null ? "—" : `${v.toFixed(digits)}${unit}`;
}

export default function AircraftDetails({ aircraft, selected, injectedAttack, stickyReason }: Props) {
  const a = aircraft;
  const liveFlagged = !!a?.is_anomaly;
  // For the injected target, keep showing "detected" once caught (sticky), so the
  // card doesn't flicker as the plane drifts in/out of replay snapshots.
  const reason = liveFlagged ? a?.reason ?? null : stickyReason;
  const showFlagged = injectedAttack ? liveFlagged || !!stickyReason : liveFlagged;

  return (
    <div className={`details ${injectedAttack ? "injected" : ""}`}>
      <div className="panel-head">
        <h3>{injectedAttack ? "⚡ Injected target" : "Selected aircraft"}</h3>
        <span className="mono">{selected}</span>
      </div>
      <div className="det-body">
        <div className="det-id">
          <span className="callsign">{a?.callsign?.trim() || "(no callsign)"}</span>
          {injectedAttack && <span className="atk-badge">spoof: {injectedAttack}</span>}
        </div>
        <div className={`det-status ${showFlagged ? "bad" : a == null ? "" : "ok"}`}>
          {showFlagged
            ? injectedAttack
              ? "⚡ ATTACK DETECTED"
              : "⚑ FLAGGED"
            : a == null
            ? "not in current view (showing history)"
            : "✓ normal"}
        </div>
        {showFlagged && reason && (
          <div className="det-reason">
            <span className="rcode">{reason}</span>
            <span className="rtext">why: {explainReason(reason)}</span>
          </div>
        )}
        <div className="det-grid">
          <div><label>altitude</label>{fmt(a?.baro_altitude, 0, " m")}</div>
          <div><label>speed</label>{fmt(a?.velocity, 0, " m/s")}</div>
          <div><label>heading</label>{fmt(a?.true_track, 0, "°")}</div>
          <div><label>vert. rate</label>{fmt(a?.vertical_rate, 1, " m/s")}</div>
          <div><label>latitude</label>{fmt(a?.lat, 3)}</div>
          <div><label>longitude</label>{fmt(a?.lon, 3)}</div>
        </div>
      </div>
    </div>
  );
}
