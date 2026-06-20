import type { Aircraft } from "../types";

interface Props {
  flagged: Aircraft[];
  selected: string | null;
  onSelect: (icao24: string) => void;
}

export default function AnomalyPanel({ flagged, selected, onSelect }: Props) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Anomalies</h2>
        <span className="count">{flagged.length}</span>
      </div>
      {flagged.length === 0 ? (
        <p className="empty">No flagged aircraft this cycle.</p>
      ) : (
        <ul className="anomaly-list">
          {flagged.map((a) => (
            <li
              key={a.icao24}
              className={a.icao24 === selected ? "row sel" : "row"}
              onClick={() => onSelect(a.icao24)}
            >
              <span className="dot" />
              <span className="id">{a.callsign?.trim() || a.icao24}</span>
              <span className="reason">{a.reason}</span>
              <span className="score">{a.score != null ? a.score.toFixed(2) : "—"}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
