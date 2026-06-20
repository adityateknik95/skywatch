import { useEffect, useMemo, useState } from "react";
import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fetchTrack } from "../lib/api";

interface Props {
  icao24: string;
}

interface Row {
  rel: number; // minutes since first sample
  score: number;
  flagged: boolean;
}

export default function ScoreTimeline({ icao24 }: Props) {
  const [data, setData] = useState<Row[]>([]);
  const [threshold, setThreshold] = useState<number | undefined>();
  const [yMax, setYMax] = useState<number | null>(null); // null = auto-fit to the signal

  useEffect(() => {
    let cancelled = false;
    fetchTrack(icao24)
      .then((t) => {
        if (cancelled) return;
        const t0 = t.scores[0]?.t ?? 0;
        setData(
          t.scores.map((s) => ({
            rel: (s.t - t0) / 60,
            score: s.score,
            flagged: s.is_anomaly,
          }))
        );
        setThreshold(t.scores[0]?.threshold);
        setYMax(null); // reset zoom when a new aircraft is selected
      })
      .catch(() => {
        if (!cancelled) setData([]);
      });
    return () => {
      cancelled = true;
    };
  }, [icao24]);

  // Auto-fit the Y-axis to the actual residuals so tiny-but-real spikes are visible.
  const dataMax = useMemo(() => data.reduce((m, r) => Math.max(m, r.score), 0), [data]);
  const autoMax = Math.max(dataMax * 1.25, 0.02);
  const effMax = yMax ?? autoMax;
  const fmt = (v: number) => (v >= 1 ? v.toFixed(1) : v.toFixed(3));
  const zoom = (factor: number) =>
    setYMax(Math.min(Math.max(effMax * factor, 0.01), 50));

  const thresholdInView = threshold != null && threshold <= effMax;

  return (
    <div className="timeline">
      <div className="panel-head">
        <h3>Score timeline</h3>
        <span className="mono">{icao24}</span>
      </div>
      {data.length === 0 ? (
        <p className="empty">No score history yet.</p>
      ) : (
        <>
          <div className="tl-controls">
            <button className="tl-btn" onClick={() => zoom(2)} title="Zoom out (larger Y range)">
              −
            </button>
            <button className="tl-btn" onClick={() => zoom(0.5)} title="Zoom in (smaller Y range — reveals small spikes)">
              +
            </button>
            <button className="tl-btn wide" onClick={() => setYMax(null)} title="Auto-fit the Y-axis to the residual signal">
              Auto
            </button>
            <button
              className="tl-btn wide"
              onClick={() => setYMax(threshold ? threshold * 1.1 : autoMax)}
              title="Show the full range including the anomaly threshold"
            >
              Full
            </button>
            <span className="tl-ymax">y-max {fmt(effMax)}</span>
          </div>
          <ResponsiveContainer width="100%" height={168}>
            <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -6 }}>
              <XAxis
                dataKey="rel"
                tickFormatter={(v) => `${Math.round(v)}m`}
                stroke="#7b8794"
                fontSize={11}
              />
              <YAxis
                domain={[0, effMax]}
                allowDataOverflow
                tickFormatter={fmt}
                stroke="#7b8794"
                fontSize={11}
                width={48}
              />
              <Tooltip
                contentStyle={{ background: "#10151c", border: "1px solid #2a3340" }}
                labelFormatter={(v) => `${Number(v).toFixed(1)} min`}
                formatter={(v: number) => [v.toFixed(4), "residual"]}
              />
              {thresholdInView && (
                <ReferenceLine
                  y={threshold}
                  stroke="#ff5252"
                  strokeDasharray="4 4"
                  ifOverflow="hidden"
                  label={{ value: "threshold", fill: "#ff5252", fontSize: 10, position: "insideTopRight" }}
                />
              )}
              <Line
                type="monotone"
                dataKey="score"
                stroke="#5aa5ff"
                dot={false}
                isAnimationActive={false}
                strokeWidth={1.6}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="hint">
            Residual (per-point MSE). Y-axis auto-fits the signal — use <b>−</b>/<b>+</b> to zoom,{" "}
            <b>Full</b> for the p99 threshold
            {threshold != null && !thresholdInView ? ` (${fmt(threshold)}, above this view)` : ""}.
          </p>
        </>
      )}
    </div>
  );
}
