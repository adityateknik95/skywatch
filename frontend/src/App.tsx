import { useEffect, useMemo, useRef, useState } from "react";

import FlightMap from "./components/FlightMap";
import AnomalyPanel from "./components/AnomalyPanel";
import AircraftDetails from "./components/AircraftDetails";
import ScoreTimeline from "./components/ScoreTimeline";
import { connectLive } from "./lib/ws";
import { injectDemo } from "./lib/api";
import type { Cycle, Status } from "./types";

export default function App() {
  const [cycle, setCycle] = useState<Cycle | null>(null);
  const [status, setStatus] = useState<Status>("disconnected");
  const [selected, setSelected] = useState<string | null>(null);
  const [injected, setInjected] = useState<{ icao24: string; attack: string } | null>(null);
  const [injectedReason, setInjectedReason] = useState<string | null>(null);
  const injectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => connectLive(setCycle, setStatus), []);

  const aircraft = cycle?.aircraft ?? [];
  const flagged = useMemo(
    () =>
      aircraft
        .filter((a) => a.is_anomaly)
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0)),
    [aircraft]
  );

  const cycleTime = cycle?.time
    ? new Date(cycle.time * 1000).toLocaleTimeString()
    : "—";

  const selectedAircraft = selected
    ? aircraft.find((a) => a.icao24 === selected)
    : undefined;

  // Capture the injected target's reason the moment it flags, and keep it (sticky)
  // for the duration of the attack so the demo card doesn't flicker as the plane
  // drifts in and out of replay snapshots.
  useEffect(() => {
    if (
      injected &&
      selected === injected.icao24 &&
      selectedAircraft?.is_anomaly &&
      selectedAircraft.reason
    ) {
      setInjectedReason(selectedAircraft.reason);
    }
  }, [cycle, injected, selected, selectedAircraft]);

  const onInject = async () => {
    const r = await injectDemo("velocity");
    if (r.armed && r.icao24) {
      setSelected(r.icao24);
      setInjected({ icao24: r.icao24, attack: r.attack ?? "velocity" });
      setInjectedReason(null);
      if (injectTimer.current) clearTimeout(injectTimer.current);
      injectTimer.current = setTimeout(() => {
        setInjected(null);
        setInjectedReason(null);
      }, 30000);
    }
  };

  const injectedAttack =
    injected && injected.icao24 === selected ? injected.attack : null;
  const stickyReason = injectedAttack ? injectedReason : null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◎</span> SkyWatch
          <span className="subtitle">ADS-B trajectory anomaly detection</span>
        </div>
        <div className="stats">
          <button className="inject-btn" onClick={onInject} title="Inject a synthetic attack into a live aircraft">
            ⚡ Inject attack
          </button>
          <span>aircraft <b>{aircraft.length}</b></span>
          <span>flagged <b className="bad">{flagged.length}</b></span>
          <span>cycle <b className="mono">{cycleTime}</b></span>
          <span className={`conn ${status}`}>
            <span className="dot" /> {status}
          </span>
        </div>
      </header>

      <main className="map-cell">
        <FlightMap aircraft={aircraft} selected={selected} onSelect={setSelected} />
      </main>

      <aside className="sidebar">
        <AnomalyPanel flagged={flagged} selected={selected} onSelect={setSelected} />
        {selected && (
          <AircraftDetails
            aircraft={selectedAircraft}
            selected={selected}
            injectedAttack={injectedAttack}
            stickyReason={stickyReason}
          />
        )}
        {selected && <ScoreTimeline icao24={selected} />}
      </aside>
    </div>
  );
}
