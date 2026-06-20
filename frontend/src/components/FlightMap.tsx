import { useEffect, useMemo, useState } from "react";
import { Map, useControl } from "react-map-gl/maplibre";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import "maplibre-gl/dist/maplibre-gl.css";

import type { Aircraft } from "../types";
import { fetchTrack } from "../lib/api";

// Free MapLibre demo style — no API token required (spec §12).
const MAP_STYLE = "https://demotiles.maplibre.org/style.json";
const INITIAL_VIEW = { longitude: 10, latitude: 51, zoom: 5 };

const COLOR_NORMAL: [number, number, number] = [110, 190, 150];
const COLOR_ANOMALY: [number, number, number] = [255, 60, 60];
const COLOR_SELECTED: [number, number, number] = [90, 165, 255];

interface Props {
  aircraft: Aircraft[];
  selected: string | null;
  onSelect: (icao24: string | null) => void;
}

// deck.gl as a synchronized MapLibre overlay — follows the base map exactly.
function DeckOverlay(props: Record<string, unknown>) {
  const overlay = useControl(() => new MapboxOverlay(props as any));
  overlay.setProps(props as any);
  return null;
}

export default function FlightMap({ aircraft, selected, onSelect }: Props) {
  const [track, setTrack] = useState<[number, number][]>([]);

  useEffect(() => {
    if (!selected) {
      setTrack([]);
      return;
    }
    let cancelled = false;
    fetchTrack(selected)
      .then((t) => {
        if (cancelled) return;
        setTrack(
          t.track
            .filter((p) => p.lon != null && p.lat != null)
            .map((p) => [p.lon, p.lat] as [number, number])
        );
      })
      .catch(() => setTrack([]));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const positioned = useMemo(
    () => aircraft.filter((a) => a.lon != null && a.lat != null),
    [aircraft]
  );
  const anomalies = useMemo(() => positioned.filter((a) => a.is_anomaly), [positioned]);

  const layers = [
    new ScatterplotLayer<Aircraft>({
      id: "glow",
      data: anomalies,
      getPosition: (a) => [a.lon!, a.lat!],
      getRadius: (a) => 9000 + Math.min((a.score ?? 0) * 1500, 30000),
      radiusUnits: "meters",
      radiusMaxPixels: 45,
      getFillColor: [255, 60, 60, 70],
      pickable: false,
      updateTriggers: { getRadius: [anomalies] },
    }),
    new ScatterplotLayer<Aircraft>({
      id: "aircraft",
      data: positioned,
      getPosition: (a) => [a.lon!, a.lat!],
      getRadius: (a) => (a.is_anomaly ? 5000 : 2500),
      radiusUnits: "meters",
      radiusMinPixels: 2.5,
      radiusMaxPixels: 16,
      getFillColor: (a) =>
        a.icao24 === selected
          ? COLOR_SELECTED
          : a.is_anomaly
          ? COLOR_ANOMALY
          : COLOR_NORMAL,
      stroked: true,
      getLineColor: [10, 15, 20],
      lineWidthMinPixels: 0.5,
      pickable: true,
      updateTriggers: { getFillColor: [selected], getRadius: [positioned] },
    }),
    track.length > 1 &&
      new PathLayer<{ path: [number, number][] }>({
        id: "track",
        data: [{ path: track }],
        getPath: (d) => d.path,
        getColor: COLOR_SELECTED,
        getWidth: 3,
        widthMinPixels: 2,
        capRounded: true,
        jointRounded: true,
      }),
  ].filter(Boolean);

  return (
    <Map
      initialViewState={INITIAL_VIEW}
      mapStyle={MAP_STYLE}
      style={{ width: "100%", height: "100%" }}
    >
      <DeckOverlay
        layers={layers}
        interleaved={false}
        onClick={(info: any) => onSelect(info?.object ? info.object.icao24 : null)}
        getTooltip={({ object }: any) =>
          object && {
            html: `<b>${object.callsign?.trim() || object.icao24}</b>${
              object.reason ? `<br/>${object.reason}` : ""
            }${object.score != null ? `<br/>score ${object.score.toFixed(2)}` : ""}`,
            style: { fontSize: "12px" },
          }
        }
      />
    </Map>
  );
}
