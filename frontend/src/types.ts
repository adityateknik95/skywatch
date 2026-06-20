export interface Aircraft {
  icao24: string;
  callsign: string | null;
  lat: number | null;
  lon: number | null;
  baro_altitude: number | null;
  geo_altitude: number | null;
  velocity: number | null;
  true_track: number | null;
  vertical_rate: number | null;
  score: number | null;
  is_anomaly: boolean;
  reason: string | null;
  threshold: number;
}

export interface Cycle {
  time: number | null;
  aircraft: Aircraft[];
}

export interface TrackPoint {
  t: number;
  lat: number;
  lon: number;
  baro_altitude: number | null;
  velocity: number | null;
  true_track: number | null;
}

export interface ScorePoint {
  t: number;
  score: number;
  threshold: number;
  is_anomaly: boolean;
  reason: string | null;
}

export interface Track {
  icao24: string;
  track: TrackPoint[];
  scores: ScorePoint[];
}

export type Status = "connected" | "disconnected";
