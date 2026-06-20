import type { Track } from "../types";
import { API_URL } from "./config";

export async function fetchTrack(icao24: string): Promise<Track> {
  const res = await fetch(`${API_URL}/aircraft/${icao24}/track?limit=300`);
  if (!res.ok) throw new Error(`track fetch failed: ${res.status}`);
  return res.json();
}

export interface InjectResult {
  armed: boolean;
  attack?: string;
  icao24?: string;
  callsign?: string;
  error?: string;
}

export async function injectDemo(attack = "velocity"): Promise<InjectResult> {
  const res = await fetch(`${API_URL}/demo/inject?attack=${attack}`, { method: "POST" });
  return res.json();
}
