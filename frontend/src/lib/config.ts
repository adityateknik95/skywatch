// Backend base URLs. Override with VITE_API_URL / VITE_WS_URL if the API isn't on
// 127.0.0.1:8000. CORS is enabled on the backend, so direct calls work in dev.
export const API_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";
export const WS_URL =
  (import.meta.env.VITE_WS_URL as string | undefined) ?? "ws://127.0.0.1:8000/ws/live";
