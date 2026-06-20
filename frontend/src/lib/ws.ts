import type { Cycle, Status } from "../types";
import { WS_URL } from "./config";

/**
 * Connect to the live WebSocket, delivering each cycle to `onCycle`. Reconnects
 * automatically on drop (spec §12). Returns a disposer to close the connection.
 */
export function connectLive(
  onCycle: (cycle: Cycle) => void,
  onStatus: (status: Status) => void
): () => void {
  let socket: WebSocket | null = null;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | undefined;

  const open = () => {
    socket = new WebSocket(WS_URL);
    socket.onopen = () => onStatus("connected");
    socket.onmessage = (event) => {
      try {
        onCycle(JSON.parse(event.data) as Cycle);
      } catch {
        /* ignore malformed frame */
      }
    };
    socket.onclose = () => {
      onStatus("disconnected");
      if (!closed) retry = setTimeout(open, 2000);
    };
    socket.onerror = () => socket?.close();
  };

  open();
  return () => {
    closed = true;
    if (retry) clearTimeout(retry);
    socket?.close();
  };
}
