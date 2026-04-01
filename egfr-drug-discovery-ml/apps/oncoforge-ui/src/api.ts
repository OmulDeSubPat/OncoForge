import type { ControlForm, DashboardPayload, DetailPayload } from "@/types";

const API_BASE =
  import.meta.env.VITE_ONCOSYNTH_API_BASE?.trim() ||
  import.meta.env.VITE_ONCOFORGE_API_BASE?.trim() ||
  "http://127.0.0.1:8000";

function apiUrl(path: string) {
  if (!API_BASE) {
    return path;
  }
  return `${API_BASE}${path}`;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const target = apiUrl(path);
  let response: Response;
  try {
    response = await fetch(target, {
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      ...init,
    });
  } catch {
    throw new Error(
      `Backend-ul OncoSynth nu raspunde la ${target}. Porneste serverul FastAPI pe http://127.0.0.1:8000 si incearca din nou.`,
    );
  }
  if (!response.ok) {
    const rawMessage = await response.text();
    let message = rawMessage;
    try {
      const parsed = JSON.parse(rawMessage) as { detail?: string | Array<{ msg?: string }> };
      if (typeof parsed.detail === "string") {
        message = parsed.detail;
      } else if (Array.isArray(parsed.detail) && parsed.detail.length > 0) {
        message = parsed.detail.map((item) => item.msg).filter(Boolean).join("; ");
      }
    } catch {
      // keep raw text
    }
    throw new Error(message || `Cererea catre ${target} a esuat.`);
  }
  return (await response.json()) as T;
}

export async function fetchDashboard(params: {
  sessionName: string;
  smiles?: string;
  limit?: number;
}): Promise<DashboardPayload> {
  const query = new URLSearchParams({
    session_name: params.sessionName,
    limit: String(params.limit ?? 120),
  });
  if (params.smiles) {
    query.set("smiles", params.smiles);
  }
  return fetchJson<DashboardPayload>(`/api/dashboard?${query.toString()}`);
}

export async function fetchMoleculeDetail(params: { sessionName: string; smiles: string }): Promise<DetailPayload> {
  const query = new URLSearchParams({
    session_name: params.sessionName,
    smiles: params.smiles,
  });
  return fetchJson<DetailPayload>(`/api/molecule?${query.toString()}`);
}

export async function startSession(control: ControlForm) {
  return fetchJson<{ ok: boolean; message: string; sessionName: string }>("/api/control/start", {
    method: "POST",
    body: JSON.stringify({
      session_name: control.sessionName,
      mode: control.mode,
      seed_count: control.seedCount,
      rounds: control.rounds,
      variants_per_seed: control.variantsPerSeed,
      beam_width: control.beamWidth,
      replace_existing: control.replaceExisting,
    }),
  });
}

export async function stopSession(sessionName: string) {
  return fetchJson<{ ok: boolean; message: string; pid?: number }>("/api/control/stop", {
    method: "POST",
    body: JSON.stringify({ session_name: sessionName }),
  });
}

export async function resetSession(sessionName: string) {
  return fetchJson<{ ok: boolean; message: string; sessionName: string }>("/api/control/reset", {
    method: "POST",
    body: JSON.stringify({ session_name: sessionName }),
  });
}
