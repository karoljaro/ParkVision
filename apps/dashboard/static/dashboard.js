import { DashboardState, projectionPoints } from "./freshness.js";

const el = (id) => document.getElementById(id);
const clock = () => ({ mono: performance.now(), wall: Date.now() });
const api = new DashboardState();
const svgNs = "http://www.w3.org/2000/svg";
const dateFormat = new Intl.DateTimeFormat("pl-PL", {
  dateStyle: "short",
  timeStyle: "medium",
});
const timeFormat = new Intl.DateTimeFormat("pl-PL", { timeStyle: "medium" });
const labels = {
  confirmed_free: ["✓ Wolne", "good"],
  confirmed_occupied: ["■ Zajęte", "occupied"],
  held: ["◷ Podtrzymane", "warning"],
  unknown: ["? Nieznane", "quiet"],
  stale: ["⊘ Nieaktualne", "danger"],
};
const viewLabels = {
  verified: ["✓ Zgodny", "good"],
  checking: ["◷ Sprawdzanie", "warning"],
  mismatch: ["! Niezgodny", "danger"],
  unverifiable: ["? Niepotwierdzony", "warning"],
};
const sources = {
  streaming: "Odbiór działa",
  waiting: "Oczekiwanie na klatkę",
  starting: "Uruchamianie",
  recovering: "Odzyskiwanie sygnału",
  degraded: "Ograniczone źródło",
  interrupted: "Sygnał utracony",
  eof: "Koniec nagrania",
  stopped: "Zatrzymane",
  error: "Błąd źródła",
};
const events = {
  observation: "Obserwacja",
  session_started: "Początek sesji",
  camera_integrity: "Kontrola widoku",
  view_expired: "Wygaśnięcie widoku",
  source_lost: "Utrata sygnału",
  source_recovered: "Odzyskano sygnał",
  frame_rejected: "Odrzucono klatkę",
  heartbeat: "Stan źródła",
  eof: "Koniec nagrania",
  stopped: "Zatrzymanie",
  error: "Błąd runnera",
};
const reasons = {
  camera_view_changed: "Przesunięcie lub zmiana kadru",
  native_roi_alignment_verified: "Zgodne położenie ROI",
  registration_in_progress: "Rejestracja obrazu w toku",
  reference_required: "Brak zatwierdzonej referencji",
  insufficient_background_features: "Za mało cech stałego tła",
  insufficient_roi_spatial_support: "Brak pokrycia obszaru ROI",
  view_verification_expired: "Wygasł dowód zgodności widoku",
  dimension_mismatch: "Inna rozdzielczość obrazu",
  end_of_file: "Koniec pliku",
  view_unverified: "Brak potwierdzenia widoku",
  stale_frame: "Klatka dotarła zbyt późno",
};
let selected = null;
let expiryTimer;
let latestHistory = null;
let historyError = false;
let lastLayoutKey;
let lastCameraKey;
let lastDetailKey;
let statusInFlight = false;
let statusTimer;

const formatted = (value, format = dateFormat) =>
  value ? format.format(new Date(value)) : "—";
const reasonText = (reason) => (reason ? (reasons[reason] ?? reason) : "—");
function node(tag, contents, className) {
  const item = document.createElement(tag);
  if (contents !== undefined) item.textContent = contents;
  if (className) item.className = className;
  return item;
}

function badge(id, [label, tone]) {
  const item = el(id);
  if (item.textContent !== label) item.textContent = label;
  item.className = `badge ${tone}`;
}

function details(id, rows) {
  const children = rows.map(([label, value, className]) => {
    const row = node("div");
    row.append(node("dt", label), node("dd", value ?? "—", className));
    return row;
  });
  el(id).replaceChildren(...children);
}

function counters(data) {
  const values = data?.availability_counts;
  const keys = [
    ["physical_total", "Fizyczne miejsca", "∑", "quiet"],
    ["confirmed_free", "Potwierdzone wolne", "✓", "good"],
    ["confirmed_occupied", "Zajęte", "■", "occupied"],
    ["held", "Podtrzymane", "◷", "warning"],
    ["unknown", "Nieznane", "?", "quiet"],
    ["stale", "Nieaktualne", "⊘", "danger"],
  ];
  if (!el("counters").hasChildNodes()) {
    for (const [key, name, symbol, tone] of keys) {
      const card = node("div", undefined, `counter ${tone}`);
      const title = node("span");
      const icon = node("i", symbol);
      icon.setAttribute("aria-hidden", "true");
      title.append(icon, document.createTextNode(name));
      const count = node("strong", "—");
      count.id = `count-${key}`;
      card.append(title, count);
      el("counters").append(card);
    }
  }
  for (const [key] of keys)
    el(`count-${key}`).textContent = values?.[key] ?? "—";
}

function notice(data) {
  const target = el("notice");
  let text = "";
  let tone = "warning";
  if (api.connection === "disconnected") {
    text = "Brak połączenia z API. Dostępność została wycofana.";
    tone = "danger";
  } else if (api.connection === "error") {
    text =
      api.error === "snapshot_missing"
        ? "Brak snapshotu. Uruchom runner z tym samym katalogiem wyników."
        : api.error === "unsupported_legacy_snapshot"
          ? "Ten snapshot nie zawiera danych kontroli widoku. Użyj runnera Stage 2 lub nowszego."
          : "Nie można odczytać stanu runnera. Sprawdź snapshot i konfigurację adaptera.";
    tone = "danger";
  } else if (data?.viewExpired || data?.effectiveFreshness === "stale") {
    text = "Dane utraciły ważność. Panel wycofał potwierdzoną dostępność.";
  } else if (data?.effectiveFreshness === "awaiting") {
    text = "API ponownie działa. Panel czeka na nowszą obserwację runnera.";
  } else if (data?.effectiveFreshness === "offline") {
    text =
      data.mode === "demo"
        ? "Nagranie demonstracyjne zakończone."
        : "Źródło jest niedostępne. Brak potwierdzonej dostępności.";
  } else if (data?.camera_integrity.status === "mismatch") {
    text =
      "Widok kamery się zmienił. Potwierdzona dostępność została wycofana.";
    tone = "danger";
  } else if (data?.effectiveFreshness === "unverified") {
    text = "Oczekiwanie na aktualny obraz i potwierdzenie zgodności widoku.";
  }
  target.hidden = !text;
  if (target.textContent !== text) target.textContent = text;
  target.className = `notice ${tone}`;
}

function choose(id) {
  selected = id;
  render();
}

function svgNode(tag, attrs) {
  const item = document.createElementNS(svgNs, tag);
  for (const [key, value] of Object.entries(attrs))
    item.setAttribute(key, value);
  return item;
}

function layout(data) {
  const geometry = data?.geometry;
  const slots = data?.slots ?? [];
  if (!slots.some((slot) => slot.slot_id === selected))
    selected = slots[0]?.slot_id ?? null;
  const key = JSON.stringify([
    geometry,
    slots.map((s) => [s.slot_id, s.effective_status]),
  ]);
  if (key !== lastLayoutKey) {
    lastLayoutKey = key;
    const diagram = el("diagram");
    diagram.replaceChildren();
    el("slot-list").replaceChildren();
    if (!slots.length) {
      el("geometry-label").textContent =
        "Fizyczne miejsca w konfiguracji runnera";
      diagram.hidden = false;
      diagram.append(
        node(
          "p",
          api.connection === "loading"
            ? "Oczekiwanie na dane stanowisk…"
            : "Brak danych stanowisk",
        ),
      );
    } else if (geometry) {
      el("geometry-label").textContent =
        "Schemat zatwierdzonych projekcji ROI · bez obrazu z kamery";
      diagram.hidden = false;
      const [width, height] = geometry.image_size;
      const svg = svgNode("svg", {
        viewBox: `0 0 ${width} ${height}`,
        "aria-label": "Projekcje miejsc parkingowych",
      });
      const byId = new Map(slots.map((slot) => [slot.slot_id, slot]));
      for (const { slot_id: id, vertices } of geometry.projections) {
        const slot = byId.get(id);
        if (!slot) continue;
        const [state, tone] = labels[slot.effective_status];
        const points = projectionPoints(vertices, geometry.image_size);
        const center = points.reduce(
          ([sx, sy], [x, y]) => [
            sx + x / points.length,
            sy + y / points.length,
          ],
          [0, 0],
        );
        const group = svgNode("g", {
          class: `roi ${tone}`,
          tabindex: 0,
          role: "button",
          "aria-label": `${id} · ${state}`,
          "data-slot": id,
        });
        const label = svgNode("text", {
          x: center[0],
          y: center[1],
          "text-anchor": "middle",
          "dominant-baseline": "middle",
          "font-size": Math.max(width, height) / 40,
        });
        label.textContent = id;
        group.append(
          svgNode("polygon", {
            points: points.map((p) => p.join(",")).join(" "),
          }),
          label,
        );
        group.addEventListener("click", () => choose(id));
        group.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            choose(id);
          }
        });
        svg.append(group);
      }
      diagram.append(svg);
    } else {
      diagram.hidden = true;
      el("geometry-label").textContent =
        "Widok kart · brak zgodnej geometrii do wyświetlenia";
    }
    for (const slot of slots) {
      const [state, tone] = labels[slot.effective_status];
      const button = node("button", undefined, "slot");
      button.type = "button";
      button.dataset.slot = slot.slot_id;
      button.append(
        node("strong", slot.slot_id),
        node("span", state, `slot-state ${tone}`),
      );
      button.addEventListener("click", () => choose(slot.slot_id));
      el("slot-list").append(button);
    }
  } else if (!slots.length && api.connection !== "loading") {
    el("diagram").firstElementChild.textContent = "Brak danych stanowisk";
  }
  for (const item of document.querySelectorAll("[data-slot]")) {
    item.setAttribute("aria-pressed", item.dataset.slot === selected);
  }
}

function render() {
  const data = api.read(clock());
  clearTimeout(expiryTimer);
  if (data?.nextUpdateMs !== null && data?.nextUpdateMs !== undefined) {
    expiryTimer = setTimeout(render, Math.max(1, data.nextUpdateMs));
  }
  badge(
    "api-status",
    {
      loading: ["◷ Łączenie z API", "quiet"],
      connected: ["✓ API połączone", "good"],
      disconnected: ["! Brak API", "danger"],
      error: ["! Błąd odczytu", "danger"],
    }[api.connection],
  );

  const view = data?.camera_integrity;
  const mode =
    data?.mode === "demo"
      ? ["▷ Nagranie demo", "warning"]
      : data?.effectiveFreshness === "current" && !data.viewExpired
        ? ["● Na żywo", "good"]
        : data?.effectiveFreshness === "stale" || data?.viewExpired
          ? ["⊘ Dane nieaktualne", "danger"]
          : data?.mode
            ? ["◷ Monitoring wstrzymany", "warning"]
            : ["◷ Oczekiwanie", "quiet"];
  badge("mode", mode);
  el("scope").textContent = data
    ? `${data.parking_id}  /  ${data.camera_id}`
    : "Oczekiwanie na dane runnera";
  el("updated").textContent = `Publikacja: ${formatted(data?.published_at)}`;
  notice(data);
  counters(data);
  layout(data);
  badge(
    "view-status",
    data?.viewExpired
      ? ["⊘ Wygasł", "danger"]
      : (viewLabels[view?.status] ?? ["—", "quiet"]),
  );
  const cameraRows = [
    [
      "Źródło",
      data?.mode === "demo" && data.source.health === "streaming"
        ? "Wideo · stan z nagrania"
        : (sources[data?.source.health] ?? "—"),
    ],
    ["Ostatnia kontrola", formatted(view?.checked_at)],
    [
      data?.mode === "demo" ? "Ważność na osi nagrania" : "Ważność dowodu",
      formatted(view?.valid_until),
    ],
    [
      "Powód",
      reasonText(
        data?.viewExpired
          ? "view_verification_expired"
          : (view?.reason ?? data?.source.reason),
      ),
    ],
    ["Pominięte klatki", data?.skipped_frames],
  ];
  if (JSON.stringify(cameraRows) !== lastCameraKey) {
    lastCameraKey = JSON.stringify(cameraRows);
    details("camera-info", cameraRows);
  }
  const slot = data?.slots.find((row) => row.slot_id === selected);
  badge("slot-status", labels[slot?.effective_status] ?? ["—", "quiet"]);
  const detailRows = [
    ["ParkingSlotId", slot?.slot_id, "code"],
    ["Stan surowy", slot?.raw_state, "code"],
    ["Temporalny · runner", slot?.temporal_status, "code"],
    [
      "Potwierdzone wolne",
      slot ? (slot.confirmed_available ? "Tak" : "Nie") : "—",
    ],
    ["Powód unknown", reasonText(slot?.unknown_reason)],
    [
      data?.mode === "demo"
        ? "Obserwacja · oś nagrania"
        : "Ostatnia obserwacja",
      formatted(data?.captured_at),
    ],
    [
      "Wiek dowodu przy zapisie",
      slot?.last_evidence_age_seconds == null
        ? "—"
        : `${slot.last_evidence_age_seconds.toFixed(2)} s`,
    ],
    ["Numer sekwencji", data?.sequence],
    [
      "Model · urządzenie",
      data
        ? [data.model.backend, data.model.device].filter(Boolean).join(" · ")
        : "—",
    ],
    ["SHA checkpointu", data?.model.checkpoint_sha256, "code"],
    [
      "Widok kamery",
      data?.viewExpired ? "Dowód wygasł" : viewLabels[view?.status]?.[0],
    ],
  ];
  if (JSON.stringify(detailRows) !== lastDetailKey) {
    lastDetailKey = JSON.stringify(detailRows);
    details("slot-info", detailRows);
  }
}

function renderHistory() {
  const target = el("history");
  if (historyError) {
    target.replaceChildren(
      node("p", "Dziennik chwilowo niedostępny.", "empty"),
    );
    return;
  }
  if (latestHistory === null) return;
  const changedOnly = el("changes-only").checked;
  const rows = [...latestHistory]
    .reverse()
    .filter(
      (event) =>
        !changedOnly ||
        event.event !== "observation" ||
        event.slots.some((slot) => slot.changed),
    );
  if (!rows.length) {
    target.replaceChildren(
      node(
        "p",
        changedOnly
          ? "Brak zmian w ostatnich wpisach."
          : "Dziennik nie zawiera jeszcze zdarzeń.",
        "empty",
      ),
    );
    return;
  }
  const table = node("table");
  const header = table.createTHead().insertRow();
  for (const title of ["Czas", "Zdarzenie", "Stanowiska i szczegóły"]) {
    const th = node("th", title);
    th.scope = "col";
    header.append(th);
  }
  const body = table.createTBody();
  for (const event of rows) {
    const row = body.insertRow();
    row.dataset.event = `${event.run_id}/${event.event_index}`;
    const time = row.insertCell();
    const replay = event.source_kind === "video";
    time.textContent = formatted(
      replay ? event.observed_at : event.published_at,
      timeFormat,
    );
    if (replay) time.append(node("div", "Oś nagrania", "mini"));
    const name = row.insertCell();
    name.textContent = event.session_reset
      ? `${events[event.event]} · reset sesji`
      : events[event.event];
    const summary = row.insertCell();
    if (event.event !== "observation") {
      const status = `${viewLabels[event.camera_status][0]} · ${sources[event.source_health]}`;
      summary.append(node("div", status));
      if (event.camera_reason || event.source_reason) {
        summary.append(
          node(
            "div",
            reasonText(event.camera_reason ?? event.source_reason),
            "mini",
          ),
        );
      }
    }
    for (const slot of event.slots.filter((s) => !changedOnly || s.changed)) {
      const change = slot.changed
        ? `${slot.previous_state} / ${slot.previous_temporal_status} → `
        : "";
      const reason = slot.unknown_reason
        ? ` · ${reasonText(slot.unknown_reason)}`
        : "";
      summary.append(
        node(
          "div",
          `${slot.slot_id} · ${change}${slot.state} / ${slot.temporal_status}${reason}`,
        ),
      );
    }
  }
  target.replaceChildren(table);
}

async function get(url) {
  const response = await fetch(url, {
    cache: "no-store",
    signal: AbortSignal.timeout(2500),
  });
  const content = await response.json();
  if (!response.ok) throw new Error(content.error ?? "invalid_response");
  return content;
}

async function pollStatus() {
  if (statusInFlight) return;
  statusInFlight = true;
  const start = clock();
  try {
    api.accept(await get("/api/v1/status"), start);
  } catch (error) {
    api.fail(
      error instanceof TypeError || error.name === "TimeoutError"
        ? "network"
        : error.message,
    );
  } finally {
    statusInFlight = false;
    render();
    clearTimeout(statusTimer);
    statusTimer = setTimeout(pollStatus, document.hidden ? 4500 : 1000);
  }
}

async function pollHistory() {
  try {
    const result = await get("/api/v1/history?limit=24");
    if (result.schema_version !== 1 || !Array.isArray(result.events))
      throw new Error("invalid_history");
    const changed =
      historyError ||
      JSON.stringify(latestHistory) !== JSON.stringify(result.events);
    historyError = false;
    latestHistory = result.events;
    if (changed) renderHistory();
  } catch {
    if (!historyError) {
      historyError = true;
      renderHistory();
    }
  } finally {
    setTimeout(pollHistory, document.hidden ? 7000 : 2500);
  }
}

el("changes-only").addEventListener("change", renderHistory);
document.addEventListener("visibilitychange", () => {
  render();
  if (!document.hidden) {
    clearTimeout(statusTimer);
    pollStatus();
  }
});
render();
pollStatus();
pollHistory();
