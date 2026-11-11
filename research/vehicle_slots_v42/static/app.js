"use strict";

const batch = window.PARKVISION_BATCH;
const byId = (id) => document.getElementById(id);
const examples = batch.examples;
const keys = new Map(examples.map((example) => [example.key, example]));
const storageKey = "parkvision-v42:" + batch.batch_sha256;
const statuses = { pending: "oczekuje", accepted: "zaakceptowane", rejected: "odrzucone", needs_review: "wymaga kontroli" };
const relations = {
  owning_slot: "Pojazd zajmuje oceniane stanowisko",
  perspective_overlap_only: "Tylko nakładanie perspektywiczne",
  physically_blocks_neighbor: "Fizycznie blokuje sąsiednie stanowisko",
  uncertain_or_occluded: "Nierozstrzygalne lub zasłonięte",
  not_assessed: "Brak oceny",
};
const geometries = {
  supported_by_visible_boundaries: "Zgodna z widocznymi granicami",
  partially_supported: "Częściowo potwierdzona",
  unverifiable: "Brak możliwości weryfikacji",
};
const visibilities = { clear: "Czytelna", partial: "Częściowo zasłonięta", obscured: "Zasłonięta" };
const confidences = { high: "wysoka", medium: "średnia", low: "niska" };
const controls = ["reviewer", "status", "relation", "note", "geometry", "visibility"];
let index = 0;
let dirty = false;
let decisions = new Map();
let author = "";

function message(value, error = false) {
  byId("message").textContent = value;
  byId("message").dataset.error = String(error);
}

function addOptions(select, values, placeholder) {
  select.append(new Option(placeholder, ""));
  for (const [value, label] of Object.entries(values)) select.append(new Option(label, value));
}

function pairKey(pair) {
  const fields = ["image_id", "car_annotation_id", "spot_annotation_id"];
  if (!pair || Object.keys(pair).length !== 3 || !fields.every((f) => Number.isSafeInteger(pair[f]) && pair[f] > 0)) {
    throw new Error("Nieprawidłowe identyfikatory pary.");
  }
  return fields.map((field) => pair[field]).join(":");
}

function loadProgress(value) {
  if (!value || value.schema_version !== 1 || value.batch_sha256 !== batch.batch_sha256 || !Array.isArray(value.decisions)) {
    throw new Error("Plik postępu nie odpowiada tej partii.");
  }
  const imported = new Map();
  for (const d of value.decisions) {
    const key = pairKey(d.pair);
    const accepted = d.review_status === "accepted";
    const allowed = ["pair", "review_status", "operator_relation", "reviewer", "reviewed_at", "note", "geometry", "visibility"];
    if (Object.keys(d).some((k) => !allowed.includes(k)) ||
        !keys.has(key) || imported.has(key) || !Object.hasOwn(statuses, d.review_status) ||
        (accepted ? (!Object.hasOwn(relations, d.operator_relation) || d.operator_relation === "not_assessed") : d.operator_relation !== null) ||
        typeof d.reviewer !== "string" || !d.reviewer.trim() ||
        typeof d.note !== "string" || !d.note.trim() ||
        (d.geometry != null && !Object.hasOwn(geometries, d.geometry)) ||
        (d.visibility != null && !Object.hasOwn(visibilities, d.visibility)) ||
        typeof d.reviewed_at !== "string" ||
        !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(d.reviewed_at) ||
        !Number.isFinite(Date.parse(d.reviewed_at))) {
      throw new Error("Nieprawidłowa lub powtórzona decyzja: " + key);
    }
    imported.set(key, d);
  }
  decisions = imported;
  author = [...decisions.values()].at(-1)?.reviewer || "";
}

function payload() {
  return { schema_version: 1, batch_sha256: batch.batch_sha256, decisions: [...decisions.values()] };
}

function persist() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(payload()));
    return true;
  } catch {
    message("Zapis w przeglądarce jest niedostępny. Pobierz postęp JSON.", true);
    return false;
  }
}

function picture(id, source) {
  byId(id).src = source;
  byId(id + "-link").href = source;
}

function updateStatus() {
  const accepted = [...decisions.values()].filter((d) => d.review_status === "accepted").length;
  byId("progress").textContent = `Zaakceptowane ${accepted} / ${examples.length} · zapisane ${decisions.size}`;
  for (const [i, ex] of examples.entries()) {
    const saved = decisions.get(ex.key);
    byId("examples").options[i].text = ex.id + " · " + (saved ? statuses[saved.review_status] : "oczekuje");
  }
}

function show() {
  const ex = examples[index];
  const saved = decisions.get(ex.key);
  byId("examples").selectedIndex = index;
  byId("prev").disabled = index === 0;
  byId("next").disabled = index === examples.length - 1;
  byId("case-title").textContent = ex.id + " · " + ex.capture_time.replace("T", " ");
  byId("identity").textContent = `Obraz ${ex.pair.image_id} · maska ${ex.pair.car_annotation_id} · stanowisko ${ex.pair.spot_annotation_id}`;
  byId("provenance").textContent = [
    "car_id " + ex.car_id, "camera_slot_key " + ex.camera_slot_key,
    "Parking " + ex.site_id, "Kamera " + ex.camera_id, "Rekord " + ex.registry_record,
    "Wersja MetaPKLot " + [...new Set(batch.versions)].join(", "),
    "JPEG SHA-256 " + ex.source.sha256, "Anotacje SHA-256 " + ex.source_subset.sha256,
    "Źródłowe powiązania stanowisk " + (ex.source_owners.join(", ") || "brak"),
    "Poligony sąsiednie na nakładce " + ex.neighbors.join(", "),
  ].join("\n");
  picture("detail-raw", ex.detail_original);
  picture("detail-overlay", ex.detail_overlay);
  picture("full-raw", ex.original_view);
  picture("full-overlay", ex.full_overlay);
  const suggested = ex.suggestion;
  byId("suggested-label").textContent = relations[suggested.suggested_relation] + " · pewność " + confidences[suggested.confidence];
  byId("suggested-note").textContent = suggested.relation_note;
  byId("suggested-geometry").textContent = geometries[suggested.geometry] + " · widoczność: " + visibilities[suggested.visibility].toLowerCase() + ". " + suggested.geometry_note;
  document.querySelector(".suggestion").open = false;
  byId("reviewer").value = saved?.reviewer ?? author;
  byId("status").value = saved?.review_status ?? "pending";
  byId("relation").value = saved?.operator_relation ?? "";
  byId("geometry").value = saved?.geometry ?? "";
  byId("visibility").value = saved?.visibility ?? "";
  byId("note").value = saved?.note ?? "";
  byId("saved-time").textContent = saved ? "Zapisano " + new Date(saved.reviewed_at).toLocaleString("pl-PL") : "Brak zapisanej oceny";
  updateRelationField();
  updateStatus();
  dirty = false;
}

function updateRelationField() {
  const active = byId("status").value === "accepted";
  byId("relation").disabled = !active;
  byId("relation").required = active;
  if (!active) byId("relation").value = "";
}

function mayDiscard() {
  return !dirty || window.confirm("Ocena ma niezapisane zmiany. Porzucić je?");
}

function go(to) {
  if (to >= 0 && to < examples.length && mayDiscard()) {
    index = to;
    show();
  } else {
    byId("examples").selectedIndex = index;
  }
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvQuote(value) {
  return '"' + String(value).replaceAll('"', '""') + '"';
}

addOptions(byId("relation"), Object.fromEntries(Object.entries(relations).filter(([k]) => k !== "not_assessed")), "Wybierz relację");
addOptions(byId("geometry"), geometries, "Brak oceny geometrii");
addOptions(byId("visibility"), visibilities, "Brak oceny widoczności");
for (const ex of examples) byId("examples").append(new Option(ex.id, ex.key));
try {
  const local = localStorage.getItem(storageKey);
  if (local) loadProgress(JSON.parse(local));
} catch (error) {
  message("Zapisany postęp jest niedostępny: " + error.message, true);
}
show();

byId("prev").addEventListener("click", () => go(index - 1));
byId("next").addEventListener("click", () => go(index + 1));
byId("examples").addEventListener("change", () => go(byId("examples").selectedIndex));
for (const id of controls) byId(id).addEventListener("input", () => { dirty = true; });
byId("status").addEventListener("change", updateRelationField);
window.addEventListener("beforeunload", (event) => { if (dirty) event.preventDefault(); });

byId("operator").addEventListener("submit", (event) => {
  event.preventDefault();
  const status = byId("status").value;
  const reviewer = byId("reviewer").value.trim();
  const note = byId("note").value.trim();
  const relation = status === "accepted" ? byId("relation").value : null;
  if (!reviewer || !note || (status === "accepted" && !relation)) {
    message("Podaj autora i uzasadnienie. Akceptacja wymaga wybranej relacji.", true);
    return;
  }
  const example = examples[index];
  decisions.set(example.key, {
    pair: example.pair, review_status: status, operator_relation: relation,
    reviewer, note, reviewed_at: new Date().toISOString(),
    geometry: byId("geometry").value || null, visibility: byId("visibility").value || null,
  });
  author = reviewer;
  show();
  if (persist()) message("Zapisano ocenę w tej przeglądarce.");
});

byId("export-progress").addEventListener("click", () => {
  if (dirty) return message("Zapisz bieżącą ocenę przed eksportem.", true);
  download("parkvision-ab-progress.json", JSON.stringify(payload(), null, 2), "application/json");
});
byId("import-progress").addEventListener("click", () => {
  if (mayDiscard()) byId("progress-file").click();
});
byId("progress-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (decisions.size && !window.confirm("Zastąpić zapisany postęp zawartością tego pliku?")) return;
    loadProgress(JSON.parse(await file.text()));
    show();
    if (persist()) message("Wczytano postęp z pliku.");
  } catch (error) {
    message(error.message, true);
  } finally {
    event.target.value = "";
  }
});
byId("export-csv").addEventListener("click", () => {
  if (dirty) return message("Zapisz bieżącą ocenę przed eksportem.", true);
  const rows = [["image_id", "car_annotation_id", "spot_annotation_id", "relation", "reviewer", "note"]];
  for (const d of decisions.values()) {
    if (d.review_status !== "accepted") continue;
    const p = d.pair;
    rows.push([p.image_id, p.car_annotation_id, p.spot_annotation_id, d.operator_relation, d.reviewer, `[${d.reviewed_at}] ${d.note}`]);
  }
  if (rows.length === 1) return message("Brak zaakceptowanych ocen do eksportu.");
  download("v41-reviews.csv", rows.map((row) => row.map(csvQuote).join(",")).join("\r\n") + "\r\n", "text/csv;charset=utf-8");
});
