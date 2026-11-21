import assert from "node:assert/strict";
import test from "node:test";
import { DashboardState, projectionPoints } from "../static/freshness.js";

const base = Date.parse("2026-11-21T12:00:00Z");
const clock = (ms) => ({ wall: base + ms, mono: ms });
const iso = (ms) => new Date(base + ms).toISOString().replace("Z", "000+00:00");
const row = (id, status, until) => ({
  slot_id: id,
  effective_status: status,
  effective_until: until,
});
const doc = (published = -100, captured = -150, session = "demo/run/0") => ({
  schema_version: 1,
  mode: "live",
  freshness: "current",
  published_at: iso(published),
  publication_valid_until: iso(published + 5000),
  server_time: iso(Math.max(published, 0)),
  captured_at: iso(captured),
  session_id: session,
  camera_integrity: { status: "verified", valid_until: iso(3600) },
  slots: [
    row("a", "confirmed_free", iso(1000)),
    row("b", "held", iso(3600)),
    row("c", "unknown", null),
    row("d", "stale", null),
  ],
});

test("expiry needs no response and does not create occupied", () => {
  const state = new DashboardState();
  state.accept(doc(), clock(0));
  assert.equal(state.read(clock(999)).availability_counts.confirmed_free, 1);
  assert.equal(state.read(clock(999)).nextUpdateMs, 1);
  assert.equal(state.read(clock(1000)).availability_counts.confirmed_free, 0);
  assert.equal(state.read(clock(1000)).availability_counts.stale, 2);
  assert.equal(
    state.read(clock(1000)).availability_counts.confirmed_occupied,
    0,
  );
  assert.equal(state.read(clock(3600)).viewExpired, true);
  assert.equal(state.read(clock(4900)).effectiveFreshness, "stale");
});

test("delayed response cannot extend availability", () => {
  const state = new DashboardState();
  state.accept(doc(), clock(0));
  assert.equal(state.read(clock(1300)).availability_counts.confirmed_free, 0);
});

test("API outage clears active availability and ignores repeated evidence after reconnect", () => {
  const state = new DashboardState();
  const original = doc();
  state.accept(original, clock(0));
  state.fail("network");
  assert.equal(state.read(clock(200)).availability_counts.confirmed_free, 0);
  assert.equal(state.connection, "disconnected");
  state.accept(original, clock(250));
  assert.equal(state.read(clock(260)).effectiveFreshness, "awaiting");
  state.accept(doc(270, -150), clock(270));
  assert.equal(state.read(clock(280)).availability_counts.confirmed_free, 0);
  state.accept(doc(320, 280), clock(320));
  assert.equal(state.read(clock(340)).availability_counts.confirmed_free, 1);
});

test("accepts legitimate new session and refuses older publication across sessions", () => {
  const state = new DashboardState();
  state.accept(doc(), clock(0));
  state.fail("invalid_json");
  state.accept(doc(550, 500, "demo/new/0"), clock(550));
  assert.equal(state.read(clock(560)).availability_counts.confirmed_free, 1);
  state.accept(doc(-1000, -1050, "demo/old/0"), clock(600));
  assert.equal(state.read(clock(650)).session_id, "demo/new/0");
});

test("publication order preserves time finer than a millisecond", () => {
  const state = new DashboardState();
  const early = { ...doc(), published_at: "2026-11-21T11:59:59.900001+00:00" };
  const late = { ...doc(), published_at: "2026-11-21T11:59:59.900002+00:00" };
  state.accept(late, clock(0));
  state.accept(early, clock(100));
  assert.equal(state.data.published_at, late.published_at);
});

test("demo does not use historical PTS as a wall-clock deadline", () => {
  const state = new DashboardState();
  const data = {
    ...doc(),
    mode: "demo",
    publication_valid_until: null,
    slots: [row("a", "confirmed_free", null)],
  };
  state.accept(data, clock(0));
  assert.equal(state.read(clock(80000)).availability_counts.confirmed_free, 1);
  assert.equal(state.read(clock(80000)).mode, "demo");
});

test("rejects invalid active live data without an expiration", () => {
  const state = new DashboardState();
  assert.throws(
    () =>
      state.accept(
        {
          ...doc(),
          slots: [row("slot", "confirmed_free", null)],
        },
        clock(0),
      ),
    /invalid_response/,
  );
});

test("multiple irregular ROI keep every vertex and the full frame aspect ratio", () => {
  const first = [
    [0, 0],
    [0.7, 0.3],
    [0.8, 0.8],
    [0.2, 1],
    [0.1, 0.5],
  ];
  const second = [
    [0, 0],
    [1, 0],
    [0.5, 0.25],
  ];
  assert.deepEqual(projectionPoints(first, [101, 201]), [
    [0, 0],
    [70, 60],
    [80, 160],
    [20, 200],
    [10, 100],
  ]);
  assert.deepEqual(projectionPoints(second, [101, 201]), [
    [0, 0],
    [100, 0],
    [50, 50],
  ]);
});
