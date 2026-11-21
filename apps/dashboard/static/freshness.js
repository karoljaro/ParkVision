// Browser may remove eligibility after a deadline or outage. Only the API can grant it.
const CATEGORIES = [
  "confirmed_free",
  "confirmed_occupied",
  "held",
  "unknown",
  "stale",
];
const ACTIVE = new Set(CATEGORIES.slice(0, 3));
const ISO_UTC = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00$/;

function timestamp(value) {
  const ms = Date.parse(value);
  if (!ISO_UTC.test(value ?? "") || !Number.isFinite(ms)) {
    throw new Error("invalid_response");
  }
  return ms + Number(value.slice(23, 26)) / 1000;
}

export function projectionPoints(vertices, [width, height]) {
  return vertices.map(([x, y]) => [x * (width - 1), y * (height - 1)]);
}

export class DashboardState {
  constructor() {
    this.data = null;
    this.connection = "loading";
    this.error = null;
    this.observationAtLoss = null;
  }

  accept(data, requestStart) {
    if (
      data?.schema_version !== 1 ||
      !["live", "demo"].includes(data.mode) ||
      !Array.isArray(data.slots) ||
      !data.camera_integrity
    ) {
      throw new Error("invalid_response");
    }
    timestamp(data.published_at);
    const serverTime = timestamp(data.server_time);
    if (data.mode === "live") {
      timestamp(data.publication_valid_until);
      if (data.captured_at !== null) timestamp(data.captured_at);
      if (data.camera_integrity.status === "verified")
        timestamp(data.camera_integrity.valid_until);
    }
    for (const row of data.slots) {
      if (!CATEGORIES.includes(row.effective_status))
        throw new Error("invalid_response");
      if (row.effective_until !== null) timestamp(row.effective_until);
      if (
        data.mode === "live" &&
        ACTIVE.has(row.effective_status) &&
        row.effective_until === null
      ) {
        throw new Error("invalid_response");
      }
    }
    this.connection = "connected";
    this.error = null;
    if (this.data && data.published_at <= this.data.published_at) return;
    if (
      this.observationAtLoss &&
      data.mode === "live" &&
      data.captured_at &&
      (data.session_id !== this.observationAtLoss.session_id ||
        data.captured_at > this.observationAtLoss.captured_at)
    ) {
      this.observationAtLoss = null;
    }
    this.data = data;
    this.requestStart = requestStart;
    this.serverTime = serverTime;
  }

  fail(error) {
    this.connection = error === "network" ? "disconnected" : "error";
    this.error = error;
    if (this.data?.mode === "live" && !this.observationAtLoss) {
      this.observationAtLoss = {
        session_id: this.data.session_id,
        captured_at: this.data.captured_at ?? "",
      };
    }
  }

  read(now) {
    if (!this.data) return null;
    const { data } = this;
    const live = data.mode === "live";
    // Starting the lease at request start accounts for network latency and clock skew.
    const elapsed = Math.max(
      now.mono - this.requestStart.mono,
      now.wall - this.requestStart.wall,
      0,
    );
    const effectiveTime = this.serverTime + elapsed;
    const outage = this.connection !== "connected";
    const awaiting = live && this.observationAtLoss !== null;
    const publicationExpired =
      live && effectiveTime >= timestamp(data.publication_valid_until);
    const counts = Object.fromEntries(
      CATEGORIES.map((category) => [category, 0]),
    );
    const deadlines = [];
    if (live && !publicationExpired)
      deadlines.push(timestamp(data.publication_valid_until));
    const viewDeadline =
      live && data.camera_integrity.status === "verified"
        ? timestamp(data.camera_integrity.valid_until)
        : null;
    const viewExpired = viewDeadline !== null && effectiveTime >= viewDeadline;
    const slots = data.slots.map((row) => {
      const deadline =
        row.effective_until === null ? null : timestamp(row.effective_until);
      const expired = deadline !== null && effectiveTime >= deadline;
      if (deadline !== null && !expired) deadlines.push(deadline);
      const status =
        ACTIVE.has(row.effective_status) &&
        (outage || awaiting || publicationExpired || viewExpired || expired)
          ? "stale"
          : row.effective_status;
      counts[status]++;
      return {
        ...row,
        effective_status: status,
        confirmed_available: status === "confirmed_free",
      };
    });
    const effectiveFreshness =
      outage || publicationExpired || viewExpired
        ? "stale"
        : awaiting
          ? "awaiting"
          : data.freshness;
    if (viewDeadline !== null && !viewExpired) deadlines.push(viewDeadline);
    return {
      ...data,
      slots,
      availability_counts: { physical_total: slots.length, ...counts },
      effectiveFreshness,
      viewExpired,
      nextUpdateMs: deadlines.length
        ? Math.min(...deadlines) - effectiveTime
        : null,
    };
  }
}
