from math import fsum, hypot, isfinite

from .line_segment import LineSegment
from .segment_group import SegmentGroup


def fit_group_line(group: SegmentGroup, *, segment_id: int) -> LineSegment:
    """Fit a pixel-space TLS line to equally weighted source endpoints.

    Project all endpoints onto the principal axis through their centroid and
    return the extreme projections. The dominant coordinate increases (x wins
    ties), independent of source order and endpoint direction. This extent may
    span unpainted gaps; the unchanged group remains the source of coverage.

    The caller allocates a fresh ID above ALL source IDs in the extraction,
    then increments it for each fitted group. Only collisions within this group
    can be checked here. Keep fitted lines separate from raw LSD segments:
    these IDs are local to one extraction/fitting run, carry no provenance by
    themselves and can change when groups change. Even a singleton gets a new
    object and ID; no source fragment is repurposed as a representative.

    Raise ValueError for empty groups, nonfinite coordinates, identical points,
    ambiguous principal directions or geometry not representable as finite
    endpoints. Validation here does not change the source models' contracts.
    """
    if not group.segments:
        raise ValueError("Segment group must contain at least one segment.")
    if segment_id <= max(segment.segment_id for segment in group.segments):
        raise ValueError("Fitted segment ID must be above source segment IDs.")

    points = tuple(
        point for segment in group.segments for point in (segment.start, segment.end)
    )
    if not all(isfinite(value) for point in points for value in point):
        raise ValueError("Source endpoints must have finite coordinates.")

    count = len(points)
    cx = fsum(x / count for x, _ in points)
    cy = fsum(y / count for _, y in points)
    offsets = tuple((x - cx, y - cy) for x, y in points)
    scale = max(abs(value) for point in offsets for value in point)
    if not isfinite(scale):
        raise ValueError("Source coordinate range is too large to fit.")
    if scale == 0:
        raise ValueError("Fitting requires at least two distinct points.")

    # Scaling avoids overflow/underflow when forming the 2D scatter matrix.
    offsets = tuple((x / scale, y / scale) for x, y in offsets)
    xx = fsum(x * x for x, _ in offsets)
    xy = fsum(x * y for x, y in offsets)
    yy = fsum(y * y for _, y in offsets)
    eigenvalue_gap = hypot(xx - yy, 2 * xy)
    if eigenvalue_gap <= 1e-12 * (xx + yy):
        raise ValueError("Source endpoints have no unique principal direction.")

    # Choose the eigenvector formula without subtracting nearly equal values.
    if xx >= yy:
        dx, dy = eigenvalue_gap + (xx - yy), 2 * xy
    else:
        dx, dy = 2 * xy, eigenvalue_gap + (yy - xx)
    norm = hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    projections = tuple(x * ux + y * uy for x, y in offsets)
    start = (
        cx + scale * (min(projections) * ux),
        cy + scale * (min(projections) * uy),
    )
    end = (
        cx + scale * (max(projections) * ux),
        cy + scale * (max(projections) * uy),
    )
    if not all(isfinite(value) for value in (*start, *end)) or start == end:
        raise ValueError("Fitted endpoints cannot represent a finite line.")

    return LineSegment(segment_id=segment_id, start=start, end=end)
