"""Screen-space silhouette outline extraction.

Coordinate and unit conventions
-------------------------------
* Screen space has x pointing right, y pointing up and z pointing towards the
  camera.  The camera looks along -z, so a triangle whose vertices wind
  counter-clockwise when seen from the camera has a normal with a positive z
  component.
* ``view_dir`` is a direction in screen space, pointing from the surface
  towards the camera.  It is not required to be unit length; functions that
  compare directions normalise it internally.
* ``width`` values are measured in screen pixels.  Widths must be strictly
  positive.
* Floating-point comparisons on directions and geometry use a tolerance of
  ``1e-6``.  "Equal" means the absolute difference is at most that tolerance.

A mesh is a mapping of the form::

    {"positions": [(x, y, z), ...], "indices": [(i, j, k), ...]}

where every index tuple describes one triangle.
"""

import math
from collections.abc import Mapping, Sequence


TOL = 1e-6


class OutlineError(Exception):
    """Raised when an outline request is invalid or cannot be completed."""


def _is_real(value):
    """Return whether ``value`` is a finite, non-boolean real number."""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value))


def _is_vector(value, size=None):
    """Return whether ``value`` is a finite numeric vector of the given size."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if size is not None and len(value) != size:
        return False
    return all(_is_real(component) for component in value)


def _subtract(p, q):
    return tuple(p[i] - q[i] for i in range(len(p)))


def _cross(p, q):
    return (
        p[1] * q[2] - p[2] * q[1],
        p[2] * q[0] - p[0] * q[2],
        p[0] * q[1] - p[1] * q[0],
    )


def _normalise(vector):
    """Normalise a vector; return ``None`` for a zero-length vector."""
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-12 or not math.isfinite(length):
        return None
    return tuple(component / length for component in vector)


def face_normal(p0, p1, p2):
    """Return the unit face normal of the triangle ``(p0, p1, p2)``.

    ``p0``, ``p1`` and ``p2`` are 3D points in screen space.  The normal is
    ``normalize(cross(p1 - p0, p2 - p0))`` and therefore follows the winding
    rule: counter-clockwise as seen from the camera means the normal points
    towards the camera (+z).

    A degenerate triangle -- one with zero area, such as coincident or
    collinear points -- must return the zero vector ``(0.0, 0.0, 0.0)``.  It
    must never return NaN or infinity.  Each component is accurate to 1e-6.

    Returns a 3-tuple of floats.
    """
    e1 = _subtract(p1, p0)
    e2 = _subtract(p2, p0)
    normal = _cross(e1, e2)
    length = math.sqrt(sum(component * component for component in normal))
    # Degenerate (zero/near-zero area) or non-finite geometry: report the zero
    # vector instead of propagating NaN/infinity into later dot products.
    if length <= 1e-12 or not math.isfinite(length):
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in normal)


def is_silhouette(face_n, view_dir, threshold):
    """Return whether a face is front-facing enough to be silhouette-relevant.

    Both ``face_n`` and ``view_dir`` are screen-space directions and are
    normalised internally; callers may pass a non-unit ``view_dir``.  The
    result is ``True`` exactly when::

        dot(normalize(face_n), normalize(view_dir)) >= threshold

    The comparison is inclusive, with the 1e-6 tolerance: a cosine within
    1e-6 of ``threshold`` counts as equal.  ``threshold`` is a direction
    cosine in ``[-1.0, 1.0]``; ``threshold=1.0`` admits only a face whose
    normal points exactly at the camera.

    Raise ``OutlineError`` if either direction has zero length.
    """
    normal = _normalise(face_n)
    if normal is None:
        raise OutlineError("face normal has zero length and cannot be used")
    view = _normalise(view_dir)
    if view is None:
        raise OutlineError("view direction has zero length and cannot be used")
    cosine = sum(normal[i] * view[i] for i in range(len(normal)))
    # Inclusive boundary (>=), widened by the 1e-6 floating-point tolerance.
    return cosine + TOL >= threshold


def validate_mesh(mesh):
    """Return a list of human-readable problems; an empty list means valid.

    The function never raises for malformed input: it reports every problem it
    finds, including a missing or non-mapping ``mesh``.  Checks include:

    * missing ``positions`` or ``indices``, or an empty mesh;
    * a position that is not a finite 3-tuple;
    * a face that is not exactly three integer indices;
    * an index outside ``[0, len(positions))``;
    * a degenerate face (area at or below 1e-6).

    A mesh with any reported problem is invalid.  Problem messages are
    deterministic strings; the exact wording is not part of the API, but an
    out-of-range index is described with the word ``index`` or ``bounds``.
    """
    problems = []

    if not isinstance(mesh, Mapping):
        return ["mesh must be a mapping with 'positions' and 'indices'"]

    positions = mesh.get("positions")
    indices = mesh.get("indices")
    if positions is None or indices is None:
        problems.append("mesh is missing 'positions' or 'indices'")
        return problems
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        problems.append("'positions' must be a sequence of 3-tuples")
        return problems
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        problems.append("'indices' must be a sequence of index triples")
        return problems

    if len(positions) == 0 or len(indices) == 0:
        problems.append("mesh must contain at least one position and one face")
        return problems

    for vertex_index, point in enumerate(positions):
        if not _is_vector(point, size=3):
            problems.append(
                "position %d must be a finite 3-tuple of numbers" % vertex_index
            )

    positions_valid = not problems
    vertex_count = len(positions)

    for face_index, face in enumerate(indices):
        if not isinstance(face, Sequence) or isinstance(face, (str, bytes)) \
                or len(face) != 3:
            problems.append(
                "face %d must contain exactly three integer indices" % face_index
            )
            continue

        face_ok = True
        for slot, index in enumerate(face):
            # Booleans are rejected even though they subclass int: an index
            # must be a genuine integer.
            if not isinstance(index, int) or isinstance(index, bool):
                problems.append(
                    "face %d slot %d has non-integer index %r"
                    % (face_index, slot, index)
                )
                face_ok = False
            elif index < 0 or index >= vertex_count:
                problems.append(
                    "face %d references index %d outside bounds [0, %d)"
                    % (face_index, index, vertex_count)
                )
                face_ok = False

        # Degeneracy can only be judged for well-formed faces with valid
        # finite positions.
        if face_ok and positions_valid:
            p0, p1, p2 = (positions[face[0]], positions[face[1]],
                          positions[face[2]])
            cross = _cross(_subtract(p1, p0), _subtract(p2, p0))
            cross_norm = math.sqrt(
                sum(component * component for component in cross)
            )
            area = cross_norm / 2.0
            if area <= TOL:
                problems.append(
                    "face %d is degenerate (area %.3g at or below %g)"
                    % (face_index, area, TOL)
                )

    return problems


def extract_edges(mesh, view_dir):
    """Return the deduplicated silhouette edges of ``mesh``.

    ``mesh`` is a ``{"positions", "indices"}`` mapping in screen space.  The
    mesh is validated first with :func:`validate_mesh`; if that returns any
    problem, or if the mesh is empty, raise ``OutlineError``.

    A face is classified with ``is_silhouette(normal, view_dir, 0.0)``.  Only
    interior edges with exactly two incident triangles are considered: an
    edge is a silhouette edge when one incident face is front-facing and the
    other is not.  Open boundary edges are ignored.  Each silhouette edge is
    reported once as an unordered vertex-index pair ``(min(i, j), max(i, j))``;
    the returned list is sorted lexicographically for deterministic output.

    ``view_dir`` may be non-unit and is normalised internally.
    """
    problems = validate_mesh(mesh)
    if problems:
        raise OutlineError("invalid mesh: " + "; ".join(problems))

    positions = mesh["positions"]
    indices = mesh["indices"]

    # Key edges by their unordered vertex pair so reversed shared edges merge
    # into a single entry instead of being counted twice.
    incident = {}
    for face in indices:
        i, j, k = face
        normal = face_normal(positions[i], positions[j], positions[k])
        front_facing = is_silhouette(normal, view_dir, 0.0)
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            incident.setdefault(key, []).append(front_facing)

    silhouette_edges = []
    for edge, classifications in incident.items():
        # Boundary edges (one face) and non-manifold edges (three or more)
        # are ignored; a silhouette needs exactly one front and one back face.
        if len(classifications) == 2 and classifications[0] != classifications[1]:
            silhouette_edges.append(edge)

    silhouette_edges.sort()
    return silhouette_edges


def _check_width(width):
    if not _is_real(width):
        raise OutlineError("width must be a finite real number")
    if width <= 0.0:
        raise OutlineError("width must be strictly positive, got %r" % (width,))


def expand_edges(edges, normals, width):
    """Expand screen-space segments outwards along screen-space normals.

    ``edges`` is a sequence of ``(a, b)`` segments in screen space; an endpoint
    may be ``(x, y)`` or ``(x, y, z)``.  ``normals`` is a sequence of unit
    screen-space normals, one per edge, given in the same dimensionality as
    the endpoints.  The expanded segment is::

        (a + n * width, b + n * width)

    so both endpoints move by the same pixel distance and the outline width is
    uniform.  The supplied normal is the screen normal of the edge; per-vertex
    normals must not be used.  Widths are in pixels and ``width`` must be
    strictly positive.  Preserve input order and do not mutate the inputs.

    Raise ``OutlineError`` if ``width <= 0``, if ``edges`` and ``normals`` have
    different lengths, or if a normal has zero length.  Coordinates are
    returned as floats with 1e-6 accuracy.
    """
    _check_width(width)
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)) \
            or not isinstance(normals, Sequence) \
            or isinstance(normals, (str, bytes)):
        raise OutlineError("edges and normals must be sequences")
    if len(edges) != len(normals):
        raise OutlineError(
            "edges and normals must have the same length (%d vs %d)"
            % (len(edges), len(normals))
        )

    expanded = []
    for edge_index, (edge, normal) in enumerate(zip(edges, normals)):
        if not isinstance(edge, Sequence) or len(edge) != 2:
            raise OutlineError(
                "edge %d must be a pair of endpoints" % edge_index
            )
        a, b = edge
        if not _is_vector(a) or not _is_vector(b):
            raise OutlineError(
                "edge %d endpoints must be finite numeric points" % edge_index
            )
        if not _is_vector(normal) or len(normal) != len(a) or len(normal) != len(b):
            raise OutlineError(
                "normal %d must match the endpoint dimensionality" % edge_index
            )
        if sum(component * component for component in normal) <= 1e-12:
            raise OutlineError(
                "normal %d has zero length and cannot define an offset"
                % edge_index
            )

        offset = tuple(component * width for component in normal)
        expanded.append((
            tuple(float(a[axis] + offset[axis])
                  for axis in range(len(offset))),
            tuple(float(b[axis] + offset[axis])
                  for axis in range(len(offset))),
        ))
    return expanded


def build_outline_mesh(edges, width):
    """Build a renderable triangle mesh for the outline around ``edges``.

    Each screen-space segment ``(a, b)`` becomes a quad of total width
    ``width`` pixels, centred on the segment.  With ``d = normalize(b - a)``
    and the screen normal ``n = (-d.y, d.x, 0)``, the four corners are::

        [a - n * width / 2, a + n * width / 2,
         b - n * width / 2, b + n * width / 2]

    and the quad is split into two triangles.  Every triangle must wind
    counter-clockwise as seen from the camera, so its normal points towards
    +z and the outline survives back-face culling.  ``z`` is copied from the
    segment endpoints.

    Return ``{"positions": [(x, y, z), ...], "indices": [(i, j, k), ...]}``.
    An empty ``edges`` sequence returns an empty mesh.  ``width`` is in pixels
    and must be strictly positive; otherwise raise ``OutlineError``.  Geometry
    is accurate to 1e-6.
    """
    _check_width(width)

    positions = []
    triangles = []
    half = width / 2.0

    for edge_index, edge in enumerate(edges):
        a, b = edge
        if not _is_vector(a) or not _is_vector(b) or len(a) != len(b):
            raise OutlineError(
                "edge %d must join two finite points of equal dimension"
                % edge_index
            )

        direction = _normalise(_subtract(b, a))
        if direction is None:
            raise OutlineError(
                "edge %d has zero length and no screen normal" % edge_index
            )

        # Screen-space normal: perpendicular to the segment in xy, no z tilt,
        # so the strip keeps a uniform on-screen width.
        n = (-direction[1], direction[0], 0.0)
        az = float(a[2]) if len(a) > 2 else 0.0
        bz = float(b[2]) if len(b) > 2 else 0.0

        base = len(positions)
        # Corner order, per the documented contract:
        #   0: a - n*w/2, 1: a + n*w/2, 2: b - n*w/2, 3: b + n*w/2
        corners = [
            (float(a[0] - n[0] * half), float(a[1] - n[1] * half), az),
            (float(a[0] + n[0] * half), float(a[1] + n[1] * half), az),
            (float(b[0] - n[0] * half), float(b[1] - n[1] * half), bz),
            (float(b[0] + n[0] * half), float(b[1] + n[1] * half), bz),
        ]
        positions.extend(corners)

        # Both triangles wind counter-clockwise as seen from +z, giving a
        # positive-z normal so back-face culling keeps the outline visible.
        triangles.append((base + 0, base + 2, base + 1))
        triangles.append((base + 1, base + 2, base + 3))

    return {"positions": positions, "indices": triangles}


def depth_offset(positions, bias):
    """Return screen-space points shifted only in the depth component.

    Every point ``(x, y, z)`` becomes ``(x, y, z + bias)``.  ``x`` and ``y``
    must not change: moving a sloped surface along an oblique view direction
    would also slide it on screen and cause colour bleeding, while a pure
    depth bias only resolves z-fighting.  ``bias`` is in screen-space depth
    units and may be positive or negative.  Do not mutate ``positions``.

    Returns a new list of 3-tuples accurate to 1e-6.
    """
    return [
        (float(point[0]), float(point[1]), float(point[2] + bias))
        for point in positions
    ]


def outline_batch(meshes, config):
    """Generate outlines for several meshes as one all-or-nothing operation.

    ``meshes`` is a non-empty sequence of meshes in input order.  ``config``
    is a mapping with ``view_dir`` (screen-space direction) and ``width``
    (pixels, strictly positive).  Every mesh is validated before any outline
    is built.  If one mesh is invalid, raise ``OutlineError`` and produce no
    partial result.

    Each mesh is processed with its own edge set, normals and output lists;
    no mutable geometry may be shared between results.  On success return::

        {
            "outlines": [outline_mesh, ...],  # same order as ``meshes``
            "count": len(meshes),
            "order": [0, 1, ..., len(meshes) - 1],
        }

    The ``order`` field records the input order explicitly.  Raise
    ``OutlineError`` for an empty ``meshes`` sequence, a missing or invalid
    config, or ``width <= 0``.
    """
    if not isinstance(config, Mapping):
        raise OutlineError("config must be a mapping with 'view_dir' and 'width'")
    if "view_dir" not in config or "width" not in config:
        raise OutlineError("config must define 'view_dir' and 'width'")

    view_dir = config["view_dir"]
    width = config["width"]
    if not _is_vector(view_dir, size=3):
        raise OutlineError("config 'view_dir' must be a finite 3-vector")
    if _normalise(view_dir) is None:
        raise OutlineError("config 'view_dir' must not have zero length")
    _check_width(width)

    if not isinstance(meshes, Sequence) or isinstance(meshes, (str, bytes)):
        raise OutlineError("meshes must be a sequence of mesh mappings")
    if len(meshes) == 0:
        raise OutlineError("meshes sequence must not be empty")

    # Validate every mesh before building anything, so a single invalid mesh
    # aborts the whole batch with no partial output.
    for mesh_index, mesh in enumerate(meshes):
        problems = validate_mesh(mesh)
        if problems:
            raise OutlineError(
                "mesh %d is invalid: %s" % (mesh_index, "; ".join(problems))
            )

    outlines = []
    for mesh in meshes:
        # Every call builds fresh local edge maps and output lists: nothing
        # mutable is shared between meshes in the batch.
        edge_pairs = extract_edges(mesh, view_dir)
        positions = mesh["positions"]
        segments = [
            (tuple(positions[a]), tuple(positions[b])) for a, b in edge_pairs
        ]
        outlines.append(build_outline_mesh(segments, width))

    return {
        "outlines": outlines,
        "count": len(meshes),
        "order": list(range(len(meshes))),
    }
