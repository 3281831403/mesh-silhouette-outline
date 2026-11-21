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


class OutlineError(Exception):
    """Raised when an outline request is invalid or cannot be completed."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def depth_offset(positions, bias):
    """Return screen-space points shifted only in the depth component.

    Every point ``(x, y, z)`` becomes ``(x, y, z + bias)``.  ``x`` and ``y``
    must not change: moving a sloped surface along an oblique view direction
    would also slide it on screen and cause colour bleeding, while a pure
    depth bias only resolves z-fighting.  ``bias`` is in screen-space depth
    units and may be positive or negative.  Do not mutate ``positions``.

    Returns a new list of 3-tuples accurate to 1e-6.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
