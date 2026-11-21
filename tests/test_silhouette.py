"""Red tests for the screen-space silhouette outline task."""

import math

import pytest

from outline import (
    OutlineError,
    build_outline_mesh,
    depth_offset,
    expand_edges,
    extract_edges,
    face_normal,
    is_silhouette,
    outline_batch,
    validate_mesh,
)

TOL = 1e-6
VIEW = (0.0, 0.0, 1.0)


def _assert_close(actual, expected, tol=TOL):
    """Compare nested point/vector sequences with the task tolerance."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if isinstance(got, (tuple, list)):
            assert isinstance(want, (tuple, list))
            _assert_close(got, want, tol)
        else:
            assert abs(got - want) <= tol


def _front_facing_quad():
    return {
        "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                      (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        "indices": [(0, 1, 2), (0, 2, 3)],
    }


def _folded_pair():
    # Triangle (0, 1, 2) faces the camera; triangle (2, 1, 3) is back-facing.
    # They share edge {1, 2}, which is listed in opposite directions.
    return {
        "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                      (0.0, 1.0, 0.0), (-1.0, 0.0, -1.0)],
        "indices": [(0, 1, 2), (2, 1, 3)],
    }


def _tetrahedron(offset_x):
    positions = [
        (offset_x + 0.0, 0.0, 0.0),
        (offset_x + 1.0, 0.0, 0.0),
        (offset_x + 0.0, 1.0, 0.0),
        (offset_x + 0.0, 0.0, 1.0),
    ]
    indices = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    return {"positions": positions, "indices": indices}


def test_face_normal_degenerate_triangle_is_zero_without_nan():
    normal = face_normal((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert normal == pytest.approx((0.0, 0.0, 0.0), abs=TOL)
    assert all(not math.isnan(component) for component in normal)


def test_is_silhouette_normalizes_view_direction():
    # cos(45 degrees) ~= 0.7071.  A non-unit view must not inflate the dot
    # product to 2.828 and wrongly pass threshold 0.8.
    normal = (math.sqrt(0.5), 0.0, math.sqrt(0.5))
    assert is_silhouette(normal, (0.0, 0.0, 4.0), 0.8) is False


def test_is_silhouette_includes_threshold_boundary():
    threshold = math.sqrt(0.5)
    normal = (math.sqrt(0.5), 0.0, math.sqrt(0.5))
    assert is_silhouette(normal, VIEW, threshold) is True


def test_extract_edges_deduplicates_reversed_shared_edge():
    # The shared edge appears as (1, 2) and (2, 1); an ordered-pair key would
    # emit it twice instead of merging it into one silhouette edge.
    edges = extract_edges(_folded_pair(), VIEW)
    assert edges == [(1, 2)]


def test_expand_edges_offsets_along_supplied_screen_normals():
    edges = [((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
             ((4.0, 0.0, 0.0), (4.0, 3.0, 0.0))]
    normals = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]

    expanded = expand_edges(edges, normals, 2.0)

    _assert_close(expanded, [
        ((0.0, 2.0, 0.0), (4.0, 2.0, 0.0)),
        ((6.0, 0.0, 0.0), (6.0, 3.0, 0.0)),
    ])


def test_build_outline_mesh_winds_triangles_towards_camera():
    mesh = build_outline_mesh([((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))], 1.0)

    positions = mesh["positions"]
    indices = mesh["indices"]
    assert len(positions) == 4
    assert len(indices) == 2

    for triangle in indices:
        p0, p1, p2 = (positions[index] for index in triangle)
        cross_z = ((p1[0] - p0[0]) * (p2[1] - p0[1])
                   - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        assert cross_z > TOL


def test_depth_offset_changes_only_screen_depth():
    positions = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    shifted = depth_offset(positions, 0.5)

    _assert_close(shifted, [(1.0, 2.0, 3.5), (4.0, 5.0, 6.5)])
    assert positions == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_validate_mesh_reports_out_of_bounds_index():
    mesh = {
        "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        "indices": [(0, 1, 9)],
    }
    problems = validate_mesh(mesh)
    assert problems
    assert any("index" in problem.lower() or "bound" in problem.lower()
               for problem in problems)


def test_outline_batch_keeps_meshes_independent():
    meshes = [_tetrahedron(0.0), _tetrahedron(10.0)]
    result = outline_batch(meshes, {"view_dir": VIEW, "width": 2.0})

    assert result["count"] == 2
    assert result["order"] == [0, 1]
    first, second = result["outlines"]
    assert first["positions"] is not second["positions"]
    assert first["indices"] is not second["indices"]

    second_positions = list(second["positions"])
    first["positions"].append((99.0, 99.0, 99.0))
    assert second["positions"] == second_positions


def test_expand_edges_rejects_zero_width():
    with pytest.raises(OutlineError):
        expand_edges([((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))],
                     [(0.0, 1.0, 0.0)], 0.0)


def test_extract_edges_front_facing_plane_returns_no_edges():
    assert extract_edges(_front_facing_quad(), VIEW) == []


def test_is_silhouette_threshold_one_only_accepts_direct_facing():
    directly_facing = (0.0, 0.0, 1.0)
    tilted = (math.sin(math.radians(1.0)), 0.0, math.cos(math.radians(1.0)))

    assert is_silhouette(directly_facing, VIEW, 1.0) is True
    assert is_silhouette(tilted, VIEW, 1.0) is False


def test_extract_edges_empty_mesh_raises_outline_error():
    with pytest.raises(OutlineError):
        extract_edges({"positions": [], "indices": []}, VIEW)


def test_build_outline_mesh_rejects_zero_width():
    with pytest.raises(OutlineError):
        build_outline_mesh([((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))], 0.0)


def test_outline_batch_preserves_input_order():
    meshes = [_tetrahedron(0.0), _tetrahedron(100.0)]
    result = outline_batch(meshes, {"view_dir": VIEW, "width": 1.0})

    assert result["order"] == [0, 1]
    assert result["count"] == 2
    assert len(result["outlines"]) == 2

    first = result["outlines"][0]["positions"]
    second = result["outlines"][1]["positions"]
    assert max(point[0] for point in first) < min(point[0] for point in second)
