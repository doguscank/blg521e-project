"""Geometry and polygon helpers for 2D (x, z) world."""

from __future__ import annotations

import math

from self_parking.core.types import Point2D, WheelPoints
from self_parking.simulation.constants import (
    CHASSIS_BACK_WHEEL_SHIFT,
    CHASSIS_FRONT_WHEEL_SHIFT,
    CHASSIS_LENGTH,
    CHASSIS_WHEEL_WIDTH,
    CHASSIS_WIDTH,
)

Polygon = list[Point2D]
EPS = 1e-9


def transform_local_to_world(local_x: float, local_z: float, x: float, z: float, yaw: float) -> Point2D:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = x + local_x * cos_yaw + local_z * sin_yaw
    world_z = z - local_x * sin_yaw + local_z * cos_yaw
    return (world_x, world_z)


def car_body_polygon(x: float, z: float, yaw: float) -> Polygon:
    half_w = CHASSIS_WIDTH / 2
    half_l = CHASSIS_LENGTH / 2
    corners_local = [
        (-half_w, half_l),
        (half_w, half_l),
        (half_w, -half_l),
        (-half_w, -half_l),
    ]
    return [transform_local_to_world(px, pz, x, z, yaw) for px, pz in corners_local]


def wheel_points(x: float, z: float, yaw: float) -> WheelPoints:
    half_w = CHASSIS_WHEEL_WIDTH / 2
    return WheelPoints(
        fl=transform_local_to_world(-half_w, CHASSIS_FRONT_WHEEL_SHIFT, x, z, yaw),
        fr=transform_local_to_world(half_w, CHASSIS_FRONT_WHEEL_SHIFT, x, z, yaw),
        br=transform_local_to_world(half_w, CHASSIS_BACK_WHEEL_SHIFT, x, z, yaw),
        bl=transform_local_to_world(-half_w, CHASSIS_BACK_WHEEL_SHIFT, x, z, yaw),
    )


def rectangle_polygon(center_x: float, center_z: float, width: float, length: float, yaw: float = 0.0) -> Polygon:
    half_w = width / 2
    half_l = length / 2
    corners_local = [
        (-half_w, half_l),
        (half_w, half_l),
        (half_w, -half_l),
        (-half_w, -half_l),
    ]
    return [transform_local_to_world(px, pz, center_x, center_z, yaw) for px, pz in corners_local]


def _sub(a: Point2D, b: Point2D) -> Point2D:
    return (a[0] - b[0], a[1] - b[1])


def _cross(a: Point2D, b: Point2D) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: Point2D, b: Point2D) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _edges(poly: Polygon) -> list[Point2D]:
    return [
        _sub(poly[(idx + 1) % len(poly)], poly[idx])
        for idx in range(len(poly))
    ]


def _normal(edge: Point2D) -> Point2D:
    return (-edge[1], edge[0])


def _project(poly: Polygon, axis: Point2D) -> tuple[float, float]:
    projections = [_dot(point, axis) for point in poly]
    return min(projections), max(projections)


def polygons_intersect(poly_a: Polygon, poly_b: Polygon) -> bool:
    axes = [_normal(edge) for edge in _edges(poly_a)] + [_normal(edge) for edge in _edges(poly_b)]
    for axis in axes:
        min_a, max_a = _project(poly_a, axis)
        min_b, max_b = _project(poly_b, axis)
        if max_a < min_b or max_b < min_a:
            return False
    return True


def intersects_any(polygon: Polygon, obstacles: list[Polygon]) -> bool:
    return any(polygons_intersect(polygon, obstacle) for obstacle in obstacles)


def ray_segment_intersection(
    origin: Point2D,
    direction: Point2D,
    segment_start: Point2D,
    segment_end: Point2D,
) -> float | None:
    """Return ray distance t when ray(origin + t*direction) intersects segment."""
    p = origin
    r = direction
    q = segment_start
    s = _sub(segment_end, segment_start)

    r_cross_s = _cross(r, s)
    q_minus_p = _sub(q, p)

    if abs(r_cross_s) < EPS:
        return None

    t = _cross(q_minus_p, s) / r_cross_s
    u = _cross(q_minus_p, r) / r_cross_s

    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def raycast_polygon(origin: Point2D, angle: float, polygon: Polygon, max_distance: float) -> float | None:
    direction = (math.sin(angle), math.cos(angle))
    nearest: float | None = None

    for idx in range(len(polygon)):
        start = polygon[idx]
        end = polygon[(idx + 1) % len(polygon)]
        distance = ray_segment_intersection(origin, direction, start, end)
        if distance is None or distance > max_distance:
            continue
        if nearest is None or distance < nearest:
            nearest = distance

    return nearest


def raycast_polygons(
    origin: Point2D,
    angle: float,
    obstacles: list[Polygon],
    max_distance: float,
) -> float | None:
    nearest: float | None = None
    for obstacle in obstacles:
        distance = raycast_polygon(origin, angle, obstacle, max_distance)
        if distance is None:
            continue
        if nearest is None or distance < nearest:
            nearest = distance
    return nearest
