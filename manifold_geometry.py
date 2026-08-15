"""Pure-Python geometry calculations for the Fusion manifold add-in.

All public lengths are millimetres. Fusion conversion to its internal
centimetre unit is deliberately kept out of this module so it can be tested
without Fusion installed.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Tuple


MODE_HEIGHT = "height"
MODE_ANGLE = "angle"
CURVE_EASE_BOTH = "ease_both"
CURVE_EASE_INLET = "ease_inlet"
CURVE_DIRECT = "direct"
CURVE_STRAIGHT = "straight"
CURVE_STYLES = (
    CURVE_EASE_BOTH,
    CURVE_EASE_INLET,
    CURVE_DIRECT,
    CURVE_STRAIGHT,
)
SECTION_SPACING_ADAPTIVE = "adaptive"
SECTION_SPACING_UNIFORM = "uniform"
SECTION_SPACING_STYLES = (
    SECTION_SPACING_ADAPTIVE,
    SECTION_SPACING_UNIFORM,
)


def opposed_join_order(quantity: int) -> List[int]:
    """Return a permutation that begins with widely separated radial branches."""

    if quantity < 1:
        raise ValueError("Join-order quantity must be at least one.")
    half = (quantity + 1) // 2
    order: List[int] = []
    for index in range(half):
        order.append(index)
        opposite_index = index + half
        if opposite_index < quantity:
            order.append(opposite_index)
    return order


@dataclass(frozen=True)
class ManifoldInputs:
    input_diameter_mm: float
    spacing_mm: float
    quantity: int
    output_diameter_mm: float
    mode: str = MODE_HEIGHT
    height_mm: float = 80.0
    inlet_angle_deg: float = 25.0
    vertical_inlets: bool = False
    curve_style: str = CURVE_EASE_BOTH
    inlet_tangency_weight: float = 0.25
    outlet_tangency_weight: float = 0.25
    bend_height_mm: float = 40.0
    section_count: int = 17
    section_spacing: str = SECTION_SPACING_ADAPTIVE
    integration_samples: int = 360


@dataclass(frozen=True)
class ResolvedInputs:
    input_diameter_mm: float
    spacing_mm: float
    quantity: int
    output_diameter_mm: float
    height_mm: float
    inlet_angle_deg: float
    contact_height_mm: float
    contact_center_radius_mm: float
    vertical_inlets: bool
    curve_style: str
    inlet_tangency_weight: float
    outlet_tangency_weight: float
    bend_height_mm: float
    section_count: int
    section_spacing: str
    integration_samples: int


@dataclass(frozen=True)
class Section:
    index: int
    t: float
    z_mm: float
    center_radius_mm: float
    branch_radius_mm: float
    ellipse_aspect: float
    local_angle_deg: float
    target_area_mm2: float
    calculated_area_mm2: float


def _contact_center_for_slope(
    inlet_radius: float, quantity: int, slope: float
) -> float:
    """Centre radius where adjacent radial ellipses are exactly tangent."""

    half_sector = math.pi / quantity
    sin_half = math.sin(half_sector)
    return (
        inlet_radius
        * math.sqrt(1.0 + slope * slope * sin_half * sin_half)
        / sin_half
    )


def _evaluate_cubic_bezier(
    control_points: Tuple[Tuple[float, float], ...], target_z: float
) -> Tuple[float, float]:
    """Evaluate radial position and dc/dz on a z-monotone cubic Bezier."""

    def at(t: float) -> Tuple[float, float, float, float]:
        one_minus = 1.0 - t
        basis = (
            one_minus**3,
            3.0 * one_minus * one_minus * t,
            3.0 * one_minus * t * t,
            t**3,
        )
        z = sum(basis[index] * control_points[index][0] for index in range(4))
        center = sum(
            basis[index] * control_points[index][1] for index in range(4)
        )
        dz_dt = 3.0 * (
            one_minus * one_minus * (control_points[1][0] - control_points[0][0])
            + 2.0
            * one_minus
            * t
            * (control_points[2][0] - control_points[1][0])
            + t * t * (control_points[3][0] - control_points[2][0])
        )
        dc_dt = 3.0 * (
            one_minus * one_minus * (control_points[1][1] - control_points[0][1])
            + 2.0
            * one_minus
            * t
            * (control_points[2][1] - control_points[1][1])
            + t * t * (control_points[3][1] - control_points[2][1])
        )
        return z, center, dz_dt, dc_dt

    low = 0.0
    high = 1.0
    for _ in range(52):
        middle = 0.5 * (low + high)
        if at(middle)[0] < target_z:
            low = middle
        else:
            high = middle
    _, center, dz_dt, dc_dt = at(0.5 * (low + high))
    return center, dc_dt / dz_dt if dz_dt > 1e-12 else 0.0


def _advanced_centerline(
    z_mm: float,
    height_mm: float,
    spacing_mm: float,
    curve_style: str,
    inlet_weight: float,
    outlet_weight: float,
    bend_height_mm: float,
) -> Tuple[float, float]:
    """Return centre radius and dc/dz for an advanced centreline style."""

    z_mm = max(0.0, min(height_mm, z_mm))
    if curve_style == CURVE_STRAIGHT:
        return spacing_mm * z_mm / height_mm, spacing_mm / height_mm

    bend_z = bend_height_mm
    bend_center = 0.5 * spacing_mm
    midpoint = (bend_z, bend_center)

    if curve_style == CURVE_EASE_BOTH:
        bottom_first = (outlet_weight * bend_z, 0.0)
    else:
        bottom_first = (bend_z / 3.0, bend_center / 3.0)

    if curve_style in (CURVE_EASE_BOTH, CURVE_EASE_INLET):
        top_third = (
            height_mm - inlet_weight * (height_mm - bend_z),
            spacing_mm,
        )
    else:
        top_third = (
            bend_z + 2.0 * (height_mm - bend_z) / 3.0,
            bend_center + 2.0 * (spacing_mm - bend_center) / 3.0,
        )

    # With equal shared handles, both cubic pieces are C1 at the bend. The
    # candidate below also makes their second derivatives equal because
    # top_third = bottom_first + 4 * shared_vector. Use it whenever the
    # resulting control polygon remains monotone. Extreme user settings fall
    # back to the previous safe C1 construction.
    candidate_dz = 0.25 * (top_third[0] - bottom_first[0])
    candidate_dc = 0.25 * (top_third[1] - bottom_first[1])
    is_monotone = (
        bottom_first[0] <= bend_z - candidate_dz <= bend_z
        and bend_z <= bend_z + candidate_dz <= top_third[0]
        and bottom_first[1] <= bend_center - candidate_dc <= bend_center
        and bend_center <= bend_center + candidate_dc <= top_third[1]
    )
    if is_monotone:
        shared_dz = candidate_dz
        shared_dc = candidate_dc
    else:
        shared_dz = 0.20 * min(bend_z, height_mm - bend_z)
        shared_dc = shared_dz * spacing_mm / height_mm

    bottom = (
        (0.0, 0.0),
        bottom_first,
        (bend_z - shared_dz, bend_center - shared_dc),
        midpoint,
    )
    top = (
        midpoint,
        (bend_z + shared_dz, bend_center + shared_dc),
        top_third,
        (height_mm, spacing_mm),
    )
    return _evaluate_cubic_bezier(bottom if z_mm <= bend_z else top, z_mm)


def _find_advanced_contact(
    height_mm: float,
    spacing_mm: float,
    inlet_radius: float,
    quantity: int,
    curve_style: str,
    inlet_weight: float,
    outlet_weight: float,
    bend_height_mm: float,
) -> Tuple[float, float]:
    """Find the first adjacent-tube contact while travelling down from top."""

    def clearance(z_mm: float) -> Tuple[float, float, float]:
        center, slope = _advanced_centerline(
            z_mm,
            height_mm,
            spacing_mm,
            curve_style,
            inlet_weight,
            outlet_weight,
            bend_height_mm,
        )
        tangent_center = _contact_center_for_slope(
            inlet_radius, quantity, slope
        )
        return center - tangent_center, center, slope

    upper_z = height_mm
    upper_clearance, _, _ = clearance(upper_z)
    if upper_clearance <= 0.0:
        raise ValueError(
            "The vertical inlet tubes touch or overlap at the starting plane. "
            "Increase input spacing or reduce input diameter."
        )

    lower_z = 0.0
    for index in range(1, 1001):
        candidate_z = height_mm * (1.0 - index / 1000.0)
        candidate_clearance, _, _ = clearance(candidate_z)
        if candidate_clearance <= 0.0:
            lower_z = candidate_z
            break
        upper_z = candidate_z
    else:
        raise RuntimeError("Could not locate the advanced-curve inlet contact plane.")

    for _ in range(64):
        middle_z = 0.5 * (lower_z + upper_z)
        middle_clearance, _, _ = clearance(middle_z)
        if middle_clearance <= 0.0:
            lower_z = middle_z
        else:
            upper_z = middle_z
    contact_z = 0.5 * (lower_z + upper_z)
    _, contact_center, _ = clearance(contact_z)
    return contact_z, contact_center


def resolve_inputs(values: ManifoldInputs) -> ResolvedInputs:
    """Validate inputs and resolve the height/angle alternative.

    The standard mode uses a straight constant-diameter inlet followed by a
    quadratic merge. Advanced mode provides multiple piecewise-Bezier and
    straight centreline styles.
    """

    errors: List[str] = []
    if values.input_diameter_mm <= 0:
        errors.append("Input tube diameter must be greater than zero.")
    if values.output_diameter_mm <= 0:
        errors.append("Output tube diameter must be greater than zero.")
    if values.spacing_mm <= 0:
        errors.append("Input tube spacing must be greater than zero.")
    if values.quantity < 2 or values.quantity > 24:
        errors.append("Input tube quantity must be between 2 and 24.")
    if values.section_count < 7 or values.section_count > 61:
        errors.append("Section count must be between 7 and 61.")
    if values.section_spacing not in SECTION_SPACING_STYLES:
        errors.append("Section spacing must be either adaptive or uniform.")
    if values.integration_samples < 90:
        errors.append("Integration samples must be at least 90.")
    if values.mode not in (MODE_HEIGHT, MODE_ANGLE):
        errors.append("Sizing mode must be either height or angle.")
    if values.vertical_inlets and values.curve_style not in CURVE_STYLES:
        errors.append("The selected centreline curve style is not supported.")
    if values.vertical_inlets and not 0.02 <= values.inlet_tangency_weight <= 0.48:
        errors.append("Inlet tangency weight must be between 0.02 and 0.48.")
    if values.vertical_inlets and not 0.02 <= values.outlet_tangency_weight <= 0.48:
        errors.append("Outlet tangency weight must be between 0.02 and 0.48.")
    if (
        values.vertical_inlets
        and values.curve_style != CURVE_STRAIGHT
        and not 0.05 * values.height_mm
        < values.bend_height_mm
        < 0.95 * values.height_mm
    ):
        errors.append(
            "Bend height must be between 5% and 95% of the total height."
        )

    height = values.height_mm
    angle = values.inlet_angle_deg
    inlet_radius = 0.5 * values.input_diameter_mm

    slope = 0.0
    contact_center = 0.0
    contact_height = 0.0
    if values.vertical_inlets:
        if height <= 0:
            errors.append("Height must be greater than zero.")
        if not errors:
            try:
                contact_height, contact_center = _find_advanced_contact(
                    height,
                    values.spacing_mm,
                    inlet_radius,
                    values.quantity,
                    values.curve_style,
                    values.inlet_tangency_weight,
                    values.outlet_tangency_weight,
                    values.bend_height_mm,
                )
                _, inlet_slope = _advanced_centerline(
                    height,
                    height,
                    values.spacing_mm,
                    values.curve_style,
                    values.inlet_tangency_weight,
                    values.outlet_tangency_weight,
                    values.bend_height_mm,
                )
                angle = math.degrees(math.atan(inlet_slope))
            except (ValueError, RuntimeError) as error:
                errors.append(str(error))
    elif values.mode == MODE_HEIGHT:
        if height <= 0:
            errors.append("Height must be greater than zero.")
        elif values.spacing_mm > 0 and inlet_radius > 0 and values.quantity >= 2:
            # Solve slope * height = spacing + contact_center(slope). This is
            # the exact piecewise-centreline height relation.
            low = 0.0
            high = 1.0

            def height_equation(candidate: float) -> float:
                return (
                    candidate * height
                    - values.spacing_mm
                    - _contact_center_for_slope(
                        inlet_radius, values.quantity, candidate
                    )
                )

            while height_equation(high) <= 0.0 and high < 1e6:
                high *= 2.0
            if height_equation(high) <= 0.0:
                errors.append(
                    "Height is too short to fit separate constant-diameter inlets "
                    "and a vertical outlet transition."
                )
            else:
                for _ in range(64):
                    middle = 0.5 * (low + high)
                    if height_equation(middle) < 0.0:
                        low = middle
                    else:
                        high = middle
                slope = 0.5 * (low + high)
                angle = math.degrees(math.atan(slope))
                contact_center = _contact_center_for_slope(
                    inlet_radius, values.quantity, slope
                )
    else:
        if not 0.0 < angle < 90.0:
            errors.append("Input tube angle must be greater than 0 and less than 90 degrees.")
        elif values.spacing_mm > 0 and inlet_radius > 0 and values.quantity >= 2:
            slope = math.tan(math.radians(angle))
            contact_center = _contact_center_for_slope(
                inlet_radius, values.quantity, slope
            )
            height = (values.spacing_mm + contact_center) / slope

    if not values.vertical_inlets and contact_center >= values.spacing_mm and not errors:
        errors.append(
            "The inlet tubes touch or overlap at the starting plane. Increase input "
            "spacing, reduce input diameter, or reduce the inlet angle."
        )

    if height <= 0:
        errors.append("The resolved height must be greater than zero.")
    if errors:
        raise ValueError("\n".join(errors))

    if not values.vertical_inlets:
        contact_height = 2.0 * contact_center / slope
    return ResolvedInputs(
        input_diameter_mm=values.input_diameter_mm,
        spacing_mm=values.spacing_mm,
        quantity=values.quantity,
        output_diameter_mm=values.output_diameter_mm,
        height_mm=height,
        inlet_angle_deg=angle,
        contact_height_mm=contact_height,
        contact_center_radius_mm=contact_center,
        vertical_inlets=values.vertical_inlets,
        curve_style=values.curve_style,
        inlet_tangency_weight=values.inlet_tangency_weight,
        outlet_tangency_weight=values.outlet_tangency_weight,
        bend_height_mm=values.bend_height_mm,
        section_count=values.section_count,
        section_spacing=values.section_spacing,
        integration_samples=values.integration_samples,
    )


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def evaluate_centerline(
    resolved: ResolvedInputs, z_mm: float
) -> Tuple[float, float]:
    """Return centre radius and radial slope at an arbitrary height."""

    if resolved.vertical_inlets:
        return _advanced_centerline(
            z_mm,
            resolved.height_mm,
            resolved.spacing_mm,
            resolved.curve_style,
            resolved.inlet_tangency_weight,
            resolved.outlet_tangency_weight,
            resolved.bend_height_mm,
        )
    straight_slope = math.tan(math.radians(resolved.inlet_angle_deg))
    if z_mm <= resolved.contact_height_mm:
        merge_fraction = z_mm / resolved.contact_height_mm
        return (
            resolved.contact_center_radius_mm
            * merge_fraction
            * merge_fraction,
            straight_slope * merge_fraction,
        )
    return (
        resolved.contact_center_radius_mm
        + straight_slope * (z_mm - resolved.contact_height_mm),
        straight_slope,
    )


def sample_centerline(
    resolved: ResolvedInputs,
    sections: List[Section],
    subdivisions_per_span: int,
) -> List[Tuple[float, float]]:
    """Densely sample a centreline while retaining every section centre.

    Returned tuples are ``(z_mm, center_radius_mm)``. A section centre occurs
    at every ``subdivisions_per_span`` index, guaranteeing that the Fusion
    guide intersects every loft profile even when extra fit points are added.
    """

    if subdivisions_per_span < 1 or subdivisions_per_span > 12:
        raise ValueError("Guide subdivisions per span must be between 1 and 12.")
    if len(sections) < 2:
        raise ValueError("At least two sections are required to sample a centreline.")

    samples: List[Tuple[float, float]] = []
    for span_index in range(len(sections) - 1):
        z_start = sections[span_index].z_mm
        z_end = sections[span_index + 1].z_mm
        for subdivision in range(subdivisions_per_span):
            fraction = subdivision / subdivisions_per_span
            z_mm = z_start + (z_end - z_start) * fraction
            center_radius, _ = evaluate_centerline(resolved, z_mm)
            samples.append((z_mm, center_radius))
    final_z = sections[-1].z_mm
    final_center, _ = evaluate_centerline(resolved, final_z)
    samples.append((final_z, final_center))
    return samples


def ellipse_union_area(
    center_radius: float,
    branch_radius: float,
    aspect: float,
    quantity: int,
    samples_per_sector: int = 360,
) -> float:
    """Area of the union of evenly spaced, radially oriented ellipses.

    Each ellipse has semi-major radius ``branch_radius * aspect`` in the
    radial direction and semi-minor radius ``branch_radius`` tangentially.
    The integral follows rays from the common origin and merges every radial
    interval hit by an ellipse. It works both before and after the ellipses
    touch, so the topology can change from separate ports to a clover.
    """

    if branch_radius <= 0.0 or aspect <= 0.0 or quantity < 1:
        return 0.0

    # At the outlet all equal circles coincide. The exact shortcut avoids
    # unnecessary work and removes numerical noise from the endpoint.
    if abs(center_radius) < 1e-12 and abs(aspect - 1.0) < 1e-12:
        return math.pi * branch_radius * branch_radius

    a = branch_radius * aspect
    b = branch_radius
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)
    sector = 2.0 * math.pi / quantity
    step = sector / samples_per_sector
    orientations = [sector * k for k in range(quantity)]

    def ray_measure(theta: float) -> float:
        intervals: List[Tuple[float, float]] = []
        for phi in orientations:
            delta = theta - phi
            cos_d = math.cos(delta)
            sin_d = math.sin(delta)
            qa = cos_d * cos_d * inv_a2 + sin_d * sin_d * inv_b2
            qb = -2.0 * center_radius * cos_d * inv_a2
            qc = center_radius * center_radius * inv_a2 - 1.0
            discriminant = qb * qb - 4.0 * qa * qc
            if discriminant < 0.0:
                continue
            root = math.sqrt(max(0.0, discriminant))
            lo = (-qb - root) / (2.0 * qa)
            hi = (-qb + root) / (2.0 * qa)
            if hi <= 0.0:
                continue
            intervals.append((max(0.0, lo), hi))

        if not intervals:
            return 0.0
        intervals.sort(key=lambda pair: pair[0])
        total = 0.0
        lo, hi = intervals[0]
        for next_lo, next_hi in intervals[1:]:
            if next_lo <= hi:
                hi = max(hi, next_hi)
            else:
                total += 0.5 * (hi * hi - lo * lo)
                lo, hi = next_lo, next_hi
        total += 0.5 * (hi * hi - lo * lo)
        return total

    integral = 0.0
    previous = ray_measure(0.0)
    for sample_index in range(1, samples_per_sector + 1):
        current = ray_measure(sample_index * step)
        integral += 0.5 * (previous + current) * step
        previous = current
    return integral * quantity


def ellipse_union_outer_radius(
    center_radius: float,
    branch_radius: float,
    aspect: float,
    quantity: int,
    theta: float,
) -> float:
    """Return the furthest positive ray intersection with any branch ellipse."""

    if branch_radius <= 0.0 or aspect <= 0.0 or quantity < 1:
        return 0.0
    a = branch_radius * aspect
    b = branch_radius
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)
    sector = 2.0 * math.pi / quantity
    outer = 0.0
    for branch_index in range(quantity):
        delta = theta - sector * branch_index
        cos_d = math.cos(delta)
        sin_d = math.sin(delta)
        qa = cos_d * cos_d * inv_a2 + sin_d * sin_d * inv_b2
        qb = -2.0 * center_radius * cos_d * inv_a2
        qc = center_radius * center_radius * inv_a2 - 1.0
        discriminant = qb * qb - 4.0 * qa * qc
        if discriminant < 0.0:
            continue
        hi = (-qb + math.sqrt(max(0.0, discriminant))) / (2.0 * qa)
        outer = max(outer, hi)
    return outer


def unified_clover_boundary(
    section: Section, quantity: int, point_count: int = 96
) -> List[Tuple[float, float]]:
    """Sample the filled outer boundary of a star-shaped merged section.

    This is area-equivalent to the ellipse union only when the origin lies
    inside the union. ``unified_core_end_index`` identifies that safe region.
    """

    if point_count < max(24, quantity * 4):
        raise ValueError(
            "Clover boundary point count must provide at least four points "
            "per lobe and at least 24 points overall."
        )
    points: List[Tuple[float, float]] = []
    for point_index in range(point_count):
        theta = 2.0 * math.pi * point_index / point_count
        radius = ellipse_union_outer_radius(
            section.center_radius_mm,
            section.branch_radius_mm,
            section.ellipse_aspect,
            quantity,
            theta,
        )
        points.append((radius * math.cos(theta), radius * math.sin(theta)))
    polygon_area = 0.5 * abs(
        sum(
            points[index][0] * points[(index + 1) % point_count][1]
            - points[(index + 1) % point_count][0] * points[index][1]
            for index in range(point_count)
        )
    )
    if polygon_area <= 1e-15:
        raise RuntimeError("Could not calculate a usable unified clover boundary.")
    # The polar samples are exact points on the ellipse-union envelope, but a
    # finite closed curve through them has a small chord-area deficit. Apply a
    # uniform sub-percent correction so the sampled closed boundary retains
    # the section's solved union area without changing its lobe proportions.
    area_scale = math.sqrt(section.calculated_area_mm2 / polygon_area)
    return [(x * area_scale, y * area_scale) for x, y in points]


def unified_core_end_index(sections: List[Section]) -> int:
    """Return the highest section safely representable as one filled clover."""

    if not sections:
        raise ValueError("At least one section is required for a unified core.")
    safe_indices = [
        section.index
        for section in sections
        if section.center_radius_mm
        <= 0.98 * section.branch_radius_mm * section.ellipse_aspect
    ]
    return max(safe_indices) if safe_indices else 0


def solve_branch_radius(
    target_area: float,
    center_radius: float,
    aspect: float,
    quantity: int,
    samples_per_sector: int,
    hint_radius: float,
) -> Tuple[float, float]:
    """Find the ellipse minor radius whose union has ``target_area``."""

    if target_area <= 0.0:
        raise ValueError("Target area must be greater than zero.")

    low = 0.0
    high = max(hint_radius, math.sqrt(target_area / math.pi), 1e-6)
    high_area = ellipse_union_area(
        center_radius, high, aspect, quantity, samples_per_sector
    )
    expansion_count = 0
    while high_area < target_area and expansion_count < 32:
        high *= 2.0
        high_area = ellipse_union_area(
            center_radius, high, aspect, quantity, samples_per_sector
        )
        expansion_count += 1
    if high_area < target_area:
        raise RuntimeError("Could not bracket the area-controlled branch radius.")

    area = high_area
    radius = high
    for _ in range(42):
        radius = 0.5 * (low + high)
        area = ellipse_union_area(
            center_radius, radius, aspect, quantity, samples_per_sector
        )
        relative_error = abs(area - target_area) / target_area
        if relative_error < 2e-6:
            break
        if area < target_area:
            low = radius
        else:
            high = radius
    return radius, area


def _uniform_section_parameters(resolved: ResolvedInputs) -> List[float]:
    """Return contact-anchored, otherwise uniform normalized heights."""

    contact_t = resolved.contact_height_mm / resolved.height_mm
    interval_count = resolved.section_count - 1
    # Reserve at least three intervals on either side of first contact for
    # stable loft interpolation and constant-diameter inlet verification.
    upper_intervals = round(interval_count * (1.0 - contact_t))
    upper_intervals = max(3, min(interval_count - 3, upper_intervals))
    lower_intervals = interval_count - upper_intervals
    parameters = [
        contact_t * index / lower_intervals
        for index in range(lower_intervals + 1)
    ]
    parameters.extend(
        contact_t + (1.0 - contact_t) * index / upper_intervals
        for index in range(1, upper_intervals + 1)
    )
    return parameters


def _adaptive_feature_vector(
    resolved: ResolvedInputs, t: float
) -> Tuple[float, ...]:
    """Describe the geometric changes that a loft section must follow.

    Feature-space arc length is used only to place profiles; exact section
    dimensions are still calculated later by the area solver. The baseline
    height component prevents unchanged regions from collapsing completely.
    """

    z_mm = resolved.height_mm * t
    center_radius, slope = evaluate_centerline(resolved, z_mm)
    center_scale = max(
        resolved.spacing_mm,
        resolved.contact_center_radius_mm,
        0.5 * resolved.input_diameter_mm,
        1.0,
    )
    contact_fraction = min(1.0, z_mm / resolved.contact_height_mm)
    area_progress = smoothstep(contact_fraction)
    # Overlap begins with a topology change at contact. Its square-root
    # coordinate deliberately adds resolution immediately below that plane,
    # while area_progress adds resolution near the middle of the area ease.
    clover_onset = math.sqrt(max(0.0, 1.0 - contact_fraction))
    angle_normalized = math.atan(slope) / (0.5 * math.pi)
    log_aspect = 0.5 * math.log1p(slope * slope)
    return (
        0.22 * t,
        center_radius / center_scale,
        1.25 * angle_normalized,
        0.50 * log_aspect,
        0.85 * area_progress,
        0.35 * clover_onset,
    )


def _feature_segment_samples(
    resolved: ResolvedInputs, start: float, end: float
) -> Tuple[List[float], List[float]]:
    """Sample cumulative adaptive feature distance over one fixed-anchor span."""

    sample_count = 96
    parameters = [
        start + (end - start) * index / sample_count
        for index in range(sample_count + 1)
    ]
    cumulative = [0.0]
    previous = _adaptive_feature_vector(resolved, parameters[0])
    for parameter in parameters[1:]:
        current = _adaptive_feature_vector(resolved, parameter)
        distance = math.sqrt(
            sum((right - left) ** 2 for left, right in zip(previous, current))
        )
        cumulative.append(cumulative[-1] + distance)
        previous = current
    return parameters, cumulative


def _adaptive_section_parameters(resolved: ResolvedInputs) -> List[float]:
    """Place a fixed number of sections according to geometric change."""

    contact_t = resolved.contact_height_mm / resolved.height_mm
    anchors = [0.0, contact_t, 1.0]
    if resolved.vertical_inlets and resolved.curve_style != CURVE_STRAIGHT:
        anchors.append(resolved.bend_height_mm / resolved.height_mm)
    anchors.sort()
    unique_anchors: List[float] = []
    for anchor in anchors:
        anchor = max(0.0, min(1.0, anchor))
        if not unique_anchors or abs(anchor - unique_anchors[-1]) > 1e-10:
            unique_anchors.append(anchor)

    segment_data = [
        _feature_segment_samples(resolved, start, end)
        for start, end in zip(unique_anchors, unique_anchors[1:])
    ]
    feature_lengths = [cumulative[-1] for _, cumulative in segment_data]
    interval_allocations = [1] * len(segment_data)

    lower_segments = [
        index
        for index in range(len(segment_data))
        if unique_anchors[index + 1] <= contact_t + 1e-10
    ]
    upper_segments = [
        index
        for index in range(len(segment_data))
        if unique_anchors[index] >= contact_t - 1e-10
    ]
    # Preserve the previous stability guarantee of at least three intervals
    # above and below first contact, even when bend height adds another anchor.
    for group in (lower_segments, upper_segments):
        while sum(interval_allocations[index] for index in group) < 3:
            chosen = max(
                group,
                key=lambda index: feature_lengths[index]
                / (interval_allocations[index] + 1),
            )
            interval_allocations[chosen] += 1

    total_intervals = resolved.section_count - 1
    while sum(interval_allocations) < total_intervals:
        chosen = max(
            range(len(segment_data)),
            key=lambda index: feature_lengths[index]
            / (interval_allocations[index] + 1),
        )
        interval_allocations[chosen] += 1

    parameters = [unique_anchors[0]]
    for segment_index, ((samples, cumulative), interval_count) in enumerate(
        zip(segment_data, interval_allocations)
    ):
        total_distance = cumulative[-1]
        for local_index in range(1, interval_count + 1):
            if local_index == interval_count:
                parameter = unique_anchors[segment_index + 1]
            else:
                target = total_distance * local_index / interval_count
                upper_index = bisect.bisect_left(cumulative, target)
                lower_index = max(0, upper_index - 1)
                distance_span = cumulative[upper_index] - cumulative[lower_index]
                fraction = (
                    0.0
                    if distance_span <= 1e-15
                    else (target - cumulative[lower_index]) / distance_span
                )
                parameter = samples[lower_index] + fraction * (
                    samples[upper_index] - samples[lower_index]
                )
            parameters.append(parameter)
    return parameters


def generate_sections(values: ManifoldInputs) -> Tuple[ResolvedInputs, List[Section]]:
    resolved = resolve_inputs(values)
    inlet_radius = 0.5 * resolved.input_diameter_mm
    outlet_radius = 0.5 * resolved.output_diameter_mm
    _, contact_slope = evaluate_centerline(resolved, resolved.contact_height_mm)
    contact_aspect = math.sqrt(1.0 + contact_slope * contact_slope)
    contact_area = (
        resolved.quantity * math.pi * inlet_radius * inlet_radius * contact_aspect
    )
    bottom_area = math.pi * outlet_radius * outlet_radius
    sections: List[Section] = []
    previous_radius = outlet_radius

    if resolved.section_spacing == SECTION_SPACING_ADAPTIVE:
        t_values = _adaptive_section_parameters(resolved)
    else:
        t_values = _uniform_section_parameters(resolved)

    for index, t in enumerate(t_values):
        z_mm = resolved.height_mm * t
        center_radius, slope = evaluate_centerline(resolved, z_mm)
        local_angle = math.degrees(math.atan(slope))
        # Every style terminates on the exact common circular outlet. Styles
        # without outlet easing may have an angled centreline there, but the
        # final loft section itself is deliberately held circular.
        aspect = 1.0 if index == 0 else math.sqrt(1.0 + slope * slope)
        if z_mm >= resolved.contact_height_mm:
            # Before contact, the true cross-section normal to each tube stays
            # circular and constant. In spline mode the horizontal ellipse
            # area varies only because the local centreline angle varies.
            target = (
                resolved.quantity
                * math.pi
                * inlet_radius
                * inlet_radius
                * aspect
            )
        else:
            merge_fraction = z_mm / resolved.contact_height_mm
            target = bottom_area + (contact_area - bottom_area) * smoothstep(
                merge_fraction
            )

        if index == 0:
            radius = outlet_radius
            calculated = bottom_area
        elif z_mm >= resolved.contact_height_mm - 1e-9:
            radius = inlet_radius
            calculated = target
        else:
            radius, calculated = solve_branch_radius(
                target,
                center_radius,
                aspect,
                resolved.quantity,
                resolved.integration_samples,
                previous_radius,
            )
        previous_radius = radius
        sections.append(
            Section(
                index=index,
                t=t,
                z_mm=z_mm,
                center_radius_mm=center_radius,
                branch_radius_mm=radius,
                ellipse_aspect=aspect,
                local_angle_deg=local_angle,
                target_area_mm2=target,
                calculated_area_mm2=calculated,
            )
        )

    return resolved, sections


def section_area_error(section: Section) -> float:
    if section.target_area_mm2 == 0.0:
        return 0.0
    return (
        section.calculated_area_mm2 - section.target_area_mm2
    ) / section.target_area_mm2
