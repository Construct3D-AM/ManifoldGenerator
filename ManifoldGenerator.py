"""Autodesk Fusion add-in entry point for the area-controlled manifold."""

from __future__ import annotations

import json
import importlib
import math
import os
import sys
import traceback

import adsk.core
import adsk.fusion


ADDIN_DIR = os.path.dirname(os.path.abspath(__file__))
if ADDIN_DIR not in sys.path:
    sys.path.insert(0, ADDIN_DIR)

# Fusion keeps imported Python modules alive after an add-in is stopped. Reload
# our numerical module explicitly so an updated entry point can never be paired
# with stale dataclasses or solver functions from an earlier add-in version.
import manifold_geometry as _manifold_geometry  # noqa: E402

_manifold_geometry = importlib.reload(_manifold_geometry)
MODE_ANGLE = _manifold_geometry.MODE_ANGLE
MODE_HEIGHT = _manifold_geometry.MODE_HEIGHT
CURVE_EASE_BOTH = _manifold_geometry.CURVE_EASE_BOTH
CURVE_EASE_INLET = _manifold_geometry.CURVE_EASE_INLET
CURVE_DIRECT = _manifold_geometry.CURVE_DIRECT
CURVE_STRAIGHT = _manifold_geometry.CURVE_STRAIGHT
SECTION_SPACING_ADAPTIVE = _manifold_geometry.SECTION_SPACING_ADAPTIVE
SECTION_SPACING_UNIFORM = _manifold_geometry.SECTION_SPACING_UNIFORM
ManifoldInputs = _manifold_geometry.ManifoldInputs
generate_sections = _manifold_geometry.generate_sections
resolve_inputs = _manifold_geometry.resolve_inputs
sample_centerline = _manifold_geometry.sample_centerline
sample_branch_centerline = _manifold_geometry.sample_branch_centerline
section_area_error = _manifold_geometry.section_area_error
opposed_join_order = _manifold_geometry.opposed_join_order
unified_clover_boundary = _manifold_geometry.unified_clover_boundary
unified_core_end_index = _manifold_geometry.unified_core_end_index


COMMAND_ID = "SeraphAreaControlledManifoldCommand"
COMMAND_NAME = "Area-Controlled Manifold"
COMMAND_DESCRIPTION = (
    "Create radial or straight-line circular inlets that merge smoothly into one circular outlet "
    "while controlling horizontal cross-sectional area."
)
PANEL_ID = "SolidScriptsAddinsPanel"

handlers = []


def _ui():
    return adsk.core.Application.get().userInterface


def _input(command_inputs, input_id):
    found = command_inputs.itemById(input_id)
    if found:
        return found
    # Group children are not returned by itemById in every Fusion release.
    for index in range(command_inputs.count):
        group = adsk.core.GroupCommandInput.cast(command_inputs.item(index))
        if group:
            found = _input(group.children, input_id)
            if found:
                return found
    return None


def _selected_mode(command_inputs):
    if _input(command_inputs, "verticalInlets").value:
        return MODE_HEIGHT
    selected = _input(command_inputs, "sizingMode").selectedItem
    return MODE_HEIGHT if selected and selected.index == 0 else MODE_ANGLE


def _selected_curve_style(command_inputs):
    selected = _input(command_inputs, "curveStyle").selectedItem
    styles = (
        CURVE_EASE_BOTH,
        CURVE_EASE_INLET,
        CURVE_DIRECT,
        CURVE_STRAIGHT,
    )
    return styles[selected.index if selected else 0]


def _selected_section_spacing(command_inputs):
    selected = _input(command_inputs, "sectionSpacing").selectedItem
    return (
        SECTION_SPACING_ADAPTIVE
        if not selected or selected.index == 0
        else SECTION_SPACING_UNIFORM
    )


def _read_values(command_inputs):
    # Fusion stores lengths internally in centimetres and angles in radians.
    return ManifoldInputs(
        input_diameter_mm=_input(command_inputs, "inputDiameter").value * 10.0,
        spacing_mm=_input(command_inputs, "inputSpacing").value * 10.0,
        quantity=_input(command_inputs, "inputQuantity").value,
        output_diameter_mm=_input(command_inputs, "outputDiameter").value * 10.0,
        mode=_selected_mode(command_inputs),
        height_mm=_input(command_inputs, "height").value * 10.0,
        inlet_angle_deg=math.degrees(_input(command_inputs, "inletAngle").value),
        vertical_inlets=_input(command_inputs, "verticalInlets").value,
        curve_style=_selected_curve_style(command_inputs),
        inlet_tangency_weight=_input(command_inputs, "inletTangencyWeight").value,
        outlet_tangency_weight=_input(command_inputs, "outletTangencyWeight").value,
        bend_height_mm=_input(command_inputs, "bendHeight").value * 10.0,
        section_count=_input(command_inputs, "sectionCount").value,
        section_spacing=_selected_section_spacing(command_inputs),
        integration_samples=360,
        linear_layout=_input(command_inputs, "linearLayout").value,
    )


def _update_mode_visibility(command_inputs):
    uses_spline = _input(command_inputs, "verticalInlets").value
    curve_style = _selected_curve_style(command_inputs)
    is_height = _selected_mode(command_inputs) == MODE_HEIGHT
    _input(command_inputs, "sizingMode").isVisible = not uses_spline
    _input(command_inputs, "height").isVisible = uses_spline or is_height
    _input(command_inputs, "inletAngle").isVisible = not uses_spline and not is_height
    _input(command_inputs, "curveStyle").isVisible = uses_spline
    _input(command_inputs, "bendHeight").isVisible = (
        uses_spline and curve_style != CURVE_STRAIGHT
    )
    _input(command_inputs, "inletTangencyWeight").isVisible = (
        uses_spline and curve_style in (CURVE_EASE_BOTH, CURVE_EASE_INLET)
    )
    _input(command_inputs, "outletTangencyWeight").isVisible = (
        uses_spline and curve_style == CURVE_EASE_BOTH
    )


def _new_offset_plane(component, z_cm):
    plane_input = component.constructionPlanes.createInput()
    plane_input.setByOffset(
        component.xYConstructionPlane,
        adsk.core.ValueInput.createByReal(z_cm),
    )
    plane = component.constructionPlanes.add(plane_input)
    if not plane:
        raise RuntimeError("Fusion could not create a section construction plane.")
    return plane


def _add_section_profile(
    component, plane, section, branch_index, quantity, profile_scale=1.0
):
    sketch = component.sketches.add(plane)
    sketch.name = "Section {:02d} - Branch {:02d}".format(
        section.index + 1, branch_index + 1
    )
    if section.branch_profiles:
        branch = section.branch_profiles[branch_index]
        phi = branch.orientation_rad
        cx = branch.center_x_mm / 10.0
        cy = branch.center_y_mm / 10.0
        minor_cm = branch.radius_mm * profile_scale / 10.0
        major_cm = minor_cm * branch.ellipse_aspect
    else:
        phi = 2.0 * math.pi * branch_index / quantity
        center_cm = section.center_radius_mm / 10.0
        minor_cm = section.branch_radius_mm * profile_scale / 10.0
        major_cm = minor_cm * section.ellipse_aspect
        cx = center_cm * math.cos(phi)
        cy = center_cm * math.sin(phi)
    center = adsk.core.Point3D.create(cx, cy, 0.0)

    if abs(major_cm - minor_cm) < 1e-8:
        sketch.sketchCurves.sketchCircles.addByCenterRadius(center, minor_cm)
    else:
        major_point = adsk.core.Point3D.create(
            cx + major_cm * math.cos(phi),
            cy + major_cm * math.sin(phi),
            0.0,
        )
        minor_point = adsk.core.Point3D.create(
            cx - minor_cm * math.sin(phi),
            cy + minor_cm * math.cos(phi),
            0.0,
        )
        sketch.sketchCurves.sketchEllipses.add(center, major_point, minor_point)

    if sketch.profiles.count != 1:
        raise RuntimeError(
            "Expected one closed profile in {}, found {}.".format(
                sketch.name, sketch.profiles.count
            )
        )
    return sketch, sketch.profiles.item(0)


def _add_unified_clover_profile(
    component, plane, section, quantity, profile_scale=1.0
):
    """Create one periodic outer-boundary profile for the merged lower core."""

    sketch = component.sketches.add(plane)
    sketch.name = "Unified Clover Section {:02d}".format(section.index + 1)
    point_count = max(96, quantity * 8)
    if section.index == 0:
        center = adsk.core.Point3D.create(0.0, 0.0, 0.0)
        sketch.sketchCurves.sketchCircles.addByCenterRadius(
            center, section.branch_radius_mm * profile_scale / 10.0
        )
    else:
        points = adsk.core.ObjectCollection.create()
        for x_mm, y_mm in unified_clover_boundary(
            section, quantity, point_count
        ):
            points.add(
                adsk.core.Point3D.create(
                    x_mm * profile_scale / 10.0,
                    y_mm * profile_scale / 10.0,
                    0.0,
                )
            )
        spline = sketch.sketchCurves.sketchFittedSplines.add(points)
        if not spline:
            raise RuntimeError(
                "Fusion could not create unified clover spline section {}."
                .format(section.index + 1)
            )
        spline.isClosed = True
        if not spline.isClosed:
            raise RuntimeError(
                "Fusion did not accept unified clover section {} as a periodic "
                "closed spline.".format(section.index + 1)
            )

    if sketch.profiles.count != 1:
        raise RuntimeError(
            "Expected one filled unified clover profile in {}, found {}."
            .format(sketch.name, sketch.profiles.count)
        )
    return sketch, sketch.profiles.item(0), point_count


def _add_branch_centerline(
    component,
    resolved,
    sections,
    branch_index,
    quantity,
    guide_subdivisions,
):
    """Create a dense 3D guide through every branch profile centre."""

    sketch = component.sketches.add(component.xYConstructionPlane)
    sketch.name = "Loft Centreline - Branch {:02d}".format(branch_index + 1)
    dense_samples = sample_branch_centerline(
        resolved, sections, branch_index, guide_subdivisions
    )
    points = adsk.core.ObjectCollection.create()
    for z_mm, x_mm, y_mm in dense_samples:
        points.add(
            adsk.core.Point3D.create(
                x_mm / 10.0,
                y_mm / 10.0,
                z_mm / 10.0,
            )
        )

    if resolved.vertical_inlets and resolved.curve_style == CURVE_STRAIGHT:
        guide_curve = sketch.sketchCurves.sketchLines.addByTwoPoints(
            points.item(0), points.item(points.count - 1)
        )
    else:
        guide_curve = sketch.sketchCurves.sketchFittedSplines.add(points)
    if not guide_curve:
        raise RuntimeError(
            "Fusion could not create the dense centreline for branch {}."
            .format(branch_index + 1)
        )
    if guide_curve.is2D:
        raise RuntimeError(
            "Fusion unexpectedly flattened branch {}'s centreline into a 2D curve."
            .format(branch_index + 1)
        )
    guide_curve.isConstruction = True
    return sketch, guide_curve, len(dense_samples)


def _add_unified_core_loft(component, profiles):
    """Loft the simply connected, area-controlled lower clover as one body."""

    if len(profiles) < 2:
        raise RuntimeError(
            "The unified clover region needs at least two profiles. Increase "
            "the manifold height or loft section count."
        )
    loft_features = component.features.loftFeatures
    loft_input = loft_features.createInput(
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )
    for profile in profiles:
        loft_input.loftSections.add(profile)
    loft_input.isSolid = True
    loft_input.isClosed = False
    loft_input.isTangentEdgesMerged = True
    try:
        loft = loft_features.add(loft_input)
    except Exception as error:
        raise RuntimeError(
            "Fusion could not loft the unified lower clover body."
        ) from error
    if not loft or loft.bodies.count != 1:
        raise RuntimeError(
            "The unified lower clover loft did not produce exactly one body."
        )
    loft.name = "Unified Area-Controlled Clover Core"
    body = loft.bodies.item(0)
    body.name = "Unified Clover Core"
    return body


def _add_joined_inlet_loft(
    component, profiles, centerline, branch_index, join_step, total_steps
):
    """Join one inlet, with a staged NewBody/Combine fallback for ASM errors."""

    loft_features = component.features.loftFeatures

    def create_input(operation):
        result = loft_features.createInput(operation)
        for profile in profiles:
            result.loftSections.add(profile)
        if not result.centerLineOrRails.addCenterLine(centerline):
            raise RuntimeError(
                "Fusion rejected the spline centreline for inlet branch {}."
                .format(branch_index + 1)
            )
        result.isSolid = True
        result.isClosed = False
        result.isTangentEdgesMerged = True
        return result

    loft_input = create_input(adsk.fusion.FeatureOperations.JoinFeatureOperation)
    try:
        loft = loft_features.add(loft_input)
    except Exception as direct_join_error:
        try:
            if component.bRepBodies.count != 1:
                raise RuntimeError(
                    "The staged fallback expected one existing manifold body."
                )
            target_body = component.bRepBodies.item(0)
            separate_loft = loft_features.add(
                create_input(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            )
            if not separate_loft or separate_loft.bodies.count != 1:
                raise RuntimeError(
                    "The staged fallback could not create one inlet tool body."
                )
            separate_loft.name = "Fallback Inlet Body {:02d}".format(
                branch_index + 1
            )
            tool_bodies = adsk.core.ObjectCollection.create()
            tool_bodies.add(separate_loft.bodies.item(0))
            combine_features = component.features.combineFeatures
            combine_input = combine_features.createInput(target_body, tool_bodies)
            combine_input.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
            combine_input.isKeepToolBodies = False
            combine = combine_features.add(combine_input)
            if not combine or component.bRepBodies.count != 1:
                raise RuntimeError(
                    "The staged fallback did not leave one manifold body."
                )
            combine.name = "Fallback Joined Inlet {:02d}".format(branch_index + 1)
            return component.bRepBodies.item(0)
        except Exception as fallback_error:
            raise RuntimeError(
                "Unified-core inlet join {} of {} failed on branch {}. Both "
                "the direct Join loft and staged NewBody/Combine fallback were "
                "rejected by Fusion. Fallback detail: {}"
                .format(
                    join_step,
                    total_steps,
                    branch_index + 1,
                    fallback_error,
                )
            ) from direct_join_error
    if not loft or loft.bodies.count != 1:
        raise RuntimeError(
            "Unified-core inlet join {} of {} did not leave exactly one result "
            "body.".format(join_step, total_steps)
        )
    loft.name = "Joined Inlet Loft {:02d}".format(branch_index + 1)
    return loft.bodies.item(0)


def _build_component(design, resolved, sections, guide_subdivisions):
    root = design.rootComponent
    occurrence = None
    try:
        occurrence = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        component = occurrence.component
        component.name = "Area Controlled {} Manifold - {} inlets".format(
            "Straight-Line" if resolved.linear_layout else "Radial",
            resolved.quantity,
        )

        planes = []
        for section in sections:
            if section.index == 0:
                planes.append(component.xYConstructionPlane)
            else:
                plane = _new_offset_plane(component, section.z_mm / 10.0)
                plane.name = "Manifold Section Plane {:02d}".format(section.index + 1)
                planes.append(plane)

        core_end_index = unified_core_end_index(sections)
        if core_end_index < 2:
            raise RuntimeError(
                "The area-controlled section set does not provide enough "
                "simply connected profiles for a unified clover core. Increase "
                "Loft sections."
            )
        branch_start_index = core_end_index - 2
        core_sections = sections[: core_end_index + 1]
        branch_sections = sections[branch_start_index:]

        all_sketches = []
        clover_profiles = []
        clover_point_count = 0
        for section in core_sections:
            core_profile_scale = 1.0
            if section.index >= branch_start_index:
                overlap_progress = (
                    (section.index - branch_start_index)
                    / (core_end_index - branch_start_index)
                )
                # Retreat only at the top of the overlap. The inlet profiles
                # grow from safely internal to exact, forcing a transverse
                # Boolean intersection instead of nearly coincident faces.
                core_profile_scale = 1.0 - 0.015 * overlap_progress**3
            sketch, profile, clover_point_count = _add_unified_clover_profile(
                component,
                planes[section.index],
                section,
                resolved.quantity,
                core_profile_scale,
            )
            all_sketches.append(sketch)
            clover_profiles.append(profile)

        branch_profiles = [[] for _ in range(resolved.quantity)]
        for section in branch_sections:
            branch_profile_scale = 1.0
            if section.index <= core_end_index:
                overlap_progress = (
                    (section.index - branch_start_index)
                    / (core_end_index - branch_start_index)
                )
                eased_progress = overlap_progress * overlap_progress * (
                    3.0 - 2.0 * overlap_progress
                )
                branch_profile_scale = 0.96 + 0.04 * eased_progress
            if abs(section.z_mm - resolved.contact_height_mm) < 1e-8:
                # Exact tangency at a shared loft section is a degenerate
                # Boolean condition in Fusion. The area transition is allowed
                # to begin at contact, so add a half-percent construction
                # overlap there; every section above contact stays exact.
                branch_profile_scale *= 1.005
            for branch_index in range(resolved.quantity):
                sketch, profile = _add_section_profile(
                    component,
                    planes[section.index],
                    section,
                    branch_index,
                    resolved.quantity,
                    branch_profile_scale,
                )
                all_sketches.append(sketch)
                branch_profiles[branch_index].append(profile)

        branch_centerlines = []
        guide_point_count = 0
        for branch_index in range(resolved.quantity):
            guide_sketch, guide_curve, guide_point_count = _add_branch_centerline(
                component,
                resolved,
                branch_sections,
                branch_index,
                resolved.quantity,
                guide_subdivisions,
            )
            all_sketches.append(guide_sketch)
            branch_centerlines.append(guide_curve)

        joined_body = _add_unified_core_loft(component, clover_profiles)
        branch_order = (
            list(range(resolved.quantity))
            if resolved.linear_layout
            else opposed_join_order(resolved.quantity)
        )
        for join_step, branch_index in enumerate(branch_order, start=1):
            joined_body = _add_joined_inlet_loft(
                component,
                branch_profiles[branch_index],
                branch_centerlines[branch_index],
                branch_index,
                join_step,
                resolved.quantity,
            )
            if component.bRepBodies.count != 1:
                raise RuntimeError(
                    "Joined inlet loft {} left {} component bodies instead of "
                    "one.".format(join_step, component.bRepBodies.count)
                )

        if component.bRepBodies.count != 1:
            raise RuntimeError(
                "Expected one joined body, but Fusion reports {} bodies.".format(
                    component.bRepBodies.count
                )
            )
        joined_body.name = "Area Controlled Manifold"

        for sketch in all_sketches:
            sketch.isLightBulbOn = False
        for plane in planes[1:]:
            plane.isLightBulbOn = False

        settings = {
            "inputDiameterMm": resolved.input_diameter_mm,
            "inputSpacingMm": resolved.spacing_mm,
            "inputQuantity": resolved.quantity,
            "layout": "straight-line" if resolved.linear_layout else "radial",
            "inletPositionsMm": list(resolved.inlet_positions_mm),
            "inputAngleDeg": resolved.inlet_angle_deg,
            "heightMm": resolved.height_mm,
            "contactHeightMm": resolved.contact_height_mm,
            "contactCenterRadiusMm": resolved.contact_center_radius_mm,
            "verticalInlets": resolved.vertical_inlets,
            "curveStyle": resolved.curve_style,
            "inletTangencyWeight": resolved.inlet_tangency_weight,
            "outletTangencyWeight": resolved.outlet_tangency_weight,
            "bendHeightMm": resolved.bend_height_mm,
            "outputDiameterMm": resolved.output_diameter_mm,
            "sectionCount": resolved.section_count,
            "sectionSpacing": resolved.section_spacing,
            "guideSubdivisionsPerSpan": guide_subdivisions,
            "guidePointCountPerBranch": guide_point_count,
            "loftGuide": "dense 3D fitted spline through analytic centreline samples",
            "unifiedCoreEndSection": core_end_index + 1,
            "unifiedCoreEndHeightMm": sections[core_end_index].z_mm,
            "branchOverlapStartSection": branch_start_index + 1,
            "branchOverlapStartHeightMm": sections[branch_start_index].z_mm,
            "cloverBoundaryPointCount": clover_point_count,
            "joinStrategy": (
                "direct inlet Join lofts with automatic staged NewBody/Combine fallback"
            ),
            "overlapHandoff": (
                "inlets scale 0.96-to-1.00 while core retreats 1.5% at handoff"
            ),
            "contactReliefScale": 1.005,
            "joinFeatureCount": resolved.quantity,
            "areaInterpretation": "horizontal cross-sectional union area",
        }
        component.attributes.add(
            "SeraphAreaControlledManifold", "settings", json.dumps(settings)
        )
        return component
    except Exception:
        if occurrence and occurrence.isValid:
            occurrence.deleteMe()
        raise


def _export_step_if_requested(design, component):
    dialog = _ui().createFileDialog()
    dialog.title = "Export manifold as STEP"
    dialog.filter = "STEP files (*.step;*.stp)"
    dialog.filterIndex = 0
    dialog.initialFilename = "area-controlled-manifold.step"
    if dialog.showSave() != adsk.core.DialogResults.DialogOK:
        return None
    path = dialog.filename
    if not path.lower().endswith((".step", ".stp")):
        path += ".step"
    options = design.exportManager.createSTEPExportOptions(path, component)
    if not design.exportManager.execute(options):
        raise RuntimeError("Fusion could not export the STEP file.")
    return path


class CommandExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            app = adsk.core.Application.get()
            design = adsk.fusion.Design.cast(app.activeProduct)
            if not design:
                raise RuntimeError("Open or create a Fusion Design before running this command.")

            command_inputs = args.firingEvent.sender.commandInputs
            values = _read_values(command_inputs)
            resolved, sections = generate_sections(values)
            guide_subdivisions = _input(
                command_inputs, "guideSubdivisions"
            ).value
            component = _build_component(
                design, resolved, sections, guide_subdivisions
            )

            export_path = None
            if _input(command_inputs, "exportStep").value:
                export_path = _export_step_if_requested(design, component)

            max_error = max(abs(section_area_error(s)) for s in sections) * 100.0
            core_end_index = unified_core_end_index(sections)
            branch_start_index = max(0, core_end_index - 2)
            branch_section_count = len(sections) - branch_start_index
            message = (
                "Created one joined manifold body.\n\n"
                "Inlet layout: {}\n"
                "Resolved height: {:.3f} mm\n"
                "Resolved inlet angle: {:.3f} deg from vertical\n"
                "Centreline mode: {}\n"
                "First contact height: {:.3f} mm above outlet\n"
                "Unified clover height: {:.3f} mm above outlet\n"
                "Generated sections: {}\n"
                "Section spacing: {}\n"
                "Guide points per branch: {}\n"
                "Joined inlet lofts: {}\n"
                "Maximum numerical section-area error: {:.4f}%"
            ).format(
                "straight line" if resolved.linear_layout else "radial circle",
                resolved.height_mm,
                resolved.inlet_angle_deg,
                (
                    "advanced {} (in {:.3f}, out {:.3f}, bend {:.3f} mm)".format(
                        resolved.curve_style,
                        resolved.inlet_tangency_weight,
                        resolved.outlet_tangency_weight,
                        resolved.bend_height_mm,
                    )
                    if resolved.vertical_inlets
                    else "straight inlet / quadratic merge"
                ),
                resolved.contact_height_mm,
                sections[core_end_index].z_mm,
                len(sections),
                resolved.section_spacing,
                (branch_section_count - 1) * guide_subdivisions + 1,
                resolved.quantity,
                max_error,
            )
            if export_path:
                message += "\n\nSTEP exported to:\n{}".format(export_path)
            _ui().messageBox(message, COMMAND_NAME)
        except Exception:
            _ui().messageBox(
                "Could not create the manifold:\n\n{}".format(traceback.format_exc()),
                COMMAND_NAME,
            )


class InputChangedHandler(adsk.core.InputChangedEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        if args.input and args.input.id in (
            "sizingMode",
            "verticalInlets",
            "curveStyle",
        ):
            _update_mode_visibility(args.inputs)


class ValidateInputsHandler(adsk.core.ValidateInputsEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            command_inputs = args.firingEvent.sender.commandInputs
            # Keep dialog validation immediate; the numerical area solve is
            # intentionally deferred until the user presses OK.
            resolve_inputs(_read_values(command_inputs))
            args.areInputsValid = True
        except Exception:
            args.areInputsValid = False


class CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        command = args.command
        command.isExecutedWhenPreEmpted = False
        inputs = command.commandInputs

        explanation = inputs.addTextBoxCommandInput(
            "areaExplanation",
            "",
            "Each isolated inlet stays at the exact input diameter. Use the standard straight "
            "inlet or an advanced centreline style; choose radial or straight-line layout. "
            "Area easing begins at first contact.",
            3,
            True,
        )
        explanation.isFullWidth = True

        inputs.addValueInput(
            "inputDiameter", "Input tube diameter", "mm", adsk.core.ValueInput.createByString("30 mm")
        )
        inputs.addValueInput(
            "inputSpacing",
            "Input spacing / line pitch",
            "mm",
            adsk.core.ValueInput.createByString("35 mm"),
        )
        inputs.addIntegerSpinnerCommandInput(
            "inputQuantity", "Input tube quantity", 2, 24, 1, 4
        )
        inputs.addBoolValueInput(
            "linearLayout",
            "Straight-line inlet layout",
            True,
            "",
            False,
        )
        inputs.addBoolValueInput(
            "verticalInlets",
            "Advanced centreline curve",
            True,
            "",
            False,
        )

        curve_style = inputs.addDropDownCommandInput(
            "curveStyle",
            "Curve style",
            adsk.core.DropDownStyles.TextListDropDownStyle,
        )
        curve_style.listItems.add("Ease both ends", True)
        curve_style.listItems.add("Ease inlet only", False)
        curve_style.listItems.add("Direct curve (no easing)", False)
        curve_style.listItems.add("Straight convergence", False)

        mode = inputs.addDropDownCommandInput(
            "sizingMode", "Size transition by", adsk.core.DropDownStyles.TextListDropDownStyle
        )
        mode.listItems.add("Height", True)
        mode.listItems.add("Input angle", False)
        inputs.addValueInput(
            "height", "Height", "mm", adsk.core.ValueInput.createByString("80 mm")
        )
        inputs.addAngleValueCommandInput(
            "inletAngle", "Input tube angle from vertical", adsk.core.ValueInput.createByString("25 deg")
        )
        inputs.addValueInput(
            "bendHeight",
            "Bend height above outlet",
            "mm",
            adsk.core.ValueInput.createByString("40 mm"),
        )
        inputs.addValueInput(
            "inletTangencyWeight",
            "Inlet tangency weight (0.02-0.48)",
            "",
            adsk.core.ValueInput.createByReal(0.25),
        )
        inputs.addValueInput(
            "outletTangencyWeight",
            "Outlet tangency weight (0.02-0.48)",
            "",
            adsk.core.ValueInput.createByReal(0.25),
        )
        inputs.addValueInput(
            "outputDiameter", "Output tube diameter", "mm", adsk.core.ValueInput.createByString("50 mm")
        )

        quality = inputs.addGroupCommandInput("qualityGroup", "Quality")
        quality.isExpanded = False
        quality.children.addIntegerSpinnerCommandInput(
            "sectionCount", "Loft sections", 7, 61, 2, 17
        )
        section_spacing = quality.children.addDropDownCommandInput(
            "sectionSpacing",
            "Loft section spacing",
            adsk.core.DropDownStyles.TextListDropDownStyle,
        )
        section_spacing.listItems.add("Adaptive", True)
        section_spacing.listItems.add("Uniform", False)
        quality.children.addIntegerSpinnerCommandInput(
            "guideSubdivisions",
            "Guide samples per loft span",
            1,
            12,
            1,
            4,
        )
        quality.children.addBoolValueInput(
            "exportStep", "Export STEP after creation", True, "", False
        )
        _update_mode_visibility(inputs)

        execute_handler = CommandExecuteHandler()
        command.execute.add(execute_handler)
        handlers.append(execute_handler)

        changed_handler = InputChangedHandler()
        command.inputChanged.add(changed_handler)
        handlers.append(changed_handler)

        validate_handler = ValidateInputsHandler()
        command.validateInputs.add(validate_handler)
        handlers.append(validate_handler)


def run(context):
    try:
        ui = _ui()
        existing = ui.commandDefinitions.itemById(COMMAND_ID)
        if existing:
            existing.deleteMe()
        command_definition = ui.commandDefinitions.addButtonDefinition(
            COMMAND_ID, COMMAND_NAME, COMMAND_DESCRIPTION
        )
        created_handler = CommandCreatedHandler()
        command_definition.commandCreated.add(created_handler)
        handlers.append(created_handler)

        panel = ui.allToolbarPanels.itemById(PANEL_ID)
        if not panel:
            raise RuntimeError("Fusion's SOLID > Utilities > Add-Ins panel was not found.")
        previous_control = panel.controls.itemById(COMMAND_ID)
        if previous_control:
            previous_control.deleteMe()
        control = panel.controls.addCommand(command_definition)
        control.isPromotedByDefault = True
        control.isPromoted = True
    except Exception:
        _ui().messageBox("Add-in start failed:\n\n{}".format(traceback.format_exc()))


def stop(context):
    try:
        ui = _ui()
        panel = ui.allToolbarPanels.itemById(PANEL_ID)
        if panel:
            control = panel.controls.itemById(COMMAND_ID)
            if control:
                control.deleteMe()
        command_definition = ui.commandDefinitions.itemById(COMMAND_ID)
        if command_definition:
            command_definition.deleteMe()
        handlers.clear()
    except Exception:
        _ui().messageBox("Add-in stop failed:\n\n{}".format(traceback.format_exc()))
