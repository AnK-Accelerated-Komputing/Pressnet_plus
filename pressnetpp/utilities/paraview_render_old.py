#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paraview_render_pressnet.py
================================================================================
STEP 2 of 2  --  run with ParaView's python:

    pvpython paraview_render_pressnet.py
    xvfb-run -a pvbatch paraview_render_pressnet.py     # headless GLX build
    pvbatch  paraview_render_pressnet.py                # headless OSMESA build

Reads the per-model / per-geometry VTU folders from `convert_pkls_to_vtu.py`
and renders, for each geometry of each model, GT / Prediction / Error panels
across time, then stitches them side-by-side (with legends beside, not
overlaid) into an mp4 + gif.

Design points:
  * 3D oblique view (CAMERA_AZIMUTH / CAMERA_ELEVATION), not a flat 2D look.
  * The die MOVES: it is warped by its real displacement and drawn as a solid
    colour (DIE_COLOR); the workpiece is coloured by the field.
  * Error panels show ONLY the workpiece (die thresholded out).
  * Legends are written as their OWN image files and placed BESIDE the panels
    (in-panel colour bars are turned off, so nothing overlays the mesh).
  * Only every SKIP-th timestep is rendered (default 10), not all 500.
  * Colour scales: GT & Pred share ONE range per geometry (time-invariant);
    the error legend has its OWN range per geometry (time-invariant).

Requires the VTU point arrays written by the converter, including 'is_die'.
================================================================================
"""

import os
import glob
import shutil

from paraview.simple import *
from vtkmodules.vtkIOImage import vtkPNGReader, vtkPNGWriter
from vtk import vtkImageAppend

LoadPalette('WhiteBackground')


# ==============================================================================
# 1. USER SETTINGS
# ==============================================================================
VTU_ROOT    = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/Paraview_render/output/vtu_root"
OUTPUT_ROOT = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/Paraview_render/output/figures"

# Optional model-name filter ([] = every model found). Geometries are all rendered.
MODELS = []

# Render every SKIP-th timestep (10 -> 50 frames out of 500).
SKIP = 10

# ---- comparison task(s) ------------------------------------------------------
#   gt/pred/error : (color_scalar, warp_vector)
# Error is drawn on the GT-warped workpiece by default ('gt_disp').
COMPARISON_TASKS = [
    dict(label="Displacement_Y",
         gt=("disp_y_gt", "gt_disp"), pred=("disp_y_pred", "pred_disp"),
         error=("pos_error", "pred_disp"),          # error on the PREDICTED shape
         value_title="y_disp", error_title="Pos Error", err_log=False),

    dict(label="Stress",
         gt=("gt_stress", "gt_disp"), pred=("pred_stress", "pred_disp"),
         error=("stress_error", "pred_disp"),       # error on the PREDICTED shape
         value_title="stress", error_title="Stress Error", err_log=False),
]

# ---- appearance --------------------------------------------------------------
VALUE_CMAP = "Jet"
ERROR_CMAP = "Jet"

# Die (rigid tool) appearance. Kept for context but SEMI-TRANSPARENT so it does
# not hide the workpiece. Set SHOW_DIE=False to drop it from the value panels
# entirely (error panels never show the die).
SHOW_DIE    = True
DIE_COLOR   = [0.769, 0.643, 0.518]   # tan (#C4A484)
DIE_OPACITY = 0.35                     # 1.0 = solid, ~0.3 = see-through

# 3D view: looks straight down +Z (workpiece cross-section facing you, +Y up,
# like animate_rollout) then tilts by these small angles for a 3D feel.
# Raise for a more isometric look; ~8/8 is nearly front-on.
CAMERA_AZIMUTH   = 20.0
CAMERA_ELEVATION = 12.0
MANUAL_CAMERA = False
CAMERA = dict(
    position       = [1670.825, -906.853, 1470.499],
    focal_point    = [3.393, -55.007, -41.802],
    view_up        = [-0.302, 0.647, 0.698],
    parallel_scale = 351.635,
)

# ---- output / quality --------------------------------------------------------
KEEP_FRAMES        = True
MAKE_MP4           = True
MAKE_GIF           = True
STITCH_WITH_LEGEND = True     # place value/error legends beside the panels
USE_FXAA           = True     # anti-aliasing -> smoother edges
PANEL_WIDTH   = 1500          # higher resolution -> crisper render
PANEL_HEIGHT  = 1050
LEGEND_WIDTH  = 320
FRAME_RATE    = 12


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def discover_units():
    """[(label, pvd_path)] for every series.pvd under VTU_ROOT (any depth)."""
    units = []
    for pvd in sorted(glob.glob(os.path.join(VTU_ROOT, "**", "series.pvd"),
                                recursive=True)):
        label = os.path.relpath(os.path.dirname(pvd), VTU_ROOT)
        if MODELS and label.split(os.sep)[0] not in MODELS:
            continue
        units.append((label, pvd))
    return units


def set_threshold(th, lo, hi):
    """Set threshold range across ParaView API versions (5.9 vs 5.10+)."""
    try:
        th.LowerThreshold = lo
        th.UpperThreshold = hi
    except Exception:
        pass
    try:
        th.ThresholdRange = [lo, hi]
    except Exception:
        pass


def workpiece(warp):
    """Threshold a warped source down to workpiece cells (is_die == 0)."""
    th = Threshold(Input=warp)
    th.Scalars = ['POINTS', 'is_die']
    set_threshold(th, -0.5, 0.5)
    return th


def die_part(warp):
    """Threshold a warped source down to die cells (is_die == 1)."""
    th = Threshold(Input=warp)
    th.Scalars = ['POINTS', 'is_die']
    set_threshold(th, 0.5, 1.5)
    return th


def get_global_range(src, color_var, step_times):
    """Min/max of color_var over the chosen timesteps -> time-invariant range."""
    lo_all, hi_all = float('inf'), float('-inf')
    for t in step_times:
        src.UpdatePipeline(t)
        arr = src.GetDataInformation().GetPointDataInformation().GetArrayInformation(color_var)
        if arr:
            nc = arr.GetNumberOfComponents()
            lo, hi = arr.GetComponentRange(0 if nc == 1 else -1)
            lo_all = min(lo_all, lo)
            hi_all = max(hi_all, hi)
    if lo_all == float('inf'):
        return 0.0, 1.0
    if lo_all == hi_all:
        hi_all = lo_all + 1.0
    return lo_all, hi_all


def workpiece_range(reader, color_var, warp_var, step_times):
    warp = WarpByVector(Input=reader); warp.Vectors = ['POINTS', warp_var]
    work = workpiece(warp)
    lo, hi = get_global_range(work, color_var, step_times)
    Delete(work); Delete(warp)
    return lo, hi


def config_lut(color_var, cmap, cmin, cmax, log_scale):
    lut = GetColorTransferFunction(color_var)
    pwf = GetOpacityTransferFunction(color_var)
    try:
        lut.ApplyPreset(cmap, True)
    except Exception:
        pass
    if log_scale:
        lo = max(cmin, 1e-10); hi = max(cmax, lo * 10)
        lut.UseLogScale = 1
        lut.RescaleTransferFunction(lo, hi); pwf.RescaleTransferFunction(lo, hi)
    else:
        lut.UseLogScale = 0
        lut.RescaleTransferFunction(cmin, cmax); pwf.RescaleTransferFunction(cmin, cmax)
    lut.AutomaticRescaleRangeMode = 'Never'
    return lut


def set_camera(view, reader):
    tmp = Show(reader, view); view.Update()
    SetActiveView(view)
    if MANUAL_CAMERA:
        view.CameraPosition      = CAMERA["position"]
        view.CameraFocalPoint    = CAMERA["focal_point"]
        view.CameraViewUp        = CAMERA["view_up"]
        view.CameraParallelScale = CAMERA["parallel_scale"]
    else:
        # Deterministic front view: look along +Z with +Y up (animate_rollout
        # style), then tilt slightly for a 3D feel. The second ResetCamera
        # refits the zoom while KEEPING this new orientation.
        ResetCamera(view)
        cam = GetActiveCamera()
        fp = cam.GetFocalPoint()
        dist = cam.GetDistance()
        cam.SetFocalPoint(fp[0], fp[1], fp[2])
        cam.SetPosition(fp[0], fp[1], fp[2] + dist)
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.Azimuth(CAMERA_AZIMUTH)
        cam.Elevation(CAMERA_ELEVATION)
        ResetCamera(view)
        Render(view)              # renderer recomputes the clipping range here
    Hide(reader, view); Delete(tmp)


def save_legend(color_var, title, cmin, cmax, cmap, out_png, log_scale=False):
    """Render just the colour bar into its own PNG (PANEL_HEIGHT tall)."""
    lut = config_lut(color_var, cmap, cmin, cmax, log_scale)
    lv = CreateView('RenderView')
    lv.ViewSize = [LEGEND_WIDTH, PANEL_HEIGHT]
    lv.Background = [1.0, 1.0, 1.0]
    lv.OrientationAxesVisibility = 0
    bar = GetScalarBar(lut, lv)
    bar.Visibility      = 1
    bar.Title           = title
    bar.ComponentTitle  = ''
    bar.WindowLocation  = 'Any Location'
    bar.Position        = [0.28, 0.1]
    bar.ScalarBarLength = 0.8
    for attr, val in (("ScalarBarThickness", 18),
                      ("TitleColor", [0, 0, 0]), ("LabelColor", [0, 0, 0])):
        try:
            setattr(bar, attr, val)
        except Exception:
            pass
    Render(lv)
    SaveScreenshot(out_png, lv, ImageResolution=[LEGEND_WIDTH, PANEL_HEIGHT])
    Delete(lv); del lv
    return out_png


def render_panel(reader, view, color_var, warp_var, cmin, cmax, cmap,
                 steps, timesteps, out_dir, include_die, log_scale=False):
    """Save one PNG per selected timestep: workpiece coloured (+ moving die)."""
    os.makedirs(out_dir, exist_ok=True)
    warp = WarpByVector(Input=reader); warp.Vectors = ['POINTS', warp_var]

    work = workpiece(warp)
    dwork = Show(work, view); dwork.Representation = 'Surface'
    Hide(reader, view)
    ColorBy(dwork, ('POINTS', color_var))
    config_lut(color_var, cmap, cmin, cmax, log_scale)
    dwork.SetScalarBarVisibility(view, False)     # no overlaid bar

    die = ddie = None
    if include_die and SHOW_DIE:
        die = die_part(warp)
        ddie = Show(die, view); ddie.Representation = 'Surface'
        # Solid colour. Avoid ColorBy(rep, None): some builds choke on the
        # 'NONE' association string. Setting an empty ColorArrayName turns off
        # scalar colouring so DiffuseColor/AmbientColor are used instead.
        for cname in (['', ''], ['POINTS', ''], [None, '']):
            try:
                ddie.ColorArrayName = cname
                break
            except Exception:
                continue
        try:
            ddie.DiffuseColor = DIE_COLOR
            ddie.AmbientColor = DIE_COLOR
            ddie.Opacity      = DIE_OPACITY   # semi-transparent -> see workpiece
        except Exception:
            pass

    view.Update()
    for k, t in enumerate(steps):
        view.ViewTime = timesteps[t]
        Render(view)
        SaveScreenshot(os.path.join(out_dir, f"frame.{k:04d}.png"),
                       view, ImageResolution=[PANEL_WIDTH, PANEL_HEIGHT])

    Hide(work, view)
    if ddie is not None:
        Hide(die, view)
    Delete(dwork)
    if ddie is not None:
        Delete(ddie)
    Delete(work)
    if die is not None:
        Delete(die)
    Delete(warp)


def stitch(sources, combined_dir, n):
    """sources: list of ('dir', path) per-frame or ('img', path) static."""
    os.makedirs(combined_dir, exist_ok=True)
    for k in range(n):
        readers = []
        for kind, p in sources:
            fn = os.path.join(p, f"frame.{k:04d}.png") if kind == 'dir' else p
            r = vtkPNGReader(); r.SetFileName(fn); r.Update()
            readers.append(r)
        app = vtkImageAppend(); app.SetAppendAxis(0)
        for r in readers:
            app.AddInputConnection(r.GetOutputPort())
        app.Update()
        w = vtkPNGWriter()
        w.SetFileName(os.path.join(combined_dir, f"frame.{k:04d}.png"))
        w.SetInputConnection(app.GetOutputPort()); w.Write()
    print(f"      stitched {n} combined frames -> {combined_dir}")


def encode_mp4(png_dir, out_mp4):
    import subprocess
    cmd = ['ffmpeg', '-y', '-framerate', str(FRAME_RATE),
           '-i', os.path.join(png_dir, 'frame.%04d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(FRAME_RATE), out_mp4]
    try:
        if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
            return True
    except FileNotFoundError:
        pass
    print(f"      NOTE: ffmpeg not found; PNGs are in {png_dir}")
    return False


def encode_gif(png_dir, out_gif):
    import subprocess
    flt = (f'fps={FRAME_RATE},scale=-1:-1:flags=lanczos,'
           f'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse')
    cmd = ['ffmpeg', '-y', '-framerate', str(FRAME_RATE),
           '-i', os.path.join(png_dir, 'frame.%04d.png'),
           '-filter_complex', flt, out_gif]
    try:
        if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
            return True
    except FileNotFoundError:
        pass
    return False


# ==============================================================================
# 3. PER-UNIT (model/geometry) PROCESSING
# ==============================================================================
def process_unit(label, pvd, view):
    out_dir = os.path.join(OUTPUT_ROOT, label)
    print(f"\n{'#'*70}\n  UNIT: {label}\n  in : {pvd}\n  out: {out_dir}\n{'#'*70}")
    if not os.path.exists(pvd):
        print("  ERROR: series.pvd not found -- skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)

    reg = label.replace(os.sep, "_").replace("-", "_")
    reader = PVDReader(registrationName=f'pvd_{reg}', FileName=pvd)
    timesteps = reader.TimestepValues
    if timesteps is None:
        timesteps = [0]
    T = len(timesteps)
    steps = list(range(0, T, SKIP))
    step_times = [timesteps[t] for t in steps]
    print(f"  {T} timesteps -> rendering {len(steps)} (skip={SKIP})")

    GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
    GetLayout().SetSize(PANEL_WIDTH, PANEL_HEIGHT)
    set_camera(view, reader)

    # array availability check (peek at first timestep)
    reader.UpdatePipeline(timesteps[0])
    pdi = reader.GetDataInformation().GetPointDataInformation()
    available = {pdi.GetArrayInformation(i).GetName() for i in range(pdi.GetNumberOfArrays())}
    print(f"  arrays: {sorted(available)}")

    for task in COMPARISON_TASKS:
        label_t = task["label"]
        gt_c, pred_c, err_c = task["gt"], task["pred"], task["error"]
        err_log = task.get("err_log", False)
        value_title = task.get("value_title", gt_c[0])
        error_title = task.get("error_title", err_c[0])
        print(f"\n  {'='*56}\n  {label_t}\n  {'='*56}")

        needed = {gt_c[0], pred_c[0], err_c[0], gt_c[1], pred_c[1], err_c[1], 'is_die'}
        missing = needed - available
        if missing:
            print(f"  SKIP '{label_t}' -- missing arrays: {sorted(missing)}")
            continue

        # ranges on the WORKPIECE only, over the rendered timesteps
        gmin, gmax = workpiece_range(reader, gt_c[0],   gt_c[1],   step_times)
        pmin, pmax = workpiece_range(reader, pred_c[0], pred_c[1], step_times)
        _, emax    = workpiece_range(reader, err_c[0],  err_c[1],  step_times)
        value_min, value_max = min(gmin, pmin), max(gmax, pmax)
        error_min, error_max = 0.0, emax
        print(f"  value range (GT&Pred): [{value_min:.4g}, {value_max:.4g}]")
        print(f"  error range (this geometry): [{error_min:.4g}, {error_max:.4g}]")

        base     = os.path.join(out_dir, "frames", label_t)
        gt_dir   = os.path.join(base, "gt")
        pred_dir = os.path.join(base, "pred")
        err_dir  = os.path.join(base, "error")
        comb_dir = os.path.join(base, "combined")

        # legends as their own files (beside, not overlaid)
        val_leg = os.path.join(out_dir, f"legend_{label_t}_value.png")
        err_leg = os.path.join(out_dir, f"legend_{label_t}_error.png")
        save_legend(pred_c[0], value_title, value_min, value_max, VALUE_CMAP, val_leg)
        save_legend(err_c[0],  error_title, error_min, error_max, ERROR_CMAP, err_leg, err_log)
        SetActiveView(view)

        print("  rendering GT ...")
        render_panel(reader, view, gt_c[0], gt_c[1], value_min, value_max, VALUE_CMAP,
                     steps, timesteps, gt_dir, include_die=True)
        print("  rendering Pred ...")
        render_panel(reader, view, pred_c[0], pred_c[1], value_min, value_max, VALUE_CMAP,
                     steps, timesteps, pred_dir, include_die=True)
        print("  rendering Error (workpiece only) ...")
        render_panel(reader, view, err_c[0], err_c[1], error_min, error_max, ERROR_CMAP,
                     steps, timesteps, err_dir, include_die=False, log_scale=err_log)

        if STITCH_WITH_LEGEND:
            sources = [('dir', gt_dir), ('dir', pred_dir), ('img', val_leg),
                       ('dir', err_dir), ('img', err_leg)]
        else:
            sources = [('dir', gt_dir), ('dir', pred_dir), ('dir', err_dir)]
        stitch(sources, comb_dir, len(steps))

        if MAKE_MP4 and encode_mp4(comb_dir, os.path.join(out_dir, f"{label_t}.mp4")):
            print(f"  mp4 -> {label_t}.mp4")
        if MAKE_GIF and encode_gif(comb_dir, os.path.join(out_dir, f"{label_t}.gif")):
            print(f"  gif -> {label_t}.gif")

        if not KEEP_FRAMES:
            for d in (gt_dir, pred_dir, err_dir, comb_dir):
                shutil.rmtree(d, ignore_errors=True)

    Delete(reader); del reader


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    paraview.simple._DisableFirstRenderCameraReset()

    view = GetActiveViewOrCreate('RenderView')
    view.UseColorPaletteForBackground = 0
    view.Background                = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility      = 0
    try:
        view.UseFXAA = 1 if USE_FXAA else 0
    except Exception:
        pass

    units = discover_units()
    if not units:
        print(f"No series.pvd found under {VTU_ROOT}. Run convert_pkls_to_vtu.py first.")
        return
    print(f"Units to render ({len(units)}):")
    for label, _ in units:
        print(f"   - {label}")

    for idx, (label, pvd) in enumerate(units, 1):
        print(f"\n{'*'*70}\n  {idx}/{len(units)}: {label}\n{'*'*70}")
        process_unit(label, pvd, view)

    print(f"\n{'='*70}\n  ALL DONE -> {OUTPUT_ROOT}\n{'='*70}")


if __name__ == "__main__":
    main()