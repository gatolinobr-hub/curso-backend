#!/usr/bin/env python3
"""
Blender scene: Ophanim eye + interlocking eye-studded rings.
Renders transparent RGBA frames for compositing onto the dark artwork.

Run:
  blender -b -P render_scene.py -- --variant computador --out /path/frames
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/ophanim")
TEX = ROOT / "textures"
DURATION = 10.0
FPS = 24
N_FRAMES = int(DURATION * FPS)  # 240


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def load_image(path: Path):
    return bpy.data.images.load(str(path), check_existing=True)


def make_ring_material(name: str, img) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.projection = "FLAT"
    tex.location = (-300, 0)
    # Prefer UV from torus
    uv = nodes.new("ShaderNodeTexCoord")
    uv.location = (-600, 0)
    mapn = nodes.new("ShaderNodeMapping")
    mapn.location = (-450, 0)
    mapn.inputs["Scale"].default_value = (4.0, 1.0, 1.0)
    links.new(uv.outputs["UV"], mapn.inputs["Vector"])
    links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.55
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.25
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    # Slight emission to match fiery look
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.6, 0.15, 0.05, 1)
        bsdf.inputs["Emission Strength"].default_value = 0.15
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_eye_materials(eye_img, iris_img):
    # Sclera / photo eye plate
    mat_eye = bpy.data.materials.new(name="EyePlate")
    mat_eye.use_nodes = True
    mat_eye.blend_method = "HASHED"
    nt = mat_eye.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = eye_img
    # Circular alpha
    texcoord = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(texcoord.outputs["UV"], sep.inputs["Vector"])
    # (u-0.5)^2+(v-0.5)^2
    math_x = nodes.new("ShaderNodeMath")
    math_x.operation = "SUBTRACT"
    math_x.inputs[1].default_value = 0.5
    math_y = nodes.new("ShaderNodeMath")
    math_y.operation = "SUBTRACT"
    math_y.inputs[1].default_value = 0.5
    links.new(sep.outputs["X"], math_x.inputs[0])
    links.new(sep.outputs["Y"], math_y.inputs[0])
    powx = nodes.new("ShaderNodeMath")
    powx.operation = "POWER"
    powx.inputs[1].default_value = 2.0
    powy = nodes.new("ShaderNodeMath")
    powy.operation = "POWER"
    powy.inputs[1].default_value = 2.0
    links.new(math_x.outputs[0], powx.inputs[0])
    links.new(math_y.outputs[0], powy.inputs[0])
    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    links.new(powx.outputs[0], add.inputs[0])
    links.new(powy.outputs[0], add.inputs[1])
    sqrt = nodes.new("ShaderNodeMath")
    sqrt.operation = "SQRT"
    links.new(add.outputs[0], sqrt.inputs[0])
    # alpha = 1 - smoothstep
    less = nodes.new("ShaderNodeMath")
    less.operation = "LESS_THAN"
    less.inputs[1].default_value = 0.48
    links.new(sqrt.outputs[0], less.inputs[0])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Alpha" in bsdf.inputs:
        links.new(less.outputs[0], bsdf.inputs["Alpha"])
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.35
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.2, 0.55, 0.7, 1)
        bsdf.inputs["Emission Strength"].default_value = 0.4
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Iris overlay (moves)
    mat_iris = bpy.data.materials.new(name="Iris")
    mat_iris.use_nodes = True
    mat_iris.blend_method = "BLEND"
    nt2 = mat_iris.node_tree
    n2, l2 = nt2.nodes, nt2.links
    n2.clear()
    out2 = n2.new("ShaderNodeOutputMaterial")
    bsdf2 = n2.new("ShaderNodeBsdfPrincipled")
    tex2 = n2.new("ShaderNodeTexImage")
    tex2.image = iris_img
    l2.new(tex2.outputs["Color"], bsdf2.inputs["Base Color"])
    if "Alpha" in bsdf2.inputs:
        l2.new(tex2.outputs["Alpha"], bsdf2.inputs["Alpha"])
    if "Emission Color" in bsdf2.inputs:
        bsdf2.inputs["Emission Color"].default_value = (0.3, 0.8, 1.0, 1)
        bsdf2.inputs["Emission Strength"].default_value = 1.2
    if "Roughness" in bsdf2.inputs:
        bsdf2.inputs["Roughness"].default_value = 0.2
    l2.new(bsdf2.outputs["BSDF"], out2.inputs["Surface"])

    # Eyelid
    mat_lid = bpy.data.materials.new(name="Eyelid")
    mat_lid.use_nodes = True
    nt3 = mat_lid.node_tree
    n3, l3 = nt3.nodes, nt3.links
    n3.clear()
    out3 = n3.new("ShaderNodeOutputMaterial")
    bsdf3 = n3.new("ShaderNodeBsdfPrincipled")
    bsdf3.inputs["Base Color"].default_value = (0.12, 0.04, 0.03, 1)
    if "Roughness" in bsdf3.inputs:
        bsdf3.inputs["Roughness"].default_value = 0.7
    l3.new(bsdf3.outputs["BSDF"], out3.inputs["Surface"])
    return mat_eye, mat_iris, mat_lid


def add_torus(name, major, minor, location, rotation_euler, material):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major,
        minor_radius=minor,
        major_segments=128,
        minor_segments=48,
        location=location,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = Euler(rotation_euler, "XYZ")
    # UV unwrap for texture flow along ring
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.cylinder_project(direction="VIEW", correct_aspect=True)
    except Exception:
        bpy.ops.uv.smart_project(angle_limit=math.radians(66))
    bpy.ops.object.mode_set(mode="OBJECT")
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    return obj


def setup_world_and_lights():
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0, 0, 0, 1)
    bg.inputs[1].default_value = 0.0

    # Key warm light (nebula feel)
    bpy.ops.object.light_add(type="AREA", location=(0, -2.5, 2.0))
    key = bpy.context.active_object
    key.data.energy = 250
    key.data.color = (1.0, 0.45, 0.2)
    key.data.size = 4
    key.rotation_euler = Euler((math.radians(60), 0, 0), "XYZ")

    bpy.ops.object.light_add(type="AREA", location=(0, 2.0, -1.5))
    fill = bpy.context.active_object
    fill.data.energy = 80
    fill.data.color = (0.4, 0.7, 1.0)
    fill.data.size = 3

    bpy.ops.object.light_add(type="POINT", location=(0, 0, 0.2))
    core = bpy.context.active_object
    core.data.energy = 40
    core.data.color = (0.5, 0.9, 1.0)


def setup_camera(variant: str):
    bpy.ops.object.camera_add(location=(0, -4.6, 0))
    cam = bpy.context.active_object
    cam.rotation_euler = Euler((math.radians(90), 0, 0), "XYZ")
    cam.data.lens = 50
    bpy.context.scene.camera = cam
    # Match aspect framing: rings fill similar portion of frame
    if variant == "celular":
        cam.location = (0, -5.2, 0.05)
        cam.data.lens = 55
    return cam


def setup_render(variant: str, out_dir: Path, scale: int = 100):
    scene = bpy.context.scene
    # Cycles CPU is reliable in this headless environment (EEVEE/SwiftShader is slower).
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = True
    if hasattr(scene.cycles, "adaptive_threshold"):
        scene.cycles.adaptive_threshold = 0.05
    if variant == "computador":
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
    else:
        scene.render.resolution_x = 1080
        scene.render.resolution_y = 1920
    scene.render.resolution_percentage = scale
    scene.render.fps = FPS
    scene.frame_start = 0
    scene.frame_end = N_FRAMES - 1
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(out_dir / "frame_")


def blink_amount(t: float) -> float:
    """Seamless eyelid close amount in [0,1]. Two blinks per loop."""

    def pulse(center: float, width: float = 0.035) -> float:
        # distance on circle
        d = min(abs(t - center), abs(t - center + 1), abs(t - center - 1))
        return math.exp(-((d / width) ** 2))

    return max(0.0, min(1.0, pulse(0.32) + pulse(0.68)))


def iris_offset(t: float) -> tuple[float, float]:
    # Seamless saccade-like drift
    x = 0.045 * math.sin(2 * math.pi * t) + 0.02 * math.sin(4 * math.pi * t + 0.7)
    y = 0.03 * math.sin(2 * math.pi * t + 1.2) + 0.015 * math.cos(6 * math.pi * t)
    return x, y


def build_scene(variant: str):
    clear_scene()
    ring_img = load_image(TEX / "ring_eyes.png")
    eye_img = load_image(TEX / "central_eye.png")
    iris_img = load_image(TEX / "iris_cyan.png")
    ring_mat = make_ring_material("RingEyes", ring_img)
    mat_eye, mat_iris, mat_lid = make_eye_materials(eye_img, iris_img)

    # Central eye group
    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=0.55, fill_type="NGON", location=(0, 0, 0))
    eye = bpy.context.active_object
    eye.name = "CentralEye"
    eye.rotation_euler = Euler((math.radians(90), 0, 0), "XYZ")
    eye.data.materials.append(mat_eye)

    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=0.28, fill_type="NGON", location=(0, -0.02, 0))
    iris = bpy.context.active_object
    iris.name = "Iris"
    iris.rotation_euler = Euler((math.radians(90), 0, 0), "XYZ")
    iris.data.materials.append(mat_iris)

    # Upper / lower lids as flat disks that scale in Y (camera space ~ Z after rot)
    bpy.ops.mesh.primitive_plane_add(size=1.3, location=(0, -0.05, 0.55))
    lid_u = bpy.context.active_object
    lid_u.name = "LidUpper"
    lid_u.rotation_euler = Euler((math.radians(90), 0, 0), "XYZ")
    lid_u.scale = (1.0, 0.001, 1.0)
    lid_u.data.materials.append(mat_lid)

    bpy.ops.mesh.primitive_plane_add(size=1.3, location=(0, -0.05, -0.55))
    lid_l = bpy.context.active_object
    lid_l.name = "LidLower"
    lid_l.rotation_euler = Euler((math.radians(90), 0, 0), "XYZ")
    lid_l.scale = (1.0, 0.001, 1.0)
    lid_l.data.materials.append(mat_lid)

    # Interlocking rings — different rest orientations; integer revolutions / loop
    # major radius ~ wraps around eye
    rings_spec = [
        # name, major, minor, rest_euler(xyz), world axis, turns (odd ints)
        ("RingA", 1.15, 0.16, (0.35, 0.15, 0.55), "X", 1),
        ("RingB", 1.25, 0.15, (1.25, 0.4, 0.2), "Y", -1),
        ("RingC", 1.35, 0.14, (0.55, 1.1, 0.35), "Z", 1),
        ("RingD", 1.45, 0.13, (0.9, 0.25, 1.2), "X", -1),
    ]
    rings = []
    for name, maj, mn, rest, axis, turns in rings_spec:
        obj = add_torus(name, maj, mn, (0, 0, 0), rest, ring_mat)
        rings.append((obj, axis, turns, Euler(rest, "XYZ").copy()))

    setup_world_and_lights()
    setup_camera(variant)
    return eye, iris, lid_u, lid_l, rings


def insert_anim(eye, iris, lid_u, lid_l, rings):
    scene = bpy.context.scene
    from mathutils import Quaternion, Vector

    for f in range(N_FRAMES):
        t = f / N_FRAMES
        scene.frame_set(f)

        # Rings: integer world-axis turns (matches Three.js scene)
        for obj, axis, turns, rest in rings:
            angle = 2 * math.pi * turns * t
            axis_vec = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}[axis]
            q_spin = Quaternion(axis_vec, angle)
            q_rest = rest.to_quaternion()
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = q_spin @ q_rest
            obj.keyframe_insert(data_path="rotation_quaternion", frame=f)

        # Iris look
        ox, oy = iris_offset(t)
        iris.location = (ox, -0.02, oy)
        iris.keyframe_insert(data_path="location", frame=f)

        # Blink
        b = blink_amount(t)
        lid_u.location = (0, -0.05, 0.55 - 0.50 * b)
        lid_l.location = (0, -0.05, -0.55 + 0.50 * b)
        lid_u.scale = (1.05, 0.001 + 0.55 * b, 1.0)
        lid_l.scale = (1.05, 0.001 + 0.55 * b, 1.0)
        lid_u.keyframe_insert(data_path="location", frame=f)
        lid_l.keyframe_insert(data_path="location", frame=f)
        lid_u.keyframe_insert(data_path="scale", frame=f)
        lid_l.keyframe_insert(data_path="scale", frame=f)

    for obj, _, _, _ in rings:
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["computador", "celular"], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--preview-only", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    eye, iris, lid_u, lid_l, rings = build_scene(args.variant)
    insert_anim(eye, iris, lid_u, lid_l, rings)
    setup_render(args.variant, out_dir)

    # Save .blend for reproducibility
    blend_path = ROOT / "src" / f"ophanim_{args.variant}.blend"
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    scene = bpy.context.scene
    if args.preview_only:
        for pct in (0, 25, 50, 75):
            f = int(round((pct / 100) * (N_FRAMES - 1)))
            scene.frame_set(f)
            scene.render.filepath = str(out_dir / f"preview_{pct:02d}.png")
            bpy.ops.render.render(write_still=True)
            print("preview", pct, "frame", f)
    else:
        # Render animation
        scene.render.filepath = str(out_dir / "frame_")
        bpy.ops.render.render(animation=True)
    print("RENDER_DONE", args.variant)


if __name__ == "__main__":
    main()
