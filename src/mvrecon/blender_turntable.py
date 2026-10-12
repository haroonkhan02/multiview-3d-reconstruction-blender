"""Blender script: import a coloured PLY mesh and render an orbit (turntable) animation.

Runs INSIDE Blender (it only needs ``bpy``), started by ``mv3d-render``::

    blender --background --python blender_turntable.py -- --mesh m.ply --out frames/ \
        --up 0 0 1 --front 1 0 0 --sweep 360 --frames 120

* ``--up``    world direction that should point up (from the cameras: minus their mean y axis);
* ``--front`` direction from the object towards where the photos were taken;
* ``--sweep`` 360 = full turn (objects); less = swing left-right around the front (facades,
  rooms), because the back of a scene photographed from one side is empty.
"""

from __future__ import annotations

import argparse
import math
import sys

import bpy
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--up", type=float, nargs=3, default=[0, 0, 1])
    ap.add_argument("--front", type=float, nargs=3, default=[0, -1, 0])
    ap.add_argument("--center", type=float, nargs=3, default=None)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--sweep", type=float, default=360)
    ap.add_argument("--elevation", type=float, default=20)
    ap.add_argument("--distance", type=float, default=2.4, help="camera distance / radius")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--res", type=int, nargs=2, default=[960, 720])
    ap.add_argument("--samples", type=int, default=32)
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def upright_matrix(up: Vector, front: Vector, center: Vector, scale: float) -> Matrix:
    """World -> render frame: centre at origin, `up` -> +Z, `front` -> -Y, unit size."""
    z = up.normalized()
    f = (front - front.dot(z) * z).normalized()
    y = -f
    x = y.cross(z).normalized()
    R = Matrix((x, y, z))  # rows: new axes expressed in old coordinates
    M = R.to_4x4() @ Matrix.Translation(-center)
    return Matrix.Scale(1.0 / scale, 4) @ M


def vertex_color_material(obj):
    mat = bpy.data.materials.new("vertex_colour")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    attr = nodes.new("ShaderNodeVertexColor")
    if obj.data.color_attributes:
        attr.layer_name = obj.data.color_attributes[0].name
    mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    # Photo colours already contain the scene's lighting: add them as emission too, so the
    # render shows the captured colours instead of re-lit, washed-out ones.
    emission = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if emission is not None:
        mat.node_tree.links.new(attr.outputs["Color"], emission)
        bsdf.inputs["Emission Strength"].default_value = 0.6
    obj.data.materials.append(mat)


def main():
    a = parse_args()
    clear_scene()
    bpy.ops.wm.ply_import(filepath=a.mesh)
    obj = bpy.context.selected_objects[0]
    verts = [v.co.copy() for v in obj.data.vertices]
    center = Vector(a.center) if a.center else sum(verts, Vector()) / len(verts)
    radius = a.radius or sorted((v - center).length for v in verts)[int(0.95 * len(verts))]
    obj.data.transform(upright_matrix(Vector(a.up), Vector(a.front), center, radius))
    obj.data.update()
    vertex_color_material(obj)
    bpy.ops.object.shade_smooth()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = a.res
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = a.out.rstrip("/") + "/frame_"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = a.samples
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.13, 0.14, 0.16, 1)
    bg.inputs["Strength"].default_value = 1.0

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 1.5
    sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(30))
    scene.collection.objects.link(sun)

    pivot = bpy.data.objects.new("pivot", None)
    scene.collection.objects.link(pivot)
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    cam.data.lens = 40
    scene.collection.objects.link(cam)
    scene.camera = cam
    el = math.radians(a.elevation)
    cam.location = (0, -a.distance * math.cos(el), a.distance * math.sin(el))  # front = -Y
    cam.parent = pivot
    track = cam.constraints.new("TRACK_TO")
    track.target = pivot
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene.frame_start, scene.frame_end = 1, a.frames
    if a.sweep >= 360:
        keys = [(1, 0.0), (a.frames + 1, 2 * math.pi)]  # +1: last frame != first frame
    else:
        h = math.radians(a.sweep / 2)
        q = a.frames // 4
        keys = [(1, 0.0), (1 + q, h), (1 + 3 * q, -h), (a.frames + 1, 0.0)]
    for frame, angle in keys:
        pivot.rotation_euler = (0, 0, angle)
        pivot.keyframe_insert("rotation_euler", index=2, frame=frame)
    # Linear for a full turn (constant speed); smooth easing for the swing.
    if a.sweep >= 360:
        fcurves = getattr(pivot.animation_data.action, "fcurves", None)
        for fc in fcurves or []:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
