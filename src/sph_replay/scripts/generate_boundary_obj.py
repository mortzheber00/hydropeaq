#!/usr/bin/env python3
"""Convert URDF collision STL meshes to OBJ files for the SPlisHSPlasH boundary simulator.

The Gazebo fluid plugin (FluidSimulator) exports collision meshes via Gazebo's
MeshManager, but that export silently fails when the mesh isn't cached under
the expected key.  This script pre-generates the OBJ files the plugin expects
so that GazeboBoundarySimulator::initBoundaryData() can load them directly.

Naming convention the plugin uses:
    <output_dir>/<model_name>_<collision_name>.obj

Run once before roslaunch (it has to finish before Gazebo loads the world, so
it is deliberately not a launch node).  The argument is a registered robot
name; see hydro_model/robots/:

    rosrun sph_replay generate_boundary_obj.py amph
    rosrun sph_replay generate_boundary_obj.py body2
"""

import struct
import os
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = "/home/ws"
sys.path.insert(0, REPO_ROOT)
from stage1_gait_optimization.hydro_model import get_spec  # noqa: E402

OUTPUT_DIR = "/home/ws/sph_boundaries"

# Collision names that cannot be read off the URDF, because a fixed joint lumps
# its child into the parent under a mangled name.  Keyed by registered robot
# name; values are STL basenames under <package_dir>/meshes.  A robot with no
# fixed joints needs no entry here -- see collisions_from_urdf().
AMPH_COLLISION_TO_STL = {
    "base_link_collision":                                                          "base_link.STL",
    "Front_Left_Side_link_collision":                                               "Front_Left_Side_link.STL",
    "Front_Left_Thigh_link_collision":                                              "Front_Left_Thigh_link.STL",
    "Front_Left_Calf_link_collision":                                               "Front_Left_Calf_link.STL",
    "Front_Left_Calf_link_fixed_joint_lump__Front_Left_Foot_link_collision_1":      "Front_Left_Foot_link.STL",
    "Front_Right_Side_link_collision":                                              "Front_Right_Side_link.STL",
    "Front_Right_Thigh_link_collision":                                             "Front_Right_Thigh_link.STL",
    "Front_Right_Calf_link_collision":                                              "Front_Right_Calf_link.STL",
    "Front_Right_Calf_link_fixed_joint_lump__Front_Right_Foot_link_collision_1":    "Front_Right_Foot_link.STL",
    "Hind_Left_Side_link_collision":                                                "Hind_Left_Side_link.STL",
    "Hind_Left_Thigh_link_collision":                                               "Hind_Left_Thigh_link.STL",
    "Hind_Left_Calf_link_collision":                                                "Hind_Left_Calf_link.STL",
    "Hind_Left_Calf_link_fixed_joint_lump__Hind_Left_Foot_link_collision_1":        "Hind_Left_Foot_link.STL",
    "Hind_Right_Side_link_collision":                                               "Hind_Right_Side_link.STL",
    "Hind_Right_Thigh_link_collision":                                              "Hind_Right_Thigh_link.STL",
    "Hind_Right_Calf_link_collision":                                               "Hind_Right_Calf_link.STL",
    "Hind_Right_Calf_link_fixed_joint_lump__Hind_Right_Foot_link_collision_1":      "Hind_Right_Foot_link.STL",
}

LUMPED_COLLISIONS = {"amph": AMPH_COLLISION_TO_STL}


def collisions_from_urdf(urdf_path):
    """Collision name -> STL basename, read straight off the URDF.

    urdf_to_sdf names each collision ``<link>_collision`` and lumps
    fixed-joint children into their parent under a mangled name -- which is
    what the amph map above spells out by hand.  A URDF with no fixed joints
    has no lumping, so the mapping is one plain entry per link; anything else
    needs its names listed explicitly.
    """
    root = ET.parse(urdf_path).getroot()
    fixed = [j.get("name") for j in root.findall("joint") if j.get("type") == "fixed"]
    if fixed:
        raise ValueError(
            f"{urdf_path} has fixed joint(s) {fixed}; their children get lumped into "
            f"the parent link's collision under a mangled name, so the collision names "
            f"have to be listed explicitly (see LUMPED_COLLISIONS)"
        )

    out = {}
    for link in root.findall("link"):
        mesh = link.find("collision/geometry/mesh")
        if mesh is None:
            continue
        out[f"{link.get('name')}_collision"] = os.path.basename(mesh.get("filename"))
    return out


def robot_config(name):
    """Return (Gazebo model name, mesh dir, collision -> STL) for one robot.

    Everything comes from the robot's registry entry, so adding a robot to
    hydro_model/robots is enough unless its URDF has fixed joints.
    """
    spec = get_spec(name)
    collisions = LUMPED_COLLISIONS.get(spec.name) or collisions_from_urdf(spec.urdf_path)
    return spec.ros, os.path.join(str(spec.package_dir), "meshes"), collisions


def read_stl(path):
    """Return (vertices, faces) from a binary or ASCII STL file.

    vertices : list of (x, y, z) float tuples  (deduplicated, shared)
    faces    : list of (i, j, k) int tuples     (0-based indices)
    """
    with open(path, "rb") as f:
        header = f.read(80)

    # ASCII STL starts with "solid"
    if header[:5] == b"solid":
        return _read_stl_ascii(path)
    return _read_stl_binary(path)


def _read_stl_binary(path):
    vertex_map = {}
    vertices = []
    faces = []

    with open(path, "rb") as f:
        f.read(80)  # header
        (num_triangles,) = struct.unpack("<I", f.read(4))
        for _ in range(num_triangles):
            f.read(12)  # normal — ignored; let the OBJ loader recompute
            tri = []
            for _ in range(3):
                v = struct.unpack("<fff", f.read(12))
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(v)
                tri.append(vertex_map[v])
            faces.append(tuple(tri))
            f.read(2)  # attribute byte count

    return vertices, faces


def _read_stl_ascii(path):
    vertex_map = {}
    vertices = []
    faces = []

    with open(path, "r") as f:
        tri = []
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                v = (float(parts[1]), float(parts[2]), float(parts[3]))
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(v)
                tri.append(vertex_map[v])
            elif line.startswith("endfacet"):
                if len(tri) == 3:
                    faces.append(tuple(tri))
                tri = []

    return vertices, faces


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        f.write("# Generated by generate_boundary_obj.py\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ uses 1-based indices
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {os.path.basename(sys.argv[0])} <robot>\n"
              f"       <robot> is a registered robot name; see hydro_model/robots/",
              file=sys.stderr)
        sys.exit(2)
    model_name, mesh_dir, collision_to_stl = robot_config(sys.argv[1])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = 0
    for collision_name, stl_file in collision_to_stl.items():
        stl_path = os.path.join(mesh_dir, stl_file)
        obj_name = f"{model_name}_{collision_name}.obj"
        obj_path = os.path.join(OUTPUT_DIR, obj_name)

        if not os.path.exists(stl_path):
            print(f"[ERROR] STL not found: {stl_path}", file=sys.stderr)
            errors += 1
            continue

        try:
            vertices, faces = read_stl(stl_path)
            write_obj(obj_path, vertices, faces)
            print(f"[OK] {obj_name}  ({len(vertices)} verts, {len(faces)} tris)")
        except Exception as e:
            print(f"[ERROR] {stl_file}: {e}", file=sys.stderr)
            errors += 1

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
