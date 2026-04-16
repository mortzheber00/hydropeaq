"""
Convert all robot link meshes from URDF format to .obj
Check and repair mesh quality for SPlisHSPlasH boundary sampling
"""
import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET

def get_mesh_files_from_urdf(urdf_path):
    """
    Parse URDF and extract visual mesh file paths per link.
    We use VISUAL not collision — full geometry as requested.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Find package path if using package:// URIs
    # URDF is at src/amph/urdf/amph.urdf, package root is src/amph/
    package_path = os.path.dirname(os.path.dirname(urdf_path))
    
    link_meshes = {}
    
    for link in root.findall('link'):
        link_name = link.get('name')
        
        # Get visual mesh — use visual, not collision
        visual = link.find('visual')
        if visual is None:
            continue
            
        geometry = visual.find('geometry')
        if geometry is None:
            continue
            
        mesh_elem = geometry.find('mesh')
        if mesh_elem is None:
            continue
        
        filename = mesh_elem.get('filename', '')
        scale_str = mesh_elem.get('scale', '1 1 1')
        scale = [float(s) for s in scale_str.split()]
        
        # Resolve package:// URIs
        if filename.startswith('package://'):
            filename = filename.replace('package://', '')
            # Remove package name from path
            parts = filename.split('/', 1)
            filename = os.path.join(package_path, parts[1] if len(parts) > 1 else parts[0])
        elif not os.path.isabs(filename):
            filename = os.path.join(package_path, filename)
        
        link_meshes[link_name] = {
            'path':  filename,
            'scale': scale
        }
    
    return link_meshes


def prepare_mesh(link_name, mesh_info, output_dir, particle_radius=0.008):
    """
    Load, validate, repair, and export a single link mesh as .obj
    Returns True if successful
    """
    input_path  = mesh_info['path']
    scale       = mesh_info['scale']
    output_path = os.path.join(output_dir, f'{link_name}.obj')
    
    print(f"\n{link_name}:")
    print(f"  Input: {input_path}")
    
    # Load mesh
    try:
        mesh = trimesh.load(input_path, force='mesh')
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return False
    
    # Apply scale from URDF
    if scale != [1.0, 1.0, 1.0]:
        mesh.apply_scale(scale)
        print(f"  Applied scale: {scale}")
    
    # Report initial state
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces:    {len(mesh.faces)}")
    print(f"  Bounds:   {mesh.bounds}")
    print(f"  Volume:   {mesh.volume*1e6:.2f} cm³")
    
    # Check minimum feature size vs particle radius
    extents = mesh.bounding_box.extents
    min_dim = extents.min()
    print(f"  Min dimension: {min_dim*1000:.1f}mm")
    print(f"  Particle radius: {particle_radius*1000:.1f}mm")
    
    if min_dim < 2 * particle_radius:
        print(f"  ⚠ Min dimension < 2×particle_radius")
        print(f"    SPH resolution may be insufficient for this link")
        print(f"    Consider reducing particle_radius or simplifying geometry")
    
    # Repair watertight issues
    if not mesh.is_watertight:
        print(f"  Repairing: not watertight...")
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        
        if mesh.is_watertight:
            print(f"  ✓ Repaired successfully")
        else:
            print(f"  ⚠ Could not fully repair — boundary sampling may be inaccurate")
    else:
        print(f"  ✓ Already watertight")
    
    # Decimate if very high polygon count
    # SPH boundary sampling doesn't benefit from >10K faces
    #if len(mesh.faces) > 20000:
    #    print(f"  Decimating {len(mesh.faces)} → 8000 faces...")
    #    mesh = mesh.simplify_quadric_decimation(8000)
    #    print(f"  ✓ Decimated to {len(mesh.faces)} faces")
    
    # Export as .obj
    mesh.export(output_path)
    print(f"  ✓ Saved: {output_path}")
    
    return True


def main():
    URDF_PATH      = 'src/amph/urdf/amph.urdf'   # adjust to your path
    OUTPUT_DIR     = 'stage2_gait_validation/sph_meshes'
    PARTICLE_RADIUS = 0.015             # adjust to your robot scale
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Parse URDF for mesh paths
    link_meshes = get_mesh_files_from_urdf(URDF_PATH)
    print(f"Found {len(link_meshes)} links with meshes")
    
    # Process each link
    results = {}
    for link_name, mesh_info in link_meshes.items():
        success = prepare_mesh(
            link_name, mesh_info, OUTPUT_DIR, PARTICLE_RADIUS
        )
        results[link_name] = success
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Mesh preparation summary:")
    ok    = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    print(f"  ✓ Ready:  {len(ok)} links")
    if failed:
        print(f"  ✗ Failed: {len(failed)} links: {failed}")
    
    # Estimate particle count for different radii
    print(f"\nParticle count estimates (for guidance):")
    import json
    
    # Compute fluid domain from mesh bounds
    all_bounds = []
    for link_name in ok:
        mesh = trimesh.load(f'{OUTPUT_DIR}/{link_name}.obj')
        all_bounds.append(mesh.bounds)
    
    all_bounds = np.array(all_bounds)
    domain_min = all_bounds[:, 0, :].min(axis=0) - 0.5  # 0.5m margin
    domain_max = all_bounds[:, 1, :].max(axis=0) + 0.5
    domain_max[2] = min(domain_max[2] + 0.1, 0.05)  # cap at water surface
    
    domain_vol = np.prod(domain_max - domain_min)
    
    for r in [0.012, 0.010, 0.008, 0.006]:
        n = domain_vol / (2*r)**3 * 0.64
        print(f"  r={r*1000:.0f}mm: ~{n/1000:.0f}K particles")


if __name__ == '__main__':
    main()