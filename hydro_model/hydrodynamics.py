"""
Simplified hydrodynamic model for an underwater quadruped robot.

Implements four force contributions for each rigid-body link:
  1. **Buoyancy** — Archimedes' principle applied to the cylinder volume.
  2. **Viscous drag** — Quadratic drag (Morison-type) with separate axial
     and transverse drag coefficients.
  3. **Pressure-gradient (Froude-Krylov)** — Force from the ambient
     pressure field on the displaced volume.
  4. **Added (virtual) mass** — Inertia of entrained water surrounding
     each moving link, using strip-theory coefficients for a circular
     cylinder.

All forces are expressed in the world frame.

References
----------
- Fossen, T. I. (2011). *Handbook of Marine Craft Hydrodynamics and
  Motion Control*. Wiley.
- Newman, J. N. (1977). *Marine Hydrodynamics*. MIT Press.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot import CylinderPrimitive, QuadrupedRobot


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

RHO_WATER = 997.0   # freshwater density [kg/m^3]
GRAVITY = 9.81       # gravitational acceleration [m/s^2]


# ---------------------------------------------------------------------------
# Per-link hydrodynamic properties
# ---------------------------------------------------------------------------

@dataclass
class LinkHydroProperties:
    """Pre-computed hydrodynamic quantities for a single cylinder-link."""
    name: str
    volume: float               # displaced volume [m^3]
    buoyancy_force_mag: float   # rho * g * V  [N]

    # Drag areas & coefficients
    Cd_transverse: float        # drag coeff for crossflow (~1.0 for cylinder)
    Cd_axial: float             # drag coeff for axial flow (~0.8 for blunt end)
    A_transverse: float         # projected area perpendicular to axis [m^2]
    A_axial: float              # end-cap area [m^2]

    # Added mass (diagonal in body frame for a cylinder)
    ma_transverse: float        # [kg]
    ma_axial: float             # [kg]

    cylinder: CylinderPrimitive


# ---------------------------------------------------------------------------
# Hydrodynamic model
# ---------------------------------------------------------------------------

class HydrodynamicModel:
    """Simplified hydrodynamic model for the full robot.

    **Important:** call ``robot.forward_kinematics(q)`` and then
    ``robot.build_cylinders()`` before constructing this model, so that
    each link has a valid skeleton-aligned cylinder.

    Parameters
    ----------
    robot : QuadrupedRobot
        Pinocchio-backed robot with cylinder-approximated links.
    rho : float
        Fluid density [kg/m^3].
    Cd_transverse, Cd_axial : float
        Drag coefficients for crossflow / axial flow.
    Ca_transverse, Ca_axial : float
        Added-mass coefficients (potential-flow value for a circular
        cylinder is Ca_transverse = 1.0).
    """

    def __init__(
        self,
        robot: QuadrupedRobot,
        rho: float = RHO_WATER,
        Cd_transverse: float = 1.0,
        Cd_axial: float = 0.8,
        Ca_transverse: float = 1.0,
        Ca_axial: float = 0.1, #for infinite cylinder Ca_axial = 0
    ):
        self.robot = robot
        self.rho = rho
        self.Cd_transverse = Cd_transverse
        self.Cd_axial = Cd_axial
        self.Ca_transverse = Ca_transverse
        self.Ca_axial = Ca_axial

        self.link_hydro: dict[str, LinkHydroProperties] = {}
        self._build(rho, Cd_transverse, Cd_axial, Ca_transverse, Ca_axial)

    def _build(self, rho, Cd_t, Cd_a, Ca_t, Ca_a):
        for name, link in self.robot.links.items():
            cyl = link.cylinder
            if cyl is None:
                continue
            self.link_hydro[name] = LinkHydroProperties(
                name=name,
                volume=cyl.volume,
                buoyancy_force_mag=rho * GRAVITY * cyl.volume,
                Cd_transverse=Cd_t,
                Cd_axial=Cd_a,
                A_transverse=cyl.cross_section_transverse,
                A_axial=cyl.cross_section_axial,
                ma_transverse=Ca_t * rho * cyl.volume,
                ma_axial=Ca_a * rho * cyl.volume,
                cylinder=cyl,
            )

    # ------------------------------------------------------------------
    # Buoyancy
    # ------------------------------------------------------------------

    def buoyancy_wrench(
        self, link_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Buoyancy force and torque about the world origin for one link."""
        hp = self.link_hydro[link_name]
        force = np.array([0.0, 0.0, hp.buoyancy_force_mag])
        # Center of buoyancy = cylinder center (world frame)
        torque = np.cross(hp.cylinder.center, force)
        return force, torque

    def total_buoyancy(self) -> tuple[np.ndarray, np.ndarray]:
        """Sum buoyancy force and torque over all links."""
        F_total = np.zeros(3)
        T_total = np.zeros(3)
        for name in self.link_hydro:
            f, t = self.buoyancy_wrench(name)
            F_total += f
            T_total += t
        return F_total, T_total

    # ------------------------------------------------------------------
    # Viscous drag  (Morison-equation style)
    # ------------------------------------------------------------------

    def drag_force(
        self,
        link_name: str,
        v_link: np.ndarray,
    ) -> np.ndarray:
        """Quadratic viscous drag on one link in world frame.

        F_drag = -0.5 * rho * Cd * A * |v| * v

        Decomposed into axial and transverse components relative to the
        cylinder axis, each with its own Cd and reference area.
        """
        hp = self.link_hydro[link_name]
        axis = hp.cylinder.axis_world

        # Decompose velocity into axial and transverse components
        v_axial_mag = np.dot(v_link, axis)
        v_axial = v_axial_mag * axis
        v_trans = v_link - v_axial

        F_axial = (
            -0.5 * self.rho * hp.Cd_axial * hp.A_axial
            * abs(v_axial_mag) * v_axial
        )
        v_trans_mag = np.linalg.norm(v_trans)
        F_trans = (
            -0.5 * self.rho * hp.Cd_transverse * hp.A_transverse
            * v_trans_mag * v_trans
        )
        return F_axial + F_trans

    def total_drag(
        self,
        velocities: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Sum drag forces over all links."""
        F = np.zeros(3)
        for name in self.link_hydro:
            if name in velocities:
                F += self.drag_force(name, velocities[name])
        return F

    # ------------------------------------------------------------------
    # Pressure-gradient (Froude-Krylov) force
    # ------------------------------------------------------------------

    def pressure_gradient_force(
        self, link_name: str, fluid_accel: np.ndarray,
    ) -> np.ndarray:
        """Force due to the ambient pressure gradient: F_A = rho * V * a_fluid."""
        hp = self.link_hydro[link_name]
        return self.rho * hp.volume * fluid_accel

    # ------------------------------------------------------------------
    # Added mass
    # ------------------------------------------------------------------

    def added_mass_matrix_link(self, link_name: str) -> np.ndarray:
        """6x6 added-mass matrix for one link in the world frame.

        Translational block only (rotational added inertia neglected).
        """
        hp = self.link_hydro[link_name]
        axis = hp.cylinder.axis_world
        a = axis.reshape(3, 1)

        # M = ma_trans * I + (ma_axial - ma_trans) * (a ⊗ a)
        M_trans = (
            hp.ma_transverse * np.eye(3)
            + (hp.ma_axial - hp.ma_transverse) * (a @ a.T)
        )
        M_A = np.zeros((6, 6))
        M_A[:3, :3] = M_trans
        return M_A

    def total_added_mass_matrix(self) -> np.ndarray:
        """Sum added-mass matrices over all links (world frame)."""
        M_total = np.zeros((6, 6))
        for name in self.link_hydro:
            M_total += self.added_mass_matrix_link(name)
        return M_total

    def added_mass_force(
        self, link_name: str, a_link: np.ndarray,
    ) -> np.ndarray:
        """Force due to added mass for one link: F = -M_A_trans @ a."""
        M_A = self.added_mass_matrix_link(link_name)
        return -M_A[:3, :3] @ a_link

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print a table of hydrodynamic properties per link."""
        total_vol = 0.0
        total_buoy = 0.0
        print(f"{'Link':<30s} {'Vol [cm³]':>10s} {'Buoy [N]':>10s} "
              f"{'r [mm]':>8s} {'L [mm]':>8s} "
              f"{'ma_t [g]':>9s} {'ma_a [g]':>9s}")
        print("-" * 95)
        for name, hp in self.link_hydro.items():
            total_vol += hp.volume
            total_buoy += hp.buoyancy_force_mag
            print(
                f"{name:<30s} "
                f"{hp.volume * 1e6:10.2f} "
                f"{hp.buoyancy_force_mag:10.4f} "
                f"{hp.cylinder.radius * 1e3:8.2f} "
                f"{hp.cylinder.length * 1e3:8.2f} "
                f"{hp.ma_transverse * 1e3:9.3f} "
                f"{hp.ma_axial * 1e3:9.3f}"
            )
        print("-" * 95)
        total_mass = self.robot.total_mass()
        weight = total_mass * GRAVITY
        print(f"{'TOTAL':<30s} {total_vol * 1e6:10.2f} {total_buoy:10.4f}")
        print(f"\nRobot mass:   {total_mass:.4f} kg")
        print(f"Robot weight: {weight:.4f} N")
        print(f"Net vertical: {total_buoy - weight:+.4f} N  "
              f"({'positively buoyant' if total_buoy > weight else 'negatively buoyant'})")
