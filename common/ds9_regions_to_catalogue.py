#!/usr/bin/env python3
"""
ds9_regions_to_catalogue.py
Convert a DS9 region file (ellipses/circles) into a bubble catalogue with the
same columns as jwst_bubble_properties_A.txt.

Output columns
    ID, RA_DMS, DEC_DMS, SEMI_MAJ_PC, SEMI_MIN_PC, AVG_RAD_PC,
    PA_DEG, ARM, DIST_ARM_PC, GAL_RAD_KPC
ARM and DIST_ARM_PC are left blank -- they require a spiral-arm model, which
is not something region geometry can supply (see --arms).
"""

import argparse
import csv
import math
import re
import sys

# ----------------------------------------------------------------------
# GALAXY PARAMETERS
# ----------------------------------------------------------------------
DISTANCE_MPC = 9.84       # adopted distance for ngc628
CENTRE_RA    = 24.173855  # deg
CENTRE_DEC   = 15.783643  # deg
# Inclination 0 = no deprojection, i.e. GAL_RAD is the sky-plane distance from
# the centre. This is what reproduces file A (NGC 628 is nearly face-on).
# For an inclined galaxy set --incl / --disc-pa to deproject properly.
INCL_DEG     = 0.0        # disc inclination (NGC 628: 8.9)
DISC_PA_DEG  = 20.7       # PA of receding major axis, degrees E of N

# DS9 writes an ellipse's rotation angle relative to the x-axis of the
# coordinate system. Set to 90 to convert it to an astronomical position
# angle (E of N); leave 0 to keep DS9's raw angle. Verify on one visibly
# elongated region before trusting a whole catalogue.
PA_OFFSET_DEG = 0.0
# ----------------------------------------------------------------------

ARCSEC = math.pi / (180.0 * 3600.0)
PC_PER_ARCSEC = DISTANCE_MPC * 1e6 * ARCSEC

SHAPE_RE = re.compile(r"^\s*[-+]?\s*(ellipse|circle)\(([^)]*)\)", re.I)
COORD_SYS = {"fk5", "icrs", "j2000", "fk4", "b1950", "galactic", "ecliptic",
             "image", "physical", "detector", "amplifier", "wcs"}


# ---------------------------------------------------------------- parsing
def size_to_arcsec(tok):
    """DS9 size token -> arcsec.  Handles 3.2\" , 0.05' , 0.001d , 0.001 ."""
    tok = tok.strip()
    if tok.endswith('"'):
        return float(tok[:-1])
    if tok.endswith("'"):
        return float(tok[:-1]) * 60.0
    if tok.lower().endswith("d"):
        return float(tok[:-1]) * 3600.0
    return float(tok) * 3600.0          # bare number in a WCS file = degrees


def sex_to_deg(tok, is_ra):
    """Parse '24:09:12.3', '+15d47m58.4s' or a plain decimal degree string."""
    tok = tok.strip()
    if ":" in tok:
        parts = tok.split(":")
        sign = -1.0 if parts[0].strip().startswith("-") else 1.0
        vals = [abs(float(parts[0]))] + [float(p) for p in parts[1:]]
        while len(vals) < 3:
            vals.append(0.0)
        val = sign * (vals[0] + vals[1] / 60.0 + vals[2] / 3600.0)
        return val * 15.0 if is_ra else val       # sexagesimal RA is hours
    m = re.match(r"^([-+]?[\d.]+)[dh]([\d.]+)m([\d.]+)s?$", tok, re.I)
    if m:
        sign = -1.0 if tok.strip().startswith("-") else 1.0
        val = (abs(float(m.group(1))) + float(m.group(2)) / 60.0
               + float(m.group(3)) / 3600.0)
        val *= sign
        return val * 15.0 if tok.lower().startswith(("+h", "h")) else val
    return float(tok)


def parse_regions(path):
    frame = "fk5"
    regs = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            bare = s.split("#")[0].strip().lower()
            if bare in COORD_SYS:
                frame = bare
                continue
            m = SHAPE_RE.match(s)
            if not m:
                continue
            kind = m.group(1).lower()
            p = [t.strip() for t in m.group(2).split(",")]
            ra = sex_to_deg(p[0], is_ra=True)
            dec = sex_to_deg(p[1], is_ra=False)
            if kind == "circle":
                a = b = size_to_arcsec(p[2])
                ang = 0.0
            else:
                a = size_to_arcsec(p[2])
                b = size_to_arcsec(p[3])
                ang = float(p[4]) if len(p) > 4 else 0.0
            regs.append([ra, dec, a, b, ang, frame])
    return regs


# ------------------------------------------------------------ formatting
def deg_to_dms(value, dec_places=1):
    """23.456 -> '23d27m21.6s' (degrees:arcmin:arcsec, as in file A)."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    d = int(v)
    rem = (v - d) * 60.0
    m = int(rem)
    s = round((rem - m) * 60.0, dec_places)
    if s >= 60.0:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    return f"{sign}{d}d{m:02d}m{s:.{dec_places}f}s"


# ----------------------------------------------------------------- sphere
def separation_arcsec(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dr = r2 - r1
    num = math.hypot(math.cos(d2) * math.sin(dr),
                     math.cos(d1) * math.sin(d2)
                     - math.sin(d1) * math.cos(d2) * math.cos(dr))
    den = (math.sin(d1) * math.sin(d2)
           + math.cos(d1) * math.cos(d2) * math.cos(dr))
    return math.degrees(math.atan2(num, den)) * 3600.0


def position_angle_deg(ra1, dec1, ra2, dec2):
    """PA of point 2 as seen from point 1, degrees east of north."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dr = r2 - r1
    y = math.sin(dr) * math.cos(d2)
    x = math.cos(d1) * math.sin(d2) - math.sin(d1) * math.cos(d2) * math.cos(dr)
    return math.degrees(math.atan2(y, x)) % 360.0


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("regfile", help="DS9 region file saved in fk5")
    ap.add_argument("-o", "--output", default="bubble_catalogue.csv")
    ap.add_argument("--distance", type=float, default=DISTANCE_MPC,
                    help="distance in Mpc (default %(default)s)")
    ap.add_argument("--centre", nargs=2, type=float,
                    default=[CENTRE_RA, CENTRE_DEC],
                    metavar=("RA_DEG", "DEC_DEG"))
    ap.add_argument("--incl", type=float, default=INCL_DEG)
    ap.add_argument("--disc-pa", type=float, default=DISC_PA_DEG)
    ap.add_argument("--pa-offset", type=float, default=PA_OFFSET_DEG,
                    help="add to DS9 angle; use 90 for PA east of north")
    args = ap.parse_args()

    pc_per_as = args.distance * 1e6 * ARCSEC

    regs = parse_regions(args.regfile)
    if not regs:
        sys.exit("No circle/ellipse regions found. Re-save from DS9 in 'ds9' "
                 "format with Coordinate System: fk5.")
    if any(r[5] in ("image", "physical", "detector", "amplifier") for r in regs):
        sys.exit("Regions are in pixel coordinates. Re-save from DS9 with "
                 "Coordinate System: fk5 so sky positions are written out.")

    rows = []
    for ra, dec, a_as, b_as, ang, _ in regs:
        if b_as > a_as:                      # enforce major >= minor
            a_as, b_as = b_as, a_as
            ang += 90.0
        semi_maj = a_as * pc_per_as
        semi_min = b_as * pc_per_as
        avg_rad = (2.0 * semi_maj + semi_min) / 3.0     # convention in file A
        # ellipse orientation is only defined mod 180 deg; wrap to [-90, 90)
        pa = (ang + args.pa_offset + 90.0) % 180.0 - 90.0

        sep = separation_arcsec(args.centre[0], args.centre[1], ra, dec)
        theta = math.radians(
            position_angle_deg(args.centre[0], args.centre[1], ra, dec)
            - args.disc_pa)
        x = sep * math.sin(theta)                                  # minor axis
        y = sep * math.cos(theta) / math.cos(math.radians(args.incl))
        gal_rad_kpc = math.hypot(x, y) * pc_per_as / 1000.0

        rows.append([ra, deg_to_dms(ra), deg_to_dms(dec),
                     round(semi_maj), round(semi_min), round(avg_rad),
                     round(pa), "", "", f"{gal_rad_kpc:.2f}"])

    rows.sort(key=lambda r: r[0])            # order by RA, as in file A

    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "RA_DMS", "DEC_DMS", "SEMI_MAJ_PC", "SEMI_MIN_PC",
                    "AVG_RAD_PC", "PA_DEG", "ARM", "DIST_ARM_PC",
                    "GAL_RAD_KPC"])
        for i, r in enumerate(rows, 1):
            w.writerow([i] + r[1:])

    print(f"{len(rows)} regions -> {args.output}")
    print(f"scale: 1 arcsec = {pc_per_as:.2f} pc at {args.distance} Mpc")
    print("ARM / DIST_ARM_PC left blank: they need a spiral-arm model.")


if __name__ == "__main__":
    main()
