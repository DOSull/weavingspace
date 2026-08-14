"""MIT License.

Copyright (c) 2021-26 David O'Sullivan & Luke Bergmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from weavingspace import TileUnit
import itertools
import math
import string

import geopandas as gpd
import numpy as np
import shapely.affinity as affine
import shapely.geometry as geom
from shapely import line_interpolate_point

from weavingspace import TileShape, tiling_utils

"""Functions for setting up a TileUnit with various tile geometries.

Some care is required in adding new functions that use existing ones to get the
sequence of setup operations right. Modify with care!

Many of the available tilings can be viewed at

https://dosull.github.io/mapweaver/app/

These tilings (and many many more!) are discussed in

Grunbaum B, Shephard G C, 1987 _Tilings and Patterns_ (W. H. Freeman and
Company, New York)

A more accessible introduction is

Kaplan C S, 2002 _Computer Graphics and Geometric Ornamental Design_, PhD
thesis, University of Washington, Seattle, WA,
https://cs.uwaterloo.ca/~csk/other/phd/kaplan_diss_full_print.pdf

and a more 'polished' version of that work focused on computer graphics is also
available

Kaplan C S, 2009 _Introductory Tiling Theory for Computer Graphics_ (Morgan &
Claypool)
"""

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

def _setup_base_tiling(unit:TileUnit) -> str|None:
  """Set up hexagonal, square, diamond, or triangular tileable unit.

  Args:
      unit (TileUnit): contains the specification requested by calling context.

  Returns:
      str|None: str if there is a problem, None otherwise.

  """
  unit.spacing = 1000 if unit.spacing is None else unit.spacing
  match unit.base_shape:
    case TileShape.RECTANGLE:
      polys = [tiling_utils.get_regular_polygon(unit.spacing, 4)]
      unit.setup_vectors((0, unit.spacing), (unit.spacing, 0))
    case TileShape.TRIANGLE:
      tri1 = tiling_utils.get_regular_polygon(unit.spacing, 3)
      tri1 = affine.translate(tri1, 0, -tri1.bounds[1])
      tri2 = affine.rotate(tri1, 180, (0, 0))
      polys = [tri1, tri2]
      unit.setup_vectors((unit.spacing / sqrt3,  unit.spacing),
                         (unit.spacing / sqrt3, -unit.spacing))
    case TileShape.DIAMOND:
      half_tile = tiling_utils.get_regular_polygon(unit.spacing, 3)
      half_tile = affine.translate(half_tile, 0, -half_tile.bounds[1])
      corners = tiling_utils.get_corners(half_tile, repeat_first = False)
      coords = [(p.x, p.y) for p in corners]
      polys = [geom.Polygon([coords[0], coords[1], coords[2],
                             (coords[1][0], -coords[1][1])])]
      unit.setup_vectors((unit.spacing / sqrt3,  unit.spacing),
                         (unit.spacing / sqrt3, -unit.spacing))
    case TileShape.HEXAGON:
      polys = [tiling_utils.get_regular_polygon(unit.spacing, 6)]
      unit.setup_vectors((0,  unit.spacing),
                         (unit.spacing * sqrt3 / 2,  unit.spacing / 2),
                         (unit.spacing * sqrt3 / 2, -unit.spacing / 2))
    case _:
      return "No proper TileShape provided to base tile unit constructor."
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("ab")[:len(polys)]},
    geometry = gpd.GeoSeries(polys),
    crs = 3857 if unit.crs is None else unit.crs)
  return None


def _setup_cairo(unit:TileUnit) -> str|None:
  """Set up the Cairo tiling. King of tilings. All hail the Cairo tiling.

  The pattern in this function shows how any new setup function should be
  organised:

    + Make the geometries and put them in a GeoDataFrame with the
      appropriate number of tile_id codes
    + Set the translation vectors
    + Return None (THIS IS IMPORTANT)

  This example shows this for a 'handcoded' set of geometries can be applied.

  Note that it is advisable to avoid intersection and union operations where
  possible, as it often yields floating point mismatches that can be hard to
  repair! (This tiling could be relatively conveniently generated by dissecting
  a square in 4 quarters at a 30 degree angle to the sides and then reflecting
  and rotating copies and joing them back together. But floating point issues
  make that very messy indeed. Much better to make the geometries 'pure'.)

  To design new tilings, you have to get out pencil and paper and do the
  trigonometry.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  # MAKE GEOMETRIES
  d = unit.spacing
  x = d/2 / (np.cos(np.pi/12) + np.cos(5*np.pi/12))
  # the following is just the geometry, it is what it is...
  # points are (more or less)
  #
  #         3
  #     2
  #            4
  #     1    0
  #
  # then rotate -15 and make 4 copies at 90 degree rotations
  p1 = geom.Polygon([(x, 0), (0, 0), (0, x),
                     (x * sqrt3/2, x + x/2),
                     (x * (1 + sqrt3) / 2, x * (3 - sqrt3) / 2)])
  p1 = affine.rotate(p1, -15, (0, 0))
  p2 = affine.rotate(p1,  90, (0, 0))
  p3 = affine.rotate(p1, 180, (0, 0))
  p4 = affine.rotate(p1, 270, (0, 0))
  # now move so they are arranged as a hexagon centered on the tile
  p1 = affine.translate(p1, -unit.spacing / 2, 0)
  p2 = affine.translate(p2,  unit.spacing / 2, 0)
  p3 = affine.translate(p3,  unit.spacing / 2, 0)
  p4 = affine.translate(p4, -unit.spacing / 2, 0)
  polys = [affine.rotate(p, 45, origin = (0, 0)) for p in [p1, p2, p3, p4]]
  # MAKE THE GEODATAFRAME
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcd")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))
  # SET THE TRANSLATION VECTORS
  unit.setup_vectors((unit.spacing / sqrt2,  unit.spacing/sqrt2),
                     (unit.spacing / sqrt2, -unit.spacing/sqrt2))
  return None


def _setup_hex_slice(unit:TileUnit) -> str|None:
  """Set up tile unit with number of radial slices of a hexagon.

  Slices will start at a corner, but may be offset from there by an optional
  offset starting point.

  The supplied unit should have offset and n set.

  Places unit.n points equally spaced around the perimeter of the hexagon.
  Slices are formed by connecting some sequence of these points (and any
  intervening hexagon corners) and the hexagon centre to form a pie slice.
  unit.offset = 0 starts from corner 0. unit.offset = 1 starts half way between
  corner 0 and where corner 1 would be if offset were 0. Other values of
  unit.offset scale linearly between these. Negative offsets similarly move the
  first corner towards the corner before corner 0 in the offset = 0 case. 

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["n", "offset"]):
    return ("""Hex slice tiling requires both n and offset values
            to be supplied.""")
  hexagon = tiling_utils.get_regular_polygon(unit.spacing, 6)
  slices = _get_radially_sliced_polygon(hexagon, unit.n, unit.offset)
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(slices))
  unit.setup_vectors((                       0,  unit.spacing    ),
                     (unit.spacing * sqrt3 / 2,  unit.spacing / 2),
                     (unit.spacing * sqrt3 / 2, -unit.spacing / 2))
  return None

def _setup_square_slice(unit:TileUnit) -> str|None:
  """Set up tile unit with number of radial slices of a square.

  The supplied unit should have offset and n set.

  Places unit.n points equally spaced around the perimeter of the square.
  Slices are formed by connecting some sequence of these points (and any
  intervening squares corners) and the square centre to form a pie slice.
  unit.offset = 0 starts from corner 0. unit.offset starts half way between
  corner 0 and where corner 1 would be if offset were 0. Other values of
  unit.offset scale linearly between these. Negative offsets similarly move the
  first corner towards the corner before corner 0 in the offset = 0 case.

  Args:
    unit (TileUnit): the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["n", "offset"]):
    return ("""Square slice tiling requires both n and offset values
            to be supplied.""")
  square = tiling_utils.get_regular_polygon(unit.spacing, 4)
  slices = _get_radially_sliced_polygon(square, unit.n, unit.offset)
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(slices))
  unit.setup_vectors((0, unit.spacing), (unit.spacing, 0))
  return None

def _get_radially_sliced_polygon(
    shape:geom.Polygon,
    n_slices:int,
    offset:float,
  ) -> gpd.GeoSeries:
  """Return a number of 'pie slices' of the supplied polygon as a GeoSeries.

  Args:
    shape (geom.Polygon): the polygon to slice up.
    n_slices (int): the desired number of slices.
    offset (float): a linearly scaled positional offset for the starting point
      point of the slices. 0 is at the first vertex. 1 is halfway between that
      and the next vertex. -1 is halfway between 0 and the previous vertex.

  Returns:
      gpd.GeoSeries: a set of slice polygons.

  """
  n_sides = len(shape.exterior.coords) - 1
  if abs(offset) > 1:
    offset = offset % 1
  offset = (offset * n_sides / n_slices / 2
            if n_slices >= n_sides
            else offset / 2)
  # the values in along are denominated in 'side counts' i.e. they run from
  # 0 to the number of sides, with e.g., 2.5 halfway along the 3rd side
  along = [n_sides * i / n_slices + offset for i in range(n_slices)]
  along = [*along, n_sides + along[0]]
  # the cut points
  boundary = geom.LineString(shape.exterior.coords)
  points = [line_interpolate_point(
    boundary, (a % n_sides) / n_sides, normalized = True) for a in along]
  # the corners of the original polygon
  corners = tiling_utils.get_corners(shape)
  slices = []
  # now we iterate over the list of values in along and in points in parallel
  # checking if any cut point has 'skipped' a corner when the integer part
  # of the values from the along list will be different
  for a1, a2, p1, p2 in zip(along[:-1],  along[1:],
                           points[:-1], points[1:], strict = True):
    if math.floor(a1) == math.floor(a2):
      # two cut points on same polygon edge, so make the slice
      slices.append(geom.Polygon([p1, p2, (0, 0)]))
    else:
      # there are one or more corners between the cut points so get them
      skipped_corners = corners[math.ceil(a1):math.ceil(a2)]
      # but discard any that are close to the cut points to avoid duplicates
      skipped_corners = [
        c for c in skipped_corners
        if not (np.isclose(c.distance(p1), 0, rtol = 1e-6, atol = 1e-6)
             or np.isclose(c.distance(p2), 0, rtol = 1e-6, atol = 1e-6))]
      slices.append(geom.Polygon([p1, *skipped_corners, p2, (0, 0)]))
  return gpd.GeoSeries(slices)


def _setup_stripes(unit:TileUnit) -> str|None:
  if "n" not in unit.__dict__:
    return ("""Stripes tiling requires n to be supplied.""")
  ymin, ymax = -unit.spacing / 2, unit.spacing / 2
  xs = [unit.spacing * x for x in np.linspace(-.5, .5, unit.n + 1)]
  stripes = [geom.Polygon([(xmin, ymin), (xmin, ymax), 
                           (xmax, ymax), (xmax, ymin)])
             for xmin, xmax in itertools.pairwise(xs)]
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(stripes))
  unit.setup_vectors((unit.spacing, 0), (0, unit.spacing))
  return None


def _setup_crosses(unit:TileUnit) -> str|None:
  """Tilings by varying numbers of crosses.

  The supplied unit should have n set. The specific arrangements used are
  more or less arbitrary, worked out on inkscape and/or with pencil and paper.
  There is probably an algorithmic way to relate the arrangement of crosses
  and the tiling translation vectors, but this is fine for now.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["n"]):
    return ("""Crosses tiling requires n to be supplied.""")
  x = unit.spacing / np.sqrt(unit.n * 5)
  cross = geom.Polygon([
      (-x/2, -x/2), (-3*x/2,   -x/2), (-3*x/2,    x/2),
      (-x/2,  x/2), (  -x/2,  3*x/2), (   x/2,  3*x/2),
      ( x/2,  x/2), ( 3*x/2,    x/2), ( 3*x/2,   -x/2),
      ( x/2, -x/2), (   x/2, -3*x/2), (  -x/2, -3*x/2)])
  match unit.n:
    case 2:
      trans = [(-x/2, -x), (x/2, x)]
      parts = [affine.translate(cross, dx, dy) for dx, dy in trans]
      unit.setup_vectors((-3*x, -x), (-x, 3*x))
    case 3:
      trans = [(-x, -2*x), (0, 0), (x, 2*x)]
      parts = [affine.translate(cross, dx, dy) for dx, dy in trans]
      unit.setup_vectors((-x, 3*x), (4*x, 3*x))
    case 4:
      trans = [(3*x/2, x/2), (x/2, -3*x/2), (-3*x/2, -x/2), (-x/2, 3*x/2)]
      parts = [affine.translate(cross, dx, dy) for dx, dy in trans]
      unit.setup_vectors((-4*x, -3*x), (-4*x,  2*x), (0,  5*x))
    case 5:
      trans = [(0, 0), (x, 2*x), (2*x, -x), (-x, -2*x), (-2*x, x)]
      parts = [affine.translate(cross, dx, dy) for dx, dy in trans]
      unit.setup_vectors((-5*x, 0), (0,  5*x))
    case 6:
      trans = [(-2*x, -3*x/2), (-x,  x/2), (0,  5*x/2),
               ( 2*x,  3*x/2), ( x, -x/2), (0, -5*x/2)]
      parts = [affine.translate(cross, dx, dy) for dx, dy in trans]
      unit.setup_vectors((-3*x, -6*x), (-5*x, 0), (-2*x,  6*x))
    case 7:
      trans = [(0, 0), (-x, -2*x), (-2*x,  x), (-x,  3*x),
                       ( x,  2*x), ( 2*x, -x), ( x, -3*x)]
      parts = [affine.translate(cross, dx, dy) for dx, dy in trans]
      unit.setup_vectors((-4*x, -3*x), (-5*x,  5*x), (-x,  8*x))
    case _:
      return (f"""a {unit.n} crosses tiling is not implemented.
              Try a number between 2 and 7.""")
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(parts))
  return None


def _setup_laves(unit:TileUnit) -> str|None:
  """Set up Laves tilings.

  See:
  https://en.wikipedia.org/wiki/List_of_Euclidean_uniform_tilings#Laves_tilings.

  These are all isohedral, but mostly not regular polygons. We prioritise them
  over the Archimedean tilings because being isohedral all tiles are the same
  size and shape which makes code to generate them a lot easier to write.
  Several are hex dissections and setup is delegated accordingly.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["code"]):
    return ("""Laves tiling requires vertex degree code to be supplied.""")
  match unit.code:
    case "3.3.3.3.3.3":
      # this is the regular hexagons
      unit.base_shape = TileShape.HEXAGON
      return ("""Laves 3.3.3.3.3.3 is a basic hexagonal tiling. Perhaps you
              want a hex-colouring, hex-slice or hex-dissection tiling?""")
    case "3.3.3.3.6":
      # this one needs its own code
      _setup_laves_3_3_3_3_6(unit)
    case "3.3.3.4.4":
      # this is a boring tiling...
      _setup_laves_3_3_3_4_4(unit)
    case "3.3.4.3.4":
      # king of tilings!
      _setup_cairo(unit)
    case "3.4.6.4":
      # the hex 6-slice
      unit.n = 6
      unit.offset = 1
      _setup_hex_slice(unit)
    case "3.6.3.6":
      # hex 3-slice
      unit.n = 3
      unit.offset = 0
      _setup_hex_slice(unit)
    case "3.12.12":
      # again this one needs its own
      _setup_laves_3_12_12(unit)
    case "4.4.4.4":
      # square grid
      unit.base_shape = TileShape.RECTANGLE
      return ("""Laves 4.4.4.4 is a basic square tiling. Perhaps you want a
              square-colouring or square-slice tiling?""")
    case "4.6.12":
      # hex 12-slice
      unit.n = 12
      unit.offset = 0
      _setup_hex_slice(unit)
    case "4.8.8":
      # square 4-slice
      unit.n = 4
      unit.offset = 0.0
      _setup_square_slice(unit)
    case "6.6.6":
      # triangles
      unit.base_shape = TileShape.TRIANGLE
      return """Note that Laves 3.3.3.3.3.3 is a basic triangle tiling."""
    case _:
      return  "[{unit.code}] is not a valid Laves code."
  return None


def _setup_laves_3_3_3_3_6(unit:TileUnit) -> None:
  """Set up Laves [3.3.3.3.6] which is like a 6 petal flower.

  Similar to the H3 7 hexagon group with the central hex removed and each
  hex 'taking' a 1/6 share of the central hex.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  sf = 1 / np.sqrt(7)
  tile = tiling_utils.get_regular_polygon(unit.spacing, 6)
  hexagon = affine.scale(tile, sf, sf)
  # translate it up by its own height
  hexagon = affine.translate(hexagon, 0, hexagon.bounds[3] - hexagon.bounds[1])
  hex_p = list(hexagon.exterior.coords)
  # now replace first and last points by (0, 0)
  # (note shapely doesn't require closure of the polygon)
  petal = geom.Polygon([(0, 0)] + hex_p[1:5])
  offset_a = np.degrees(np.arctan(1 / 3 / sqrt3))
  petals = [
    affine.rotate(petal, a + offset_a, origin = (0, 0))
    for a in range(30, 360, 60)]
  unit.setup_vectors((                     0,  unit.spacing    ),
                     (unit.spacing * sqrt3/2,  unit.spacing / 2),
                     (unit.spacing * sqrt3/2, -unit.spacing / 2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")},
    crs = unit.crs,
    geometry = gpd.GeoSeries(petals))


def _setup_laves_3_3_3_4_4(unit:TileUnit) -> None:
  """Set up Laves [3.3.3.4.4] which is like two houses back to back.

  Similar to the H3 7 hexagon group with the central hex removed and each
  hex 'taking' a 1/6 share of the central hex.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  x = unit.spacing / np.sqrt(2)
  tile1 = geom.Polygon([
    ( x/2,  0),
    ( x/2,  x * (0.5 + sqrt3/6)),
    (   0,  x * (0.5 + sqrt3/3)),
    (-x/2,  x * (0.5 + sqrt3/6)),
    (-x/2,  0),
  ])
  tile2 = geom.Polygon([
    (-x/2,  0),
    (-x/2, -x * (0.5 + sqrt3/6)),
    (   0, -x * (0.5 + sqrt3/3)),
    ( x/2, -x * (0.5 + sqrt3/6)),
    ( x/2,  0),
  ])
  unit.setup_vectors((x, 0), (x / 2, x * (1 + sqrt3 / 2)))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("ab")},
    crs = unit.crs,
    geometry = gpd.GeoSeries([tile1, tile2]))


def _setup_laves_3_12_12(unit:TileUnit) -> None:
  """Set up Laves [3.12.12] - like [3.6.3.6] with rhombuses sliced lengthwise.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  hexagon = tiling_utils.get_regular_polygon(unit.spacing, 6)
  pts = list(hexagon.exterior.coords)
  tri1 = geom.Polygon([pts[0], pts[2], [0, 0]])
  tri2 = geom.Polygon([pts[0], pts[1], pts[2]])
  tris1 = [affine.rotate(tri1, a, origin = (0, 0))
       for a in range(0, 360, 120)]
  tris2 = [affine.rotate(tri2, a, origin = (0, 0))
       for a in range(0, 360, 120)]
  # reorder so the 'inner' and 'outer' triangles are labelled alternately
  tris = itertools.chain(*zip(tris1, tris2, strict = True))
  unit.setup_vectors((                     0,  unit.spacing    ),
                     (unit.spacing * sqrt3/2,  unit.spacing / 2),
                     (unit.spacing * sqrt3/2, -unit.spacing / 2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")}, crs = unit.crs,
    geometry = gpd.GeoSeries(tris))


def _setup_archimedean(unit:TileUnit) -> str|None:
  r"""Set up Archimedean 'regular' tilings.

  See:
  https://en.wikipedia.org/wiki/Euclidean_tilings_by_convex_regular_polygons

  Many of these could be constructed as duals of the Laves tilings. However,
  inaccuracies in the `tiling_utils.get_dual_tiling()` function yield tiles
  that don't sustain reliable Topology detection... ¯\_(ツ)_/¯

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["code"]):
    return "Archimedean tiling requires vertex degree code to be supplied."
  match unit.code:
    case "3.3.3.3.3.3":
      unit.base_shape = TileShape.TRIANGLE
      return "Note that Archimedean 3.3.3.3.3.3 is a basic triangle tiling?"
    case "3.3.3.3.6":
      _setup_archimedean_3_3_3_3_6(unit)
    case "3.3.3.4.4":
      _setup_archimedean_3_3_3_4_4(unit)
    case "3.3.4.3.4":
      _setup_archimedean_3_3_4_3_4(unit)
    case "3.4.6.4":
      _setup_archimedean_3_4_6_4(unit)
    case "3.6.3.6":
      _setup_archimedean_3_6_3_6(unit)
    case "3.12.12":
      _setup_archimedean_3_12_12(unit)
    case "4.4.4.4":
      unit.base_shape = TileShape.RECTANGLE
      return ("""Archimedean 4.4.4.4 is a basic square tiling. Perhaps you
              want a square-colouring or square-slice tiling?""")
    case "4.6.12":
      _setup_archimedean_4_6_12(unit)
    case "4.8.8":
      _setup_archimedean_4_8_8(unit)
    case "6.6.6":
      unit.base_shape = TileShape.HEXAGON
      return ("""Archimedean 6.6.6 is a basic hexagonal tiling. Perhaps you
              want a hex-colouring, hex-slice or hex-dissection tiling?""")
    case _:
      unit.base_shape = TileShape.RECTANGLE
      return  f"({unit.code}) is not a valid Archimedean code."
  return None


# Note that where possible in the functions that follow the following short
# variable names are used:
#  t for height of triangles
#  s for height of squares
#  h for height of hexagons
#  o for height of octagons
#  d for height of dodecagons
# rescaling so that the resulting tile unit meets the unit.spacing request is
# in most cases exact, but where the tile unit departs from a plain square or
# hexagon it's approximate.
def _setup_archimedean_3_3_3_3_6(unit:TileUnit) -> None:
  """Set up (3.3.3.3.6)."""
  h = unit.spacing * np.sqrt(3/7)
  t = h / 2
  hexagon = tiling_utils.get_regular_polygon(h, 6)
  tri = tiling_utils.get_regular_polygon(t, 3)
  tri_up = affine.translate(tri, 0, -tri.bounds[3])
  tri_dn = affine.rotate(tri_up, 180, origin = (0, 0))
  wdth, hght = tri.bounds[2] - tri.bounds[0], tri.bounds[3] - tri.bounds[1]
  triangles = [affine.translate(tri_up,        0,  2*hght),
               affine.translate(tri_up,   wdth  ,  2*hght),
               affine.translate(tri_dn,   wdth  ,       0),
               affine.translate(tri_up, 3*wdth/2,    hght),
               affine.translate(tri_dn, 3*wdth/2,   -hght),
               affine.translate(tri_up,   wdth  ,       0),
               affine.translate(tri_dn,   wdth  , -2*hght),
               affine.translate(tri_dn,        0, -2*hght)]
  polys = [hexagon, *triangles]
  unit.setup_vectors((-wdth/2, 3*hght), (2*wdth, 2*hght), (5*wdth/2, -hght))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdefghi")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_archimedean_3_3_3_4_4(unit:TileUnit) -> None:
  """Set up (3.3.3.4.4)."""
  s = unit.spacing / np.sqrt(1 + sqrt3)
  t = s * sqrt3/2
  square = tiling_utils.get_regular_polygon(s, 4)
  tri = tiling_utils.get_regular_polygon(t, 3)
  tri_up = affine.translate(tri, 0, -tri.bounds[1] + s/2)
  tri_dn = affine.rotate(tri_up, 180, origin = (0, 0))
  polys = [square, tri_up, tri_dn]
  unit.setup_vectors((s/2, s+t), (s, 0), (s/2, -s-t))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abc")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_archimedean_3_3_4_3_4(unit:TileUnit) -> None:
  """Set up (3.3.4.3.4)."""
  s = unit.spacing/2/np.cos(np.pi/12)
  t = s * sqrt3/2
  square = tiling_utils.get_regular_polygon(s, 4)
  tri = tiling_utils.get_regular_polygon(t, 3)
  tri1 = affine.translate(tri, 0, -tri.bounds[3])
  tri2 = affine.rotate(tri1,  180, origin = (0, -t))
  tri3 = affine.rotate(tri1,  150, origin = (0,  0))
  tri4 = affine.rotate(tri1, -150, origin = (0,  0))
  sq1 = affine.translate(square, s/2, -s/2)
  sq1 = affine.rotate(sq1,   30, origin = (0, 0))
  sq2 = affine.rotate(sq1, -150, origin = (0, 0))
  polys = [tri2, tri1, sq2, tri4, tri3, sq1]
  polys = [affine.translate(p, 0, t - s/2) for p in polys]
  unit.setup_vectors((t + s/2,  t + s/2), (t + s/2, -t - s/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_archimedean_3_4_6_4(unit:TileUnit) -> None:
  """Set up (3.4.6.4)."""
  h = unit.spacing / (1 + sqrt3) * sqrt3
  s = h / sqrt3
  t = h / 2
  hexagon = tiling_utils.get_regular_polygon(h, 6)
  square = tiling_utils.get_regular_polygon(s, 4)
  tri = tiling_utils.get_regular_polygon(t, 3)
  sq1 = affine.translate(square, 0, s / 2 + t)
  sq2 = affine.rotate(sq1,  60, origin = (0, 0))
  sq3 = affine.rotate(sq1, -60, origin = (0, 0))
  tri1 = affine.translate(tri, 0, t - tri.bounds[1])
  tri2 = affine.translate(tri, 0, t - tri.bounds[1])
  tri1 = affine.rotate(tri1, 90, origin = (-s / 2, t))
  tri2 = affine.rotate(tri2, -90, origin = (s / 2, t))
  polys = [hexagon, sq2, tri1, sq1, tri2, sq3]
  unit.setup_vectors((0, h + s), (t + 3*s/2,  t + s/2), (t + 3*s/2, -t - s/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_archimedean_3_6_3_6(unit:TileUnit) -> None:
  """Set up (3.6.3.6)."""
  h = unit.spacing * sqrt3/2
  t = h / 2
  hexagon = tiling_utils.get_regular_polygon(h, 6)
  tri = tiling_utils.get_regular_polygon(t, 3)
  tri1 = affine.translate(tri, 0, t - tri.bounds[1])
  tri2 = affine.rotate(tri, 180)
  tri2 = affine.translate(tri2, 0, -t - tri.bounds[3])
  polys = [hexagon, tri1, tri2]
  unit.setup_vectors((t*2/sqrt3,  h), (t*2/sqrt3, -h))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abc")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_archimedean_3_12_12(unit:TileUnit) -> None:
  """Set up (3.12.12)."""
  d = unit.spacing
  t = d * sqrt3/2 / (2 + sqrt3)
  dodecagon = tiling_utils.get_regular_polygon(d, 12)
  tri = tiling_utils.get_regular_polygon(t, 3)
  tri = affine.translate(tri, 0, unit.spacing / 2 - tri.bounds[1])
  tri_1 = affine.rotate(tri,  30, origin = (0, 0))
  tri_2 = affine.rotate(tri, -30, origin = (0, 0))
  polys = [dodecagon, tri_1, tri_2]
  unit.setup_vectors((0, d), (d * sqrt3/2,  d/2), (d * sqrt3/2, -d/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abc")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_archimedean_4_6_12(unit:TileUnit) -> None:
  """Set up (4.6.12)."""
  d = unit.spacing * (2 + sqrt3) / (3 + sqrt3)
  s = d / (2 + sqrt3)
  h = s * sqrt3
  dodecagon = tiling_utils.get_regular_polygon(d, 12)
  hexagon = tiling_utils.get_regular_polygon(h, 6)
  square = tiling_utils.get_regular_polygon(s, 4)
  sq1 = affine.translate(square, 0, (s + d) / 2)
  sq2 = affine.rotate(sq1,  60, origin = (0, 0))
  sq3 = affine.rotate(sq1, -60, origin = (0, 0))
  hex1 = affine.translate(hexagon, 0, (h + d) / 2)
  hex1 = affine.rotate(hex1,  30, origin = (0, 0))
  hex2 = affine.rotate(hex1, -60, origin = (0, 0))
  polys = [dodecagon, sq2, hex1, sq1, hex2, sq3]
  unit.setup_vectors((                0,   d + s     ),
                     ((d + s) * sqrt3/2,  (d + s) / 2),
                     ((d + s) * sqrt3/2, -(d + s) / 2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_archimedean_4_8_8(unit:TileUnit) -> None:
  """Set up (4.8.8)."""
  o = unit.spacing
  s = o / (1 + sqrt2)
  octagon = tiling_utils.get_regular_polygon(o, 8)
  square = tiling_utils.get_regular_polygon(s, 4)
  square = affine.translate(square, 0, (o + s) / 2)
  square = affine.rotate(square, 45, origin = (0, 0))
  polys = [octagon, square]
  unit.setup_vectors((0, o), (o, 0))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("ab")}, crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_hex_colouring(unit:TileUnit) -> str|None:
  """Set up colourings of a regular array of hexagons.

  The supplied unit must have unit.n set.

  Args:
    unit (TileUnit): the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["n"]):
    return ("""Hex colouring tiling requires n to be supplied.""")
  hexagon = tiling_utils.get_regular_polygon(unit.spacing / np.sqrt(unit.n), 6)
  hexagon = affine.rotate(hexagon, 30, origin = (0, 0))
  bounds = hexagon.bounds
  # most distance offsets in these tilings conveniently expressed in terms of
  # half the width and one quarter the height of point-up hexagons
  w = (bounds[2] - bounds[0]) / 2
  h = (bounds[3] - bounds[1]) / 4
  match unit.n:
    case 2:
      # Pair of point up hexes sideways displaced by width of a hex
      # Vectors are rectangular
      hexes = [affine.translate(hexagon, dx, 0) for dx in [-w, w]]
      unit.setup_vectors((w, 3*h), (4*w, 0))
    case 3:
      #  b
      # c a
      # Copy and translate to alternate corners
      corners = [p for i, p in enumerate(hexagon.exterior.coords)
                 if i in (0, 2, 4)]
      hexes = [affine.translate(hexagon, p[0], p[1]) for p in corners]
      unit.setup_vectors((0, 6*h), (3*w,  3*h), (3*w, -3*h))
    case 4:
      # Point up hex at '*' displaced to 4 positions:
      #  b
      # c a
      #  d
      tr = [(w, 0), (0, 3*h), (-w, 0), (0, -3*h)]
      hexes = [affine.translate(hexagon, v[0], v[1]) for v in tr]
      unit.setup_vectors((2*w, 6*h), (2*w, -6*h))
    case 5:
      # a b c
      #  d e
      tr = [(-2*w, h), (0, h), (2*w, h), (-w, -2*h), (w, -2*h)]
      hexes = [affine.translate(hexagon, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, 6*h), (5*w, 3*h), (5*w, -3*h))
    case 6:
      # a b c
      #  d e f
      tr = [(-5*w/2,  3*h/2), (-w/2,  3*h/2), (3*w/2,  3*h/2),
            (-3*w/2, -3*h/2), ( w/2, -3*h/2), (5*w/2, -3*h/2)]
      hexes = [affine.translate(hexagon, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, 6*h), (6*w, 0))
    case 7:  # the 'H3' tile
      #  a b
      # c d e
      #  f g
      offsets = [(-w,  3*h), (w,  3*h),
          (-2*w, 0),     (0, 0),    (2*w, 0),
                 (-w, -3*h), (w, -3*h) ]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((w, 9*h), (5*w, 3*h), (4*w, -6*h))
    case 8:
      #  a b
      # c d e
      #  f g
      #   h
      offsets = [(-w,  3*h),    ( w,  3*h),
          (-2*w, 0),     (0,    0),     ( 2*w, 0),
                 (-w, -3*h),    ( w, -3*h),
                         (0, -6*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((0, 12*h), (4*w, 6*h), (4*w, -6*h))
    case 9:
      #   a
      #  b c
      # d e f
      #  g h
      #   i
      offsets = [      (0,   6*h),
               (-w,  3*h),    (w,  3*h),
        (-2*w, 0),     (0,    0),    ( 2*w, 0),
               (-w, -3*h),    (w, -3*h),
                       (0, -6*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((3*w, 9*h), (3*w, -9*h))
    case 10:
      #  a b c
      # d e f g
      #  h i j
      offsets = [(-2*w,  3*h),  (0,  3*h), (2*w,  3*h),
          (-3*w, 0),       (-w, 0),    (w, 0),      (3*w, 0),
                 (-2*w, -3*h),  (0, -3*h), (2*w, -3*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((w, 9*h), (7*w, 3*h), (6*w, -6*h))
    case 11:
      # a b c d
      #  e f g
      # h i j k
      offsets = [(-3*w,  3*h),    (-w,  3*h), (w,  3*h),   (3*w, 3*h),
                           (-2*w, 0),     (0, 0),    (2*w, 0),
                 (-3*w, -3*h),    (-w, -3*h), (w, -3*h),   (3*w, -3*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((w, 9*h), (8*w, 6*h), (7*w, -3*h))
    case 12:
      #   a b c d
      #  e f g h
      # i j k l
      offsets = [(-2*w,  3*h), (   0,  3*h), (2*w,  3*h), (4*w,  3*h),
                 (-3*w,    0), (  -w,    0), (  w,    0), (3*w,    0),
                 (-4*w, -3*h), (-2*w, -3*h), (  0, -3*h), (2*w, -3*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((3*w, 9*h), (8*w, 0), (5*w, -9*h))
    case 13:
      #  a b c d
      # e f g h i
      #  j k l m
      offsets = [
             (-3*w,  3*h),    (-w,  3*h), (w,  3*h),    (3*w,  3*h),
      (-4*w, 0),       (-2*w, 0),     (0, 0),    ( 2*w, 0),      (4*w, 0),
             (-3*w, -3*h),    (-w, -3*h), (w, -3*h),    (3*w, -3*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((w, 9*h), (9*w, 3*h), (8*w, -6*h))
    case 14:
      #   a b c
      #  d e f g
      # h i j k
      #  l m n
      offsets = [
                          (-3*w/2,  9*h/2), (  w/2,  9*h/2), ( 5*w/2,  9*h/2),
        (-5*w/2,  3*h/2), (  -w/2,  3*h/2), (3*w/2,  3*h/2), ( 7*w/2,  3*h/2),
        (-7*w/2, -3*h/2), (-3*w/2, -3*h/2), (  w/2, -3*h/2), ( 5*w/2, -3*h/2),
        (-5*w/2, -9*h/2), (  -w/2, -9*h/2), (3*w/2, -9*h/2)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((2*w, 12*h), (8*w, 6*h), (6*w, -6*h))
    case 15:
      #   a b c d e
      #  f g h i j
      # k l m n o
      offsets = [
        (-3*w,  3*h), (  -w,  3*h), ( w,  3*h), (3*w,  3*h), (5*w,  3*h),
        (-4*w,    0), (-2*w,    0), ( 0,    0), (2*w,    0), (4*w,    0),
        (-5*w, -3*h), (-3*w, -3*h), (-w, -3*h), (  w, -3*h), (3*w, -3*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((3*w, 9*h), (10*w, 0))
    case 16:
      #    a
      #   b c
      #  d e f
      # g h i j
      #  k l m
      #   n o
      #    p
      offsets = [              (0,  9*h),
                       (-w,  6*h),    (w,  6*h),
             (-2*w,  3*h),     (0,  3*h),    ( 2*w,  3*h),
      (-3*w, 0),       (-w,    0),    ( w,   0),       ( 3*w, 0),
             (-2*w, -3*h),     (0, -3*h),    ( 2*w, -3*h),
                       (-w, -6*h),    (w, -6*h),
                               (0, -9*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      unit.setup_vectors((4*w, 12*h), (4*w, -12*h))
    case 19:
      #    a b c
      #   l m n d
      #  k r s o e
      #   j q p f
      #    i h g
      offsets = [      (-2*w,  6*h), (0,  6*h), (2*w,  6*h),
               (-3*w,  3*h), (-w,  3*h), ( w,  3*h), ( 3*w,  3*h),
           (-4*w,  0), (-2*w,    0), (0,    0), (2*w,    0), ( 4*w,  0),
               (-3*w, -3*h), (-w, -3*h), ( w, -3*h), ( 3*w, -3*h),
                       (-2*w, -6*h), (0, -6*h), (2*w, -6*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      reordered = [
        0, 1, 2, 6, 11, 15, 18, 17, 16, 12, 7, 3, 4, 5, 10, 14, 13, 8, 9]
      hexes = [hexes[i] for i in reordered]
      unit.setup_vectors((w, 15*h), (8*w, 6*h), (7*w, -9*h))
    case 37:
      #    A B C D
      #   g h i j E
      #  f s t u k F
      # e r y z v l G
      #  d q x w m H
      #   c p o n I
      #    b a K J 
      offsets = [ (-3*w, 9*h), (-w, 9*h), (w, 9*h), (3*w, 9*h),
              (-4*w, 6*h), (-2*w, 6*h), (0, 6*h), (2*w, 6*h), ( 4*w, 6*h),
          (-5*w, 3*h), (-3*w, 3*h), (-w, 3*h), (w, 3*h), (3*w, 3*h), (5*w, 3*h), 
      (-6*w, 0), (-4*w, 0), (-2*w, 0), (0, 0), (2*w, 0), (4*w, 0), (6*w, 0),  
          (-5*w, -3*h), (-3*w, -3*h), (-w, -3*h), (w, -3*h), (3*w, -3*h), (5*w, -3*h), 
               (-4*w, -6*h), (-2*w, -6*h), (0, -6*h), (2*w, -6*h), (4*w, -6*h),
                       (-3*w, -9*h), (-w, -9*h), (w, -9*h), (3*w, -9*h)]
      hexes = [affine.translate(hexagon, dx, dy) for dx, dy in offsets]
      reordered = [0, 1, 2, 3, 8, 14, 21, 27, 32, 36, 35, 34, 33,
                   28, 22, 15, 9, 4, 5, 6, 7, 13, 20, 26, 31, 30,
                   29, 23, 16, 10, 11, 12, 19, 25, 24, 17, 18]
      hexes = [hexes[i] for i in reordered]
      unit.setup_vectors((w, 21*h), (11*w, 9*h), (10*w, -12*h))
    case _:
      return (f"""{unit.n}-colouring of hexes is not supported.
              Try a number between 2 and 16, or 19, or 37.""")
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(hexes))
  return None


def _setup_grid(unit:TileUnit) -> str|None:
  """Set up grid colouring of a regular array of squares.

  The supplied unit must have unit.n, unit.nrows, and unit.ncols set.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["n", "nrows", "ncols"]):
    return "Grid tiling requires n, nrows, and ncols to be supplied."
  sq = tiling_utils.get_regular_polygon(
    unit.spacing / np.sqrt(unit.nrows * unit.ncols), 4)
  s = sq.bounds[2] - sq.bounds[0]
  h = s * unit.nrows
  w = s * unit.ncols
  unit.n = min(unit.n, unit.nrows * unit.ncols)
  tr = [(-w/2 + s/2 + s*col, h/2 - s/2 - s*row)
        for row in range(unit.nrows)
        for col in range(unit.ncols)]
  squares = [affine.translate(sq, v[0], v[1]) for v in tr]
  unit.setup_vectors((0, h), (w, 0))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": sorted(list(string.ascii_letters)[:unit.n])},
    crs = unit.crs,
    geometry = gpd.GeoSeries(squares))
  return None


def _setup_square_colouring(unit:TileUnit) -> str|None:
  """Set up colourings of a regular array of squares.

  The supplied unit must have unit.n set.

  Args:
    unit (TileUnit):  the TileUnit to setup.

  """
  if any(k not in unit.__dict__ for k in ["n"]):
    return "Square colouring tiling requires n to be supplied."
  sq = tiling_utils.get_regular_polygon(unit.spacing / np.sqrt(unit.n), 4)
  s = sq.bounds[2] - sq.bounds[0]
  match unit.n:
    case 2:
      # Copy and translate square
      tr = [(-s/2, 0), (s/2, 0)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, s), (2*s, 0))
    case 3:
      # Copy and translate square
      tr = [(-s/2,  s/2),
            (-s/2, -s/2), (s/2, -s/2)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((s, s), (-s, 2*s), (2*s, -s))
    case 4:
      # Copy and translate square
      tr = [(-s/2,  s/2), ( s/2,  s/2), 
            (-s/2, -s/2), ( s/2, -s/2)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, unit.spacing), (unit.spacing, 0))
    case 5:
      # Copy and translate square
      tr = [         (0,  s), 
            (-s, 0), (0,  0), ( s, 0),
                     (0, -s)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((s, 2*s), (2*s, -s))
    case 6:
      # Copy and translate square
      tr = [(-s, s), (0,  s),
            (-s, 0), (0,  0), ( s,  0),
                     (0, -s)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, 3*s), (2*s, s), (2*s, -2*s))
    case 7:
      # Copy and translate square
      tr = [(-s, s), (0,  s),
            (-s, 0), (0,  0), ( s,  0),
                     (0, -s), ( s, -s)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((2*s, s), (3*s, -2*s), (s, -3*s))
    case 8:
      # Copy and translate square
      tr = [(-s,  s), (0,  s), ( s, s),
            (-s,  0), (0,  0), ( s, 0),
            (-s, -s), (0, -s)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((s, 3*s), (3*s, s), (2*s, -2*s))
    case 9:
      # Copy and translate square
      tr = [(-s,  s), (0,  s), ( s,  s),
            (-s,  0), (0,  0), ( s,  0),
            (-s, -s), (0, -s), ( s, -s)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, unit.spacing), (unit.spacing, 0))
    case 16:
      # Copy and translate square
      tr = [(-3*s/2,  3*s/2), (-s/2,  3*s/2), ( s/2,  3*s/2), ( 3*s/2,  3*s/2),
            (-3*s/2,    s/2), (-s/2,    s/2), ( s/2,    s/2), ( 3*s/2,    s/2),
            (-3*s/2,   -s/2), (-s/2,   -s/2), ( s/2,   -s/2), ( 3*s/2,   -s/2),
            (-3*s/2, -3*s/2), (-s/2, -3*s/2), ( s/2, -3*s/2), ( 3*s/2, -3*s/2)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, unit.spacing), (unit.spacing, 0))
    case 25:
      # Copy and translate square
      tr = [(-2*s,  2*s), (-s,  2*s), (0,  2*s), ( s,  2*s), ( 2*s,  2*s),
            (-2*s,    s), (-s,    s), (0,    s), ( s,    s), ( 2*s,    s),
            (-2*s,    0), (-s,    0), (0,    0), ( s,    0), ( 2*s,    0),
            (-2*s,   -s), (-s,   -s), (0,   -s), ( s,   -s), ( 2*s,   -s),
            (-2*s, -2*s), (-s, -2*s), (0, -2*s), ( s, -2*s), ( 2*s, -2*s)]
      squares = [affine.translate(sq, v[0], v[1]) for v in tr]
      unit.setup_vectors((0, unit.spacing), (unit.spacing, 0))
    case _:
      return (f"""{unit.n}-colouring of squares is not supported.
              Try a number between 2 and 9, 16, or 25.""")

  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(squares))
  return None


def _get_corners_and_mid_points(
      shape:geom.Polygon,
      offset:float = 0.5,
    ) -> list[geom.Point]:
  """Return corners and midpoints of polygon.

  Args:
    shape (geom.Polygon): polygon for which corners and midpoints are requested.
    offset (float): how far along sides midpoints should be.

  Returns:
    list[geom.Point]: list of points in order (i.e corners and midpoints are
      interleaved).

  """
  corners = [geom.Point(p) for p in shape.exterior.coords][:-1]
  midpoints = [ls.interpolate(offset, normalized = True)
               for ls in tiling_utils.get_sides(shape)]
  return list(itertools.chain(*[
                [a, b] for a, b in zip(corners, midpoints, strict = True)]))


def _setup_square_dissection(unit:TileUnit) -> str|None:
  """Set up dissection of a square."""
  if any(k not in unit.__dict__ for k in ["n", "offset", "offset_angle"]):
    return ("""Square dissection tiling requires n, offset, and offset_angle
            to be be supplied.""")
  outer = tiling_utils.get_regular_polygon(unit.spacing, 4)
  inner = affine.scale(outer, 1 / np.sqrt(unit.n), 1 / np.sqrt(unit.n))
  inner = affine.rotate(inner, unit.offset_angle)
  outer_pts = _get_corners_and_mid_points(outer)
  inner_pts = _get_corners_and_mid_points(inner)
  outer_pts = outer_pts + outer_pts[:1]
  inner_pts = inner_pts + inner_pts[:1]
  inner_pts = inner_pts[::-1]
  match unit.n:
    case 3:
      polys = [inner]
      if unit.offset == 1:
        polys.extend([
          geom.Polygon(outer_pts[1:3] + outer_pts[4:6] + inner_pts[4:9:2]),
          geom.Polygon(outer_pts[5:7] + outer_pts[0:2] + inner_pts[0:5:2])])
      else:
        polys.extend([
          geom.Polygon(outer_pts[0:5:2] + inner_pts[4:9:2]),
          geom.Polygon(outer_pts[4:9:2] + inner_pts[0:5:2])])
    case 5:
      polys = [inner]
      if unit.offset == 1:
        polys.extend([
          geom.Polygon(outer_pts[1:4] + inner_pts[6:9:2]),
          geom.Polygon(outer_pts[3:6] + inner_pts[4:7:2]),
          geom.Polygon(outer_pts[5:8] + inner_pts[2:5:2]),
          geom.Polygon(outer_pts[7:9] + outer_pts[1:2] + inner_pts[0:3:2])])
      else:
        polys.extend([
          geom.Polygon(outer_pts[0:3:2] + inner_pts[6:9:2]),
          geom.Polygon(outer_pts[2:5:2] + inner_pts[4:7:2]),
          geom.Polygon(outer_pts[4:7:2] + inner_pts[2:5:2]),
          geom.Polygon(outer_pts[6:9:2] + inner_pts[0:3:2])])
    case 9:
      polys = [inner]
      if unit.offset == 1:
        polys.extend([
          geom.Polygon(outer_pts[1:3] + inner_pts[7:9]),
          geom.Polygon(outer_pts[2:4] + inner_pts[6:8]),
          geom.Polygon(outer_pts[3:5] + inner_pts[5:7]),
          geom.Polygon(outer_pts[4:6] + inner_pts[3:6]),
          geom.Polygon(outer_pts[5:7] + inner_pts[3:5]),
          geom.Polygon(outer_pts[6:8] + inner_pts[2:4]),
          geom.Polygon(outer_pts[7:9] + inner_pts[1:3]),
          geom.Polygon(outer_pts[0:2] + inner_pts[0:2])])
      else:
        polys.extend([
          geom.Polygon(outer_pts[0:2] + inner_pts[7:9]),
          geom.Polygon(outer_pts[1:3] + inner_pts[6:8]),
          geom.Polygon(outer_pts[2:4] + inner_pts[5:7]),
          geom.Polygon(outer_pts[3:5] + inner_pts[4:6]),
          geom.Polygon(outer_pts[4:6] + inner_pts[3:5]),
          geom.Polygon(outer_pts[5:7] + inner_pts[2:4]),
          geom.Polygon(outer_pts[6:8] + inner_pts[1:3]),
          geom.Polygon(outer_pts[7:9] + inner_pts[0:2])])
    case _:
      unit.base_shape = TileShape.RECTANGLE
      return f"{unit.n} dissection of square not available"
  unit.setup_vectors((0, unit.spacing), (unit.spacing, 0))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))
  return None


def _setup_hex_dissection(unit:TileUnit) -> str|None:
  """Set up dissection of a hexagon."""
  if any(k not in unit.__dict__ for k in ["n", "offset", "offset_angle"]):
    return ("""Hex dissection tiling requires n, offset, and offset_angle
            to be be supplied.""")
  outer = tiling_utils.get_regular_polygon(unit.spacing, 6)
  if unit.n == 9:
    inner = affine.scale(outer, 1/np.sqrt(3), 1/np.sqrt(3))
  else:
    inner = affine.scale(outer, 1 / np.sqrt(unit.n), 1 / np.sqrt(unit.n))
  inner = affine.rotate(inner, unit.offset_angle)
  outer_pts = _get_corners_and_mid_points(outer)
  inner_pts = _get_corners_and_mid_points(inner)
  outer_pts = outer_pts + outer_pts[:1]
  inner_pts = inner_pts + inner_pts[:1]
  inner_pts = inner_pts[::-1]
  match unit.n:
    case 3:
      polys = [inner]
      if unit.offset == 1:
        polys.extend([
          geom.Polygon(outer_pts[1:3] + outer_pts[2:7:2] +
                       outer_pts[7:8] + inner_pts[6::2]),
          geom.Polygon(outer_pts[7:9] + outer_pts[8::2] +
                       outer_pts[0:2] + inner_pts[0:7:2])])
      else:
        polys.extend([
          geom.Polygon(outer_pts[0:7:2] + inner_pts[6::2]),
          geom.Polygon(outer_pts[6::2]  + inner_pts[0:7:2])])
    case 4:
      polys = [inner]
      if unit.offset == 1:
        polys.extend([
          geom.Polygon(outer_pts[1:3]  + outer_pts[2:5:2] +
                       outer_pts[4:6]  + inner_pts[8::2]),
          geom.Polygon(outer_pts[5:7]  + outer_pts[6:11:2] +
                       outer_pts[8:10] + inner_pts[4:9:2]),
          geom.Polygon(outer_pts[9:11] + outer_pts[10::2] +
                       outer_pts[0:2]  + inner_pts[0:5:2])])
      else:
        polys.extend([
          geom.Polygon(outer_pts[0:5:2] + inner_pts[8::2]),
          geom.Polygon(outer_pts[4:9:2] + inner_pts[4:9:2]),
          geom.Polygon(outer_pts[8::2]  + inner_pts[0:5:2])])
    case 7:
      polys = [inner]
      if unit.offset == 1:
        polys.extend([
          geom.Polygon(outer_pts[1:3]  + outer_pts[2:4]   + inner_pts[10::2]),
          geom.Polygon(outer_pts[3:5]  + outer_pts[4:6]   + inner_pts[8:11:2]),
          geom.Polygon(outer_pts[5:7]  + outer_pts[6:8]   + inner_pts[6:9:2]),
          geom.Polygon(outer_pts[7:9]  + outer_pts[8:10]  + inner_pts[4:7:2]),
          geom.Polygon(outer_pts[9:11] + outer_pts[10:12] + inner_pts[2:5:2]),
          geom.Polygon(outer_pts[11:]  + outer_pts[:2]    + inner_pts[0:3:2])])
      else:
        polys.extend([
          geom.Polygon(outer_pts[0:3:2]  + inner_pts[10::2]),
          geom.Polygon(outer_pts[2:5:2]  + inner_pts[8:11:2]),
          geom.Polygon(outer_pts[4:7:2]  + inner_pts[6:9:2]),
          geom.Polygon(outer_pts[6:9:2]  + inner_pts[4:7:2]),
          geom.Polygon(outer_pts[8:11:2] + inner_pts[2:5:2]),
          geom.Polygon(outer_pts[10::2]  + inner_pts[0:3:2])])
    case 9:
      if unit.offset == 1:
        polys = [
          geom.Polygon(inner_pts[1:3]  + inner_pts[2:5:2] +
                       inner_pts[4:6]  + [(0, 0)]),
          geom.Polygon(inner_pts[5:7]  + inner_pts[6:9:2] +
                       inner_pts[8:10] + [(0, 0)]),
          geom.Polygon(inner_pts[9:11] + inner_pts[10::2] +
                       inner_pts[0:2]  + [(0, 0)]),
          geom.Polygon(outer_pts[1:3]  + outer_pts[2:4]   + inner_pts[10::2]),
          geom.Polygon(outer_pts[3:5]  + outer_pts[4:6]   + inner_pts[8:11:2]),
          geom.Polygon(outer_pts[5:7]  + outer_pts[6:8]   + inner_pts[6:9:2]),
          geom.Polygon(outer_pts[7:9]  + outer_pts[8:10]  + inner_pts[4:7:2]),
          geom.Polygon(outer_pts[9:11] + outer_pts[10:12] + inner_pts[2:5:2]),
          geom.Polygon(outer_pts[11:]  + outer_pts[:2]    + inner_pts[0:3:2])]
      else:
        polys = [
          geom.Polygon([*inner_pts[0:5:2], (0, 0)]),
          geom.Polygon([*inner_pts[4:9:2], (0, 0)]),
          geom.Polygon([*inner_pts[8::2], (0, 0)]),
          geom.Polygon(outer_pts[0:3:2]  + inner_pts[10::2]),
          geom.Polygon(outer_pts[2:5:2]  + inner_pts[8:11:2]),
          geom.Polygon(outer_pts[4:7:2]  + inner_pts[6:9:2]),
          geom.Polygon(outer_pts[6:9:2]  + inner_pts[4:7:2]),
          geom.Polygon(outer_pts[8:11:2] + inner_pts[2:5:2]),
          geom.Polygon(outer_pts[10::2]  + inner_pts[0:3:2])]
    case _:
      unit.base_shape = TileShape.HEXAGON
      return f"{unit.n} dissection of hexagon not available"
  unit.setup_vectors((0, unit.spacing),
                     (unit.spacing * sqrt3/2, unit.spacing / 2),
                     (unit.spacing * sqrt3/2, -unit.spacing / 2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:unit.n]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))
  return None


def _setup_star_polygon_1(unit:TileUnit) -> str|None:
  """Set up a star-polygon tiling."""
  if any(k not in unit.__dict__ for k in ["code", "point_angle"]):
    return ("""Star tiling 1 requires code and point_angle
            to be be supplied.""")
  match unit.code:
    case "33":
      _setup_star_polys_33(unit)
    case "36":
      _setup_star_polys_36(unit)
    case "44":
      _setup_star_polys_44(unit)
    case "63":
      _setup_star_polys_63(unit)
    case _:
      return "Invalid code supplied for star tiling 1"
  # star polygons cause issues for _setup_regularised_prototile
  # due to negative buffering required in unioning, so each of
  # these calculates a unit.union polygon and we assign here
  unit.regularised_prototile = gpd.GeoDataFrame(
    geometry = gpd.GeoSeries([unit.union]), crs = unit.crs)
  return None

def _setup_star_polys_33(unit:TileUnit) -> None:
  """Set up polygon by 3-star and triangles."""
  star = _get_star(3, unit.spacing / 2, unit.point_angle)
  s_sides = tiling_utils.get_sides(star)
  tri1 = _get_regular_polygon_from_one_side(s_sides[0], 3)
  tri2 = _get_regular_polygon_from_one_side(s_sides[5], 3)
  unit.union = geom.Polygon(tri1.exterior.coords[2:0:-1] +
                       star.exterior.coords[2:6] +
                       tri2.exterior.coords[2:0:-1])
  tips = list(star.exterior.coords)[0::2]
  dents = list(star.exterior.coords)[1::2]
  v1 = geom.Point(dents[0][0] - tips[2][0], dents[0][1] - tips[2][1])
  v2 = affine.rotate(v1, -60, (0, 0))
  v3 = affine.rotate(v2, -60, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abc")},
    crs = unit.crs,
    geometry = gpd.GeoSeries([star, tri1, tri2]))


def _setup_star_polys_36(unit:TileUnit) -> None:
  """Set up tiling by 3-star and hexagons."""
  star1 = _get_star(3, unit.spacing / 3, unit.point_angle)
  hexagon = _get_regular_polygon_from_one_side(
    tiling_utils.get_sides(star1)[0], 6)
  c = hexagon.centroid
  star1 = affine.translate(star1, -c.x, -c.y)
  hexagon = affine.translate(hexagon, -c.x, -c.y)
  star2 = affine.rotate(star1, 180, (0, 0))
  unit.union = geom.Polygon(
    hexagon.exterior.coords[5:3:-1] + star2.exterior.coords[1:7] +
    hexagon.exterior.coords[2:0:-1] + star1.exterior.coords[1:7])
  tips1 = list(star1.exterior.coords)[0::2]
  dents2 = list(star2.exterior.coords)[1::2]
  v1 = geom.Point(dents2[2][0] - tips1[0][0], dents2[2][1] - tips1[0][1])
  v2 = affine.rotate(v1, -60, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abc")},
    crs = unit.crs,
    geometry = gpd.GeoSeries([star1, star2, hexagon]))


def _setup_star_polys_44(unit:TileUnit) -> None:
  """Set up tiling by 4-stars and squares."""
  star = _get_star(4, unit.spacing / 2, unit.point_angle)
  square = _get_regular_polygon_from_one_side(
    tiling_utils.get_sides(star)[0], 4)
  unit.union = geom.Polygon(
    square.exterior.coords[3:1:-1] + star.exterior.coords[1:])
  tips = list(star.exterior.coords)[0::2]
  dents = list(star.exterior.coords)[1::2]
  v1 = geom.Point(tips[0][0] - dents[1][0], tips[0][1] - dents[1][1])
  v2 = affine.rotate(v1, 90, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("ab")},
    crs = unit.crs,
    geometry = gpd.GeoSeries([star, square]))


def _setup_star_polys_63(unit:TileUnit) -> None:
  """Set up tiling by 6-star and triangles."""
  star = _get_star(6, unit.spacing / 2, unit.point_angle)
  s_sides = tiling_utils.get_sides(star)
  tri1 = _get_regular_polygon_from_one_side(s_sides[0], 3)
  tri2 = _get_regular_polygon_from_one_side(s_sides[6], 3)
  unit.union = geom.Polygon(
    tri1.exterior.coords[2:0:-1] + star.exterior.coords[2:7] +
    tri2.exterior.coords[2:0:-1] + star.exterior.coords[8:])
  tips = list(star.exterior.coords)[0::2]
  dents = list(star.exterior.coords)[1::2]
  v1 = geom.Point(dents[0][0] - tips[4][0], dents[0][1] - tips[4][1])
  v2 = affine.rotate(v1, -60, (0, 0))
  v3 = affine.rotate(v2, -60, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abc")},
    crs = unit.crs,
    geometry = gpd.GeoSeries([star, tri1, tri2]))


def _get_regular_polygon_from_one_side(
    side:geom.LineString,
    n:int,
  ) -> geom.Polygon:
  """Return a regular polygon given one side.

  Args:
    side (geom.LineString): one side of the requested polygon.
    n (int): number of sides of the polygon.

  Returns:
    geom.Polygon: the polygon.

  """
  points = list(side.coords)
  turn_angle = np.pi - 2 * np.pi / n
  for _ in range(1, n):
    points.append(
      affine.rotate(geom.Point(points[-2]),
                    turn_angle, origin = points[-1],
                    use_radians = True))
  return geom.Polygon(points[:-1])


def _get_star(
    n:int,
    point_length:float,
    point_angle:float,
  ) -> geom.Polygon:
  """Return a star polygon.

  Args:
    n (int): the number of points.
    point_length(float): the length of start points.
    point_angle(float): angle at the tip of each point.

  Returns:
    geom.Polygon: the star polygon.

  """
  pt_angle = np.radians(point_angle)
  hp = point_length * (np.cos(pt_angle / 2) + np.sin(pt_angle / 2) /
                       np.tan(np.pi / n))
  hd = point_length * np.sin(pt_angle / 2) / np.sin(np.pi / n)
  interior_angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint = False)
  tips = [geom.Point(hp * np.cos(a), hp * np.sin(a))
          for a in interior_angles[::2]]
  dents = [geom.Point(hd * np.cos(a), hd * np.sin(a))
             for a in interior_angles[1::2]]
  return geom.Polygon(list(itertools.chain(*zip(tips, dents, strict = True))))


def _setup_star_polygon_2(unit:TileUnit) -> str|None:
  """Set up a star-polygon tiling with no tiling point angle option."""
  if any(k not in unit.__dict__ for k in ["code"]):
    return "Star tiling 2 requires code to be be supplied."
  match unit.code:
    case "45":
      _setup_star_polys_45(unit)
    case "464":
      _setup_star_polys_464(unit)
    case "466":
      _setup_star_polys_466(unit)
    case "64":
      _setup_star_polys_64(unit)
    case "66":
      _setup_star_polys_66(unit)
    case "663":
      _setup_star_polys_663(unit)
    case "844":
      _setup_star_polys_844(unit)
    case _:
      return "Invalid code supplied for star tiling 2"
  # star polygons cause issues for _setup_regularised_prototile
  # due to negative buffering required in unioning, so each of
  # these calculates a unit.union polygon and we assign here
  unit.regularised_prototile = gpd.GeoDataFrame(
    geometry = gpd.GeoSeries([unit.union]), crs = unit.crs)
  return None


def _setup_star_polys_45(unit:TileUnit) -> None:
  """Set up tiling by regular pentagon and 4-stars."""
  pents = []
  pent = tiling_utils.get_regular_polygon(unit.spacing / 2, 5)
  pents.append(affine.translate(pent, 0, -pent.bounds[1]))
  pents.append(
    affine.rotate(pents[0], -126, tiling_utils.get_corners(pents[0])[0]))
  pents.extend([affine.rotate(p, 180, (0, 0)) for p in pents])
  star_diag1 = geom.LineString([
    tiling_utils.get_corners(pents[0])[2],
    affine.rotate(tiling_utils.get_corners(pents[0])[2], -90,
                  tiling_utils.get_corners(pents[0])[0])])
  star_diag2 = affine.rotate(star_diag1, 90)
  star_centre = star_diag1.intersection(star_diag2)
  star_pts = [affine.rotate(
    tiling_utils.get_corners(pents[0])[0], a, star_centre)
    for a in range(0, 360, 90)]
  star_dents = [
    affine.rotate(tiling_utils.get_corners(pents[0])[1], a, star_centre)
    for a in range(0, 360, 90)]
  star1 = geom.Polygon(
    itertools.chain(*zip(star_dents, star_pts, strict = True)))
  star2 = affine.rotate(star1, -126, tiling_utils.get_corners(pents[0])[2])
  unit.union = geom.Polygon(
    star1.exterior.coords[4:8] + star2.exterior.coords[:4] +
    pents[3].exterior.coords[3:] + pents[2].exterior.coords[1:5] +
    pents[1].exterior.coords[1:4])
  ppts = list(pents[0].exterior.coords)[:-1]
  spts = list(star1.exterior.coords)[:-1]
  v1 = geom.Point(spts[5][0] - ppts[-1][0], spts[5][0] - ppts[-1][0])
  v2 = affine.rotate(v1, -90, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")},
    geometry = gpd.GeoSeries([*pents, star1, star2]),
    crs = unit.crs)


def _setup_star_polys_464(unit:TileUnit) -> None:
  """Set up 4-star and hexagon and square tiling."""
  star = _get_star(4, unit.spacing / 3, 30)
  hex1 = _get_regular_polygon_from_one_side(tiling_utils.get_sides(star)[0], 6)
  hex2 = _get_regular_polygon_from_one_side(tiling_utils.get_sides(star)[2], 6)
  square = _get_regular_polygon_from_one_side(
    tiling_utils.get_sides(hex2)[5].reverse(), 4)
  unit.union = geom.Polygon(
    hex1.exterior.coords[5:2:-1] + square.exterior.coords[2:1:-1] +
    hex2.exterior.coords[5:1:-1] + star.exterior.coords[5:])
  p0 = tiling_utils.get_corners(star)[6]
  p1 = tiling_utils.get_corners(hex2)[3]
  v1 = geom.Point(p1.x - p0.x, p1.y - p0.y)
  v2 = affine.rotate(v1, -90, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcd")},
    geometry = gpd.GeoSeries([star, hex1, hex2, square]),
    crs = unit.crs)


def _setup_star_polys_466(unit:TileUnit) -> None:
  """Set up 4-star and two hexagon tiling."""
  hex1 = tiling_utils.get_regular_polygon(
    unit.spacing / np.sqrt(1 + sqrt3 / 2), 6)
  hex2 = affine.scale(hex1, 0.5, 0.5)
  hex1 = affine.rotate(hex1, 30)
  w1 = hex1.bounds[2] - hex1.bounds[0]
  w2 = hex2.bounds[2] - hex2.bounds[0]
  hex2 = affine.translate(hex2, w1/2 + w2/2)
  hexes = [hex1] + [affine.rotate(hex2, a, (0, 0)) for a in [60, 120]]
  h1_pts = list(hexes[1].exterior.coords)
  h2_pts = list(hexes[2].exterior.coords)
  long_diag_1 = geom.LineString([h1_pts[4], h2_pts[0]])
  long_diag_2 = affine.rotate(long_diag_1, 90)
  short_diag_1 = geom.LineString([h1_pts[3], h2_pts[5]])
  short_diag_2 = affine.rotate(short_diag_1, 90)
  star = geom.Polygon([long_diag_1.coords[0], short_diag_1.coords[0],
                       long_diag_2.coords[0], short_diag_2.coords[0],
                       long_diag_1.coords[1], short_diag_1.coords[1],
                       long_diag_2.coords[1], short_diag_2.coords[1]])
  stars = [affine.rotate(star, a, (0, 0)) for a in [-60, 0, 60]]
  unit.union = geom.Polygon(
    hexes[0].exterior.coords[4:6] + hexes[0].exterior.coords[0:1] +
    stars[0].exterior.coords[7:2:-1] + hexes[1].exterior.coords[2:4] +
    stars[1].exterior.coords[4:5] + hexes[2].exterior.coords[2:4] +
    stars[2].exterior.coords[4:0:-1])
  p0 = tiling_utils.get_corners(hexes[0])[5]
  p1 = tiling_utils.get_corners(stars[1])[3]
  v1 = geom.Point(p1.x - p0.x, p1.y - p0.y)
  v2, v3 = tuple([affine.rotate(v1, a, (0, 0)) for a in [-60, -120]])
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")},
    geometry = gpd.GeoSeries(hexes + stars),
    crs = unit.crs)


def _setup_star_polys_64(unit:TileUnit) -> None:
  """Set up 6-star and square tiling."""
  star = _get_star(6, unit.spacing / 2, 30)
  square = _get_regular_polygon_from_one_side(
    tiling_utils.get_sides(star)[0], 4)
  squares = [affine.rotate(square, a, (0, 0)) for a in range(0, 360, 60)]
  unit.union = geom.Polygon(
    squares[0].exterior.coords[3:1:-1] + squares[1].exterior.coords[3:1:-1] +
    squares[2].exterior.coords[3:1:-1] + star.exterior.coords[6:])
  p1 = tiling_utils.get_corners(star)[4]
  p0 = tiling_utils.get_corners(star)[8]
  v1 = geom.Point(p1.x - p0.x, p1.y - p0.y)
  v2, v3 = tuple([affine.rotate(v1, a, (0, 0)) for a in [-60, -120]])
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcd")},
    geometry = gpd.GeoSeries([star, *squares]),
    crs = unit.crs)


def _setup_star_polys_66(unit:TileUnit) -> None:
  """Set up 6-star and hexagon tiling."""
  star = _get_star(6, unit.spacing / 4, 60)
  hex1 = _get_regular_polygon_from_one_side(tiling_utils.get_sides(star)[0], 6)
  hex2 = _get_regular_polygon_from_one_side(tiling_utils.get_sides(star)[6], 6)
  polys = [affine.rotate(p, -30, (0, 0)) for p in [star, hex1, hex2]]
  unit.union = geom.Polygon(
    polys[1].exterior.coords[5:1:-1] + polys[0].exterior.coords[3:7] +
    polys[2].exterior.coords[5:1:-1] + polys[0].exterior.coords[8:])
  vs = tiling_utils.get_corners(polys[0])[0:12:2]
  p0, p1 = vs[0], vs[3]
  v1 = geom.Point(p1.x - p0.x, p1.y - p0.y)
  v2 = affine.rotate(v1, -120, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abc")},
    geometry = gpd.GeoSeries(polys),
    crs = unit.crs)


def _setup_star_polys_663(unit:TileUnit) -> None:
  """Set up 6-star, hexagon, and triangle tiling."""
  star = _get_star(6, unit.spacing / 4, 60)
  hexagon = _get_regular_polygon_from_one_side(
    tiling_utils.get_sides(star)[0], 6)
  hexes = [affine.rotate(hexagon, a, (0, 0))
           for a in range(180, 360, 60)]
  tri = _get_regular_polygon_from_one_side(
    tiling_utils.get_sides(hexes[0])[2].reverse(), 3)
  triangles = [affine.rotate(tri, a, (0, 0)) for a in [0, 60]]
  unit.union = geom.Polygon(
    star.exterior.coords[0:6] + hexes[0].exterior.coords[-1:-5:-1] +
    hexes[1].exterior.coords[5:2:-1] + hexes[2].exterior.coords[5:2:-1])
  vs = tiling_utils.get_corners(star)[1:12:2]
  p0, p1 = vs[0], vs[3]
  v1 = geom.Point(2 * (p1.x - p0.x), 2 * (p1.y - p0.y))
  v2, v3 = tuple([affine.rotate(v1, a, (0, 0)) for a in [-60, -120]])
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcdef")},
    geometry = gpd.GeoSeries([star, *hexes, *triangles]),
    crs = unit.crs)


def _setup_star_polys_844(unit:TileUnit) -> None:
  """Set up 8-star and two square tiling."""
  star = _get_star(8, unit.spacing / 4, 45)
  square = _get_regular_polygon_from_one_side(
    tiling_utils.get_sides(star)[0], 4)
  squares = [affine.rotate(square, a, (0, 0)) for a in range(0, 135, 45)]
  polys = [affine.rotate(p, -22.5, (0, 0)) for p in [star, *squares]]
  polys[2] = affine.scale(
    polys[2], 2, 2, origin = tiling_utils.get_corners(polys[2])[1])
  unit.union = geom.Polygon(
    polys[1].exterior.coords[3:1:-1] + polys[2].exterior.coords[4:1:-1] +
    polys[3].exterior.coords[4:1:-1] + polys[0].exterior.coords[7:])
  vs = tiling_utils.get_corners(polys[0])
  p0, p1 = vs[10], vs[0]
  v1 = geom.Point(p1.x - p0.x, p1.y - p0.y)
  v2 = affine.rotate(v1, 90, (0, 0))
  unit.setup_vectors((v1.x, v1.y), (v2.x, v2.y))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list("abcd")},
    geometry = gpd.GeoSeries(polys),
    crs = unit.crs)


def _setup_chavey(unit:TileUnit) -> str|None:
  """Set up a selection of k-regular tilings by polygons.

  Tilings are from:

  Chavey DP. 1989. Tilings by regular polygons—II: A catalog of tilings.
    Computers & Mathematics with Applications. 17(1-3) 149-165.
    https://doi.org/10.1016/0898-1221(89)90156-9.

  The 'code' letters (alphabetical) are entirely made up and arbitrary as the
  vertex degree based codes offered by Chavey are impossible to make sensible
  use of. See https://www.biglist-tilings.com/ for another approach, which it
  would be interesting to see developed into something a little more coherent!

  NOTE: Not sure that these should be called 'chavey' - more of a grab bag for
  a bunch of other tilings. Unfortunately, their proper codes are unwieldy such
  as 3.3.3.4.4;3.3.4.3.4;4.4.4.4.
  """
  if any(k not in unit.__dict__ for k in ["code"]):
    return "Chavey tilings require a letter code to be be supplied."
  match unit.code:
    case "a":
      _setup_chavey_a(unit)
    case "b":
      _setup_chavey_b(unit)
    case "c":
      _setup_chavey_c(unit)
    case "d":
      _setup_chavey_d(unit)
    case "e":
      _setup_chavey_e(unit)
    case "f":
      _setup_chavey_f(unit)
    case "g":
      _setup_chavey_g(unit)
    case "h":
      _setup_chavey_h(unit)
    case "i":
      _setup_chavey_i(unit)
    case "j":
      _setup_chavey_j(unit)
    case "k":
      _setup_chavey_k(unit)
    case _:
      unit.base_shape = TileShape.HEXAGON
      return "No valid Chavey tiling code supplied."
  return None


def _setup_chavey_a(unit:TileUnit) -> None:
  """Set up 3.3.3.4.4;3.3.4.3.4;4.4.4.4."""
  s = unit.spacing / np.sqrt(5 + 2 * sqrt3)
  t = s * sqrt3 / 2
  sq = tiling_utils.get_regular_polygon(s, 4)
  squares = [affine.translate(sq, dx/2, dy/2) for dx in [-s, s]
                                              for dy in [-s, s]]
  squares.append(affine.rotate(squares[0], 150, origin = (-s, -s)))
  tri = tiling_utils.get_regular_polygon(t, 3)
  triangles = [affine.translate(tri, -s/2, s - tri.bounds[1])]
  triangles.append(affine.translate(tri, s/2, s - tri.bounds[1]))
  triangles.extend([affine.rotate(tr, 180, (s/4, s + t/2))
                    for tr in triangles])
  triangles.extend([affine.rotate(tr, 90, (0, 0)) for tr in triangles])
  triangles = [affine.rotate(tr, 90, (0, 0)) for tr in triangles]
  polys = [affine.translate(p, s / sqrt2 * np.cos(np.pi/12),
                               s / sqrt2 * np.sin(np.pi/12))
                            for p in squares + triangles]
  unit.setup_vectors((s/2, 2 * s + t), (2 * s + t, -s/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_b(unit:TileUnit) -> None:
  """Set up 3.4.4.6;3.4.6.4;4.4.4.4."""
  h = unit.spacing * sqrt3 / (1 + sqrt3) / sqrt2
  s = h / sqrt3
  t = h / 2
  hexes_and_tris = [tiling_utils.get_regular_polygon(h, 6)]
  hexes_and_tris.append(geom.Polygon([(-s/2, -t), (s/2, -t), (0, -2*t)]))
  hexes_and_tris.append(affine.rotate(hexes_and_tris[-1], 180, (0, 0)))
  hexes_and_tris.extend([
    affine.rotate(affine.translate(p, 0, h + s), 90, (0, h + s))
    for p in hexes_and_tris])
  squares = [geom.Polygon([(-s, 0), (-s - t, s/2),
                           (-t - s/2, t + s/2), (-s/2, t)])]
  squares.append(affine.translate(squares[0], s/2, t))
  squares.extend([affine.translate(sq, -t, s/2) for sq in squares])
  squares.extend([affine.rotate(sq, 60, (0, h + 2 * s))
                  for sq in squares])
  polys = [affine.translate(p, 0, -s) for p in hexes_and_tris + squares]
  unit.setup_vectors((s + 2 * t,  s + 2 * t),
                     (s + 2 * t, -s - 2 * t))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_c(unit:TileUnit) -> None:
  """Set up 3.4.4.6;3.4.6.4."""
  h = unit.spacing * sqrt3 / (3 + sqrt3)
  s = h / sqrt3
  t = h / 2
  hexes, squares, triangles = [], [], []
  squares.append(tiling_utils.get_regular_polygon(s, 4))
  squares.append(affine.translate(squares[0], 0,  s))
  squares.append(affine.translate(squares[0], 0, -s))
  squares.extend([affine.rotate(sq,  60, origin = (0, t + 3 * s / 2))
                  for sq in squares[:2]])
  squares.extend([affine.rotate(sq, -60, origin = (0, t + 3 * s / 2))
                  for sq in squares[:2]])
  squares.extend([affine.rotate(squares[1], a, origin = (0, t + 3 * s / 2))
                  for a in (120, -120)])
  hexes.append(tiling_utils.get_regular_polygon(h, 6))
  hexes[0] = affine.translate(hexes[0], 0, t + 3 * s / 2)
  hexes.append(affine.rotate(hexes[0], -90, origin = (-s / 2, s / 2)))
  hexes.append(affine.rotate(hexes[1], 180, origin = (0, 0)))
  triangles.append(geom.Polygon([(s/2,    s/2), (s/2, 3*s/2), (s/2 + t, s)]))
  triangles.append(geom.Polygon([(s/2, -3*s/2), (s/2,  -s/2), (s/2 + t, -s)]))
  triangles.extend([affine.rotate(tr, 180, origin = (0, 0))
                    for tr in triangles])
  triangles.extend([affine.rotate(triangles[0], a, origin = (0, t + 3 * s / 2))
                    for a in (60, -120)])
  polys = [affine.translate(p, 0, -t) for p in hexes + squares + triangles]
  unit.setup_vectors((0, h + 3 * s),
                     (3 * (h + s) / 2,  3 * s / 2 + t),
                     (3 * (h + s) / 2, -3 * s / 2 - t))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_d(unit:TileUnit) -> None:
  """Set up 3.3.3.4.4;3.3.4.3.4."""
  s = unit.spacing * sqrt2 / (3 + sqrt3)
  t = s * sqrt3 / 2
  squares, triangles = [], []
  squares.append(tiling_utils.get_regular_polygon(s, 4))
  squares.append(affine.translate(squares[0],    0 , -s))
  squares.append(affine.translate(squares[0], -s/2 , t + s))
  squares.append(affine.translate(squares[0],  s/2 , t + s))
  squares.append(affine.rotate(squares[0],  150, ( s/2, s/2)))
  squares.append(affine.rotate(squares[0], -150, (-s/2, s/2)))
  triangles.append(geom.Polygon([(-s/2, -s/2), (-s/2 - t,  0), (-s/2,  s/2)]))
  triangles.append(geom.Polygon([( s/2, -s/2), ( s/2 + t,  0), ( s/2,  s/2)]))
  triangles.append(geom.Polygon([(-s/2, s/2), (-s, t + s/2), ( 0, t + s/2)]))
  triangles.append(geom.Polygon([(-s/2, s/2), ( 0, t + s/2), ( s/2, s/2)]))
  triangles.append(geom.Polygon([( s/2, s/2), ( 0, t + s/2), ( s, t + s/2)]))
  triangles.extend([affine.rotate(tr, 180, (0, s + t))
                    for tr in triangles[2:5]])
  triangles.extend([affine.rotate(triangles[2], a, (-s, t + s/2))
                    for a in (-150, -210)])
  triangles.extend([affine.rotate(triangles[4], a, ( s, t + s/2))
                    for a in ( 150,  210)])
  polys = [affine.translate(p, 0, -t) for p in squares + triangles]
  unit.setup_vectors((unit.spacing / sqrt2,  unit.spacing / sqrt2),
                     (unit.spacing / sqrt2, -unit.spacing / sqrt2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_e(unit:TileUnit) -> None:
  """Set up 3.3.3.3.3.3;3.3.4.3.4."""
  s = unit.spacing / (1 + sqrt3)
  t = s * sqrt3 / 2
  squares, triangles = [], []
  triangles.append(geom.Polygon([(0, 0), (s/2, -t), (-s/2, -t)]))
  triangles.extend([affine.rotate(triangles[0], a, (0, 0))
                    for a in range(60, 360, 60)])
  triangles.append(geom.Polygon([(-s/2    ,       t),
                                 (-s/2 - t, s/2 + t),
                                 (-s/2    , s   + t)]))
  triangles.append(geom.Polygon([( s/2    ,        t),
                                 ( s/2    ,  s   + t),
                                 ( s/2 + t,  s/2 + t)]))
  squares.append(affine.translate(
    tiling_utils.get_regular_polygon(s, 4), 0, t + s/2))
  squares.append(affine.rotate(squares[0],  60, (0, 0)))
  squares.append(affine.rotate(squares[0], -60, (0, 0)))
  polys = [affine.translate(p, 0, 0) for p in squares + triangles]
  unit.setup_vectors((0, unit.spacing),
                     (unit.spacing * sqrt3/2, unit.spacing / 2),
                     (unit.spacing * sqrt3/2, -unit.spacing / 2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_f(unit:TileUnit) -> None:
  """Set up 3.3.3.3.3.3;3.3.3.3.6."""
  h = unit.spacing * sqrt3 / np.sqrt(7)
  s = h / sqrt3
  t = h / 2
  hexes, triangles = [], []
  hexes.append(tiling_utils.get_regular_polygon(h, 6))
  tri = tiling_utils.get_regular_polygon(t, 3)
  triangles.append(affine.translate(tri, s, -t - tri.bounds[1]))
  triangles.append(affine.rotate(triangles[0], 180, (5*s/4, -t/2)))
  triangles.append(affine.rotate(triangles[1], 180, (3*s/2, 0)))
  triangles.append(affine.rotate(triangles[2], 180, (5*s/4,  t/2)))
  triangles.extend([affine.rotate(tr, 180, (0, 0)) for tr in triangles])
  triangles.extend([affine.rotate(triangles[0], a, (0, 0))
                    for a in (120, 300)])
  polys = hexes + triangles
  unit.setup_vectors((2*s, h), (2*s, -h))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_g(unit:TileUnit) -> None:
  """Set up 3.3.3.3.3.3;3.3.3.3.6 (another one)."""
  h = unit.spacing / 2
  s = h / sqrt3
  t = h / 2
  hexes, triangles = [], []
  hexes.append(tiling_utils.get_regular_polygon(h, 6))
  tri = tiling_utils.get_regular_polygon(t, 3)
  triangles.append(affine.translate(tri, s, -t - tri.bounds[1]))
  triangles.append(affine.rotate(triangles[0], 180, (5*s/4, -t/2)))
  triangles.append(affine.rotate(triangles[1], 180, (3*s/2, 0)))
  triangles.extend([affine.rotate(tr, a, (0, 0)) for a in range(60, 360, 60)
                                                 for tr in triangles])
  polys = hexes + triangles
  unit.setup_vectors((0, 2*h), (3*s, h), (3*s, -h))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_h(unit:TileUnit) -> None:
  """Set up 3.3.4.3.4;3.4.6.4."""
  h = unit.spacing / (1 + 2 / sqrt3)
  s = h / sqrt3
  t = h / 2
  hexes, squares, triangles = [], [], []
  hexes.append(tiling_utils.get_regular_polygon(h, 6))
  squares.append(
    affine.translate(tiling_utils.get_regular_polygon(s, 4), 0, t + s/2))
  squares.extend([affine.rotate(squares[0], a, (0, 0))
                  for a in range(60, 360, 60)])
  polys = [affine.rotate(p, 30, (0, 0)) for p in hexes + squares]
  triangles.append(geom.Polygon([(-s/2, -s - t), (0, -s), (s/2, -s - t)]))
  triangles.extend([affine.rotate(triangles[0], a, (0, 0))
                    for a in range(60, 360, 60)])
  triangles.append(geom.Polygon(
                     [(s/2, s + t), (s/2 + t, 3*s/2 + t), (s/2 + t, s/2 + t)]))
  triangles.append(affine.rotate(triangles[-1], 180, (0, 0)))
  polys = [affine.rotate(p, 30, (0, 0)) for p in polys + triangles]
  unit.setup_vectors(( unit.spacing / 2, unit.spacing * sqrt3/2),
                     ( unit.spacing    , 0),
                     (-unit.spacing / 2, unit.spacing * sqrt3/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_i(unit:TileUnit) -> None:
  """Set up 3.3.3.3.3.3;3.3.4.3.4."""
  h = unit.spacing / (1 + 2 / sqrt3)
  s = h / sqrt3
  t = h / 2
  triangles1, squares, triangles2 = [], [], []
  hex_c = tiling_utils.get_corners(tiling_utils.get_regular_polygon(h, 6))
  triangles1 = [geom.Polygon([(0, 0), p1, p2])
                for p1, p2 in zip(hex_c[:-1], hex_c[1:], strict = True)]
  squares.append(
    affine.translate(tiling_utils.get_regular_polygon(s, 4), 0, t + s/2))
  squares.extend([affine.rotate(squares[0], a, (0, 0))
                  for a in range(60, 360, 60)])
  polys = [affine.rotate(p, 30, (0, 0)) for p in triangles1 + squares]
  triangles2.append(geom.Polygon([(-s/2, -s - t), (0, -s), (s/2, -s - t)]))
  triangles2.extend([affine.rotate(triangles2[0], a, (0, 0))
                     for a in range(60, 360, 60)])
  triangles2.append(geom.Polygon([(s/2    ,   s   + t),
                                  (s/2 + t, 3*s/2 + t),
                                  (s/2 + t,   s/2 + t)]))
  triangles2.append(affine.rotate(triangles2[-1], 180, (0, 0)))
  polys = [affine.rotate(p, 30, (0, 0)) for p in polys + triangles2]
  unit.setup_vectors(( unit.spacing / 2, unit.spacing * sqrt3/2),
                     ( unit.spacing    , 0),
                     (-unit.spacing / 2, unit.spacing * sqrt3/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_j(unit:TileUnit) -> None:
  h = unit.spacing / (1 + 2 / sqrt3)
  s = h / sqrt3
  t = h / 2
  hexagon = tiling_utils.get_regular_polygon(h, 6)
  sq = affine.translate(tiling_utils.get_regular_polygon(s, 4), 0, (s + h)/2)
  squares = [affine.rotate(sq, a, (0, 0)) for a in range(0, 360, 60)]
  tri = _get_regular_polygon_from_one_side(tiling_utils.get_sides(sq)[0], 3)
  triangles = [affine.rotate(tri, a, (0, 0)) for a in range(0, 360, 60)]
  triangles.append(affine.translate(triangles[3],  s + t,  h + 3*s/2))
  triangles.append(affine.translate(triangles[0], -s - t, -h - 3*s/2))
  polys = [affine.rotate(p, 30, (0, 0))
           for p in [hexagon, *squares, *triangles]]
  unit.setup_vectors(( unit.spacing / 2, unit.spacing * sqrt3/2),
                     ( unit.spacing    , 0),
                     (-unit.spacing / 2, unit.spacing * sqrt3/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))


def _setup_chavey_k(unit:TileUnit) -> None:
  h = unit.spacing / (1 + 2 / sqrt3)
  s = h / sqrt3
  t = h / 2
  triangles1, squares, triangles2 = [], [], []
  hex_c = tiling_utils.get_corners(tiling_utils.get_regular_polygon(h, 6))
  triangles1 = [geom.Polygon([(0, 0), p1, p2])
                for p1, p2 in zip(hex_c[:-1], hex_c[1:], strict = True)]
  squares.append(
    affine.translate(tiling_utils.get_regular_polygon(s, 4), 0, t + s/2))
  squares.extend([affine.rotate(squares[0], a, (0, 0))
                  for a in range(60, 360, 60)])
  polys = [affine.rotate(p, 30, (0, 0)) for p in triangles1 + squares]
  triangles2.append(geom.Polygon([(-s/2, -s - t), (0, -s), (s/2, -s - t)]))
  triangles2.extend([affine.rotate(triangles2[0], a, (0, 0))
                     for a in range(60, 360, 60)])
  triangles2.append(geom.Polygon([(s/2    ,   s   + t),
                                  (s/2 + t, 3*s/2 + t),
                                  (s/2 + t,   s/2 + t)]))
  triangles2.append(affine.rotate(triangles2[-1], 180, (0, 0)))
  polys = polys + triangles2
  polys[-2:] = [affine.rotate(p, 30, (0, 0)) for p in polys[-2:]]
  unit.setup_vectors(( unit.spacing / 2, unit.spacing * sqrt3/2),
                     ( unit.spacing    , 0),
                     (-unit.spacing / 2, unit.spacing * sqrt3/2))
  unit.tiles = gpd.GeoDataFrame(
    data = {"tile_id": list(string.ascii_letters)[:len(polys)]},
    crs = unit.crs,
    geometry = gpd.GeoSeries(polys))
