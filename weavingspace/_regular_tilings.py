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

import itertools
import json
import math
import string
from collections.abc import Iterable

import geopandas as gpd
import shapely.geometry as geom

from weavingspace import tiling_utils

"""Manually transcoded from javascript implementation of

JE Soto Sánchez, T Weyrich, LH de Figueiredo, A Medeiros e Sá. 2021. An integer
representation for periodic tilings of the plane with regular polygons Computer
& Graphics 95 69-80. doi: 10.1016/j.cag.2021.01.007

available at https://chequesoto.info/tiling/. Variable names have been
python-ised and in many cases changed to more closely reflect weavingspace's
model of tiling and what we're doing here!
"""

num = int | float # convenience for cleaner typing


class ArchimedeanMesh:
  """Convenience class to store information about a tiling."""

  def __init__(self):
    self.tvec1 = []
    self.tvec2 = []
    self.tiles = []
    self.dual_faces = []


# read tile specs from supplied JSON
# note that there are definitely duplicates
with open("tiles.json") as file:
  s1 = json.load(file)
with open("tiles-2.json") as file:
  s2 = json.load(file)
Symbols = {k.lower(): v for k, v in (s1 | s2).items()}
del s1
del s2


EPS = 1e-4
R2 = 2 ** 0.5
R3 = 3 ** 0.5

TILE_IDS = [
  *string.ascii_lowercase,
  *["".join(x) for x in itertools.product(
    string.ascii_lowercase, string.ascii_lowercase)]
]

# unit vectors in four 30º directions in the first quadrant
X_BASE = 1, R3/2,  0.5, 0
Y_BASE = 0,  0.5, R3/2, 1

# functions for combining arbitrary numbers of arbitrary length vectors
def vsum(*a:Iterable[num]) -> tuple[num]:
  return tuple(sum(_) for _ in zip(*a, strict = True))

def vprod(*a:Iterable[num]) -> tuple[num]:
  if isinstance(a[0], Iterable):
    return tuple(math.prod(_) for _ in zip(*a, strict = True))
  return tuple(a[0] * math.prod(_) for _ in zip(*a[1:], strict = True))

def dot(a:Iterable[num], b:Iterable[num]) -> float:
  return sum(vprod(a, b))

def norm(it:Iterable[num]) -> float:
  return math.sqrt(dot(it, it))

def area(a:tuple[num], b:tuple[num]) -> float:
  A, B = norm(a), norm(b)
  angle = math.acos(dot(a, b) / A / B)
  return A * B * math.sin(angle)

def negate(a:Iterable[num]) -> tuple[num]:
  return tuple(-x for x in a)

def negate_and_extend(a:Iterable[Iterable[num]]) -> tuple[tuple[num]]:
  return (*a, *(negate(x) for x in a))

# Coordinates of the twelve 30º spaced unit vectors expressed as combinations
# of 0, 30, 60, and 90º unit vectors (described in the Soto Sanchéz paper as
# the 12th root of unity in the complex plane). The first four are the basis,
# all others are linear combinations of these four.
BASE_COORDS = [
  ( 1,  0,  0,  0),  # ω^0, i.e., to right along x-axis
  ( 0,  1,  0,  0),  # ω^1, up to right at 30º
  ( 0,  0,  1,  0),  # ω^2, up to right at 60º
  ( 0,  0,  0,  1),  # ω^3, up along y-axis, 90º... and so on
  (-1,  0,  1,  0),  # e.g, this one is left + up to right at 60º, or -ω^0 + ω^2
  ( 0, -1,  0,  1),
]
# last six are 180º rotations, i.e. negated copies of first six
BASE_COORDS = negate_and_extend(BASE_COORDS)

# Offsets of centroids of tiles from a vertex expressed in BASE_COORDS (times 6)
# indexed by number of sides of polygon and heading of incident edge from vertex
# E.g., CENTROID_OFFSETS[4][2] is offset of centroid of a square for edge at 60º
# Six multiplier keeps all numbers integers, allowing precise maths - divide by
# six as needed when Euclidean space coordinates are requested. Used to build
# dual tiles, whose corners are centroids of tiles around each vertex. The six
# multiplier ensures all coordinates are integers allowing exact arithmetic.
CENTROID_OFFSETS = {
  3:  ((2, 0, 2, 0), (0, 2, 0, 2), (-2, 0, 4, 0), 
       (0, -2, 0, 4), (-4, 0, 2, 0), (0, -4, 0, 2)),
  4:  ((3, 0, 0, 3), (-3, 3, 3, 0), (0, -3, 3, 3),
       (-3, 0, 0, 3), (-3, -3, 3, 0), (0, -3, -3, 3)),
  6:  ((0, 0, 6, 0), (0, 0, 0, 6), (-6, 0, 6, 0),
       (0, -6, 0, 6), (-6, 0, 0, 0), (0, -6, 0, 0)),
  12: ((0, 0, 6, 6), (-6, 0, 6, 6), (-6, -6, 6, 6),
       (-6, -6, 0, 6), (-6, -6, 0, 0), (0, -6, -6, 0)),
}
# last six are 180º rotations of first six
CENTROID_OFFSETS = {
  k: negate_and_extend(v) for k, v in CENTROID_OFFSETS.items()
}


def _get_finalised_unit(unit:"TileUnit",
                        tiles:gpd.GeoSeries,
                        mesh:ArchimedeanMesh) -> "TileUnit":
  unit.tiles = gpd.GeoDataFrame(
    geometry = gpd.GeoSeries(tiles), crs = unit.crs,
    data = {"tile_id": TILE_IDS[:len(tiles)]})
  unit.setup_vectors(base_to_point(mesh.tvec1), base_to_point(mesh.tvec2))
  # v1, v2 = base_to_point(mesh.tvec1), base_to_point(mesh.tvec2)
  # sf = unit.spacing / math.sqrt(area(v1, v2))
  # dxdy = vprod(0.5, vsum(v1, v2))
  # unit.tiles.geometry = unit.tiles.geometry.scale(sf, sf, origin = dxdy)
  # unit.setup_vectors(vprod(sf, v1), vprod(sf, v2))
  return unit


def _setup_tile_unit(unit:"TileUnit") -> str|None:
  """Set elements in supplied tile unit by constructing mesh.

  The supplied unit will have code (str) and dual (bool) set, along with spacing
  and crs, which are set as requested.

  Args:
    unit (TileUnit): TileUnit object with `code` and `dual` parameters.

  Returns:
    str|None: str error code is returned if there is a problem.

  """
  mesh = get_mesh(unit.code)
  if isinstance(mesh, str):
    return mesh
  if unit.option is not None:
    if unit.option.lower() in ("base", "composite"):
      polys = [geom.Polygon([base_to_point(xy) for xy in points])
               for points in mesh.tiles]
      if unit.option.lower() == "base":
        unit = _get_finalised_unit(unit, polys, mesh)
      else: # 'composite' i.e., tiles overlaid with their dual
        # turns out it's much easier to do this by pie-slicing regular polygons
        # than making polygons and their dual polygons and intersecting them...
        all_polys = []
        for p in polys:
          n = len(p.exterior.coords) - 1
          slices = list(tiling_utils.get_radially_sliced_polygon(p, n, 1.0))
          all_polys.extend(slices)
        unit = _get_finalised_unit(unit, all_polys, mesh)
    elif unit.option.lower() == "dual":
      polys = [geom.Polygon([base_to_point(vprod(1/6, xy)) for xy in dual])
               for dual in mesh.dual_tiles]
      unit = _get_finalised_unit(unit, polys, mesh)
    else:
      return "Please specify 'base', 'dual' or 'composite' option."
  v1, v2 = base_to_point(mesh.tvec1), base_to_point(mesh.tvec2)
  sf = unit.spacing / math.sqrt(area(v1, v2))
  dxdy = vprod(0.5, vsum(v1, v2))
  unit.tiles.geometry = unit.tiles.geometry.scale(sf, sf, origin = dxdy)
  unit.setup_vectors(vprod(sf, v1), vprod(sf, v2))
  return None


def change_base(a:Iterable[num],
                t1:Iterable[num],
                t2:Iterable[num]) -> Iterable[num]:
  """Transform supplied vector to basis specified by two vectors.

  Args:
    a (Iterable[num]): vector to transform.
    t1 (Iterable[num]): first basis vector.
    t2 (Iterable[num]): second basis vector.

  Returns:
    Iterable[num]: transformed vector.

  """
  det = t1[0] * t2[1] - t1[1] * t2[0]
  d1 =   a[0] * t2[1] -  a[1] * t2[0]
  d2 =  t1[0] *  a[1] - t1[1] *  a[0]
  return d1 / det, d2 / det


def base_to_point(coords:Iterable[int]) -> tuple[float]:
  """Convert supplied integer coordinates to point in Euclidean space.

  Args:
    coords (Iterable[int]): 4-tuple of integer coordinates.

  Returns:
    tuple[float]: corresponding tuple of coordinates in Euclidean space.

  """
  return sum(vprod(coords, X_BASE)), \
         sum(vprod(coords, Y_BASE))


def get_patch(points:Iterable[Iterable[int]],
              t1:Iterable[int],
              t2:Iterable[int],
              n0:int = -1, n1:int = 2) -> set[tuple[int]:int]:
  """Get a set of all points in local patch defined by t1 and t2.

  Args:
    points (Iterable[int]): points in base tile.
    t1 (Iterable[int]): first translation vector.
    t2 (Iterable[int]): second translation vector.
    n0 (int): lower bound on offset tiles to check in patch.
    n1 (int): upper bound on offset tiles to check in patch.

  Returns:
    set[tuple[int]:int]: a set of all points in the surrounding patch
      corresponding to the supplied points displace by the supplied translation
      vectors.

  """
  patch = set()
  for k1 in range(n0, n1 + 1):
    for k2 in range(n0, n1 + 1):
      for point in points:
        # coords are ints so are safe to add to a set
        hashcode = vsum(point, vprod(k1, t1), vprod(k2, t2))
        patch.add(hashcode)
  return patch


def get_star(v:Iterable[int], patch:Iterable[Iterable[int]]) -> list[int]:
  """Return direction in which supplied vertex has neighhours in patch.

  Args:
    v (Iterable[int]): vertex to find neighbours of.
    patch (Iterable[Iterable[int]]): collection of candidate neighbors.

  Returns:
    list[int]: list of directions in which vertex has neighbouring vertices.

  """
  star = []
  for k in range(12):
    candidate = vsum(v, BASE_COORDS[k])
    if candidate in patch:
      star.append(k)
  return star


def interior_angle_to_n_sides(angle:int) -> int:
  """Return polygon degree based on angle supplied as multiple of 30º.

  Args:
    angle (int): number of 30º turns.

  Returns:
    int: number of sides of regular polygon.

  """
  k = abs(angle)
  return (3 if k == 2
    else  4 if k == 3
    else  6 if k == 4
    else 12 if k == 5
    else interior_angle_to_n_sides(12 - k) if k >= 6
    else None)


def get_face(vertex:Iterable[int],
             heading:int,
             n_sides:int) -> tuple[tuple[int],list[tuple[int]]]:
  """Return polygon at vertex with given start heading and number of sides.

  Args:
    vertex (Iterable[int]): start vertex.
    heading (int): heading expressed as number of 30º increments from 0º.
    n_sides (int): number of sides of required polygon.

  Returns:
    list[tuple[int]]: corners of required polygon.

  """
  J = 12 // n_sides
  face = [vertex]
  for angle in range(0, 12, J):
    p = vsum(face[-1], BASE_COORDS[(heading + angle) % 12])
    face.append(p)
  return face[:-1]


def star_to_faces(star:Iterable[int],
                  vertex:Iterable[int]) -> list[tuple[int]]:
  """Return centroids and polygons incident on this vertex.

  Args:
    star (Iterable[int]): directions in 30º increments of edges at this vertex.
    vertex (Iterable[int]): vertex coordinates.

  Returns:
    list[list[tuple[int]]]]: list of list of vertices of tiles incident at this
      vertex.

  """
  faces = []
  for left, right in zip(star, [*star[1:], star[0]], strict = True):
    n = interior_angle_to_n_sides((right - left) % 12)
    if n is not None:
      face = get_face(vertex, left, n)
      faces.append(face)
  return faces


def get_dual_face(star:Iterable[int],
                  centre:Iterable[int]) -> Iterable[Iterable[int]]:
  """Return dual polygon centred on supplied centre based on vertex star.

  Args:
    star (Iterable[int]): vertex on which this dual face is based.
    centre (Iterable[int]): coordinates of vertex.

  Returns:
    Iterable[Iterable[int]]: coordinates (times 6) of dual polygon.

  """
  corners = []
  for left, right in zip(star, [*star[1:], star[0]], strict = True):
    n = interior_angle_to_n_sides((right - left) % 12)
    if n > 0:
      # scale centre by 6 to align with CENTROID_OFFSET coords
      corners.append(vsum(vprod(6, centre), CENTROID_OFFSETS[n][left]))
  return corners


def is_in_parallelogram(p:Iterable[int], t1:list[int], t2:list[int]) -> bool:
  """Check if point p is in parallelogram defined by two vectors.

  Test is conducted by transforming point into the basis space of the vectors,
  and checking if it falls in the unit square in the transformed coordinate
  space.

  Args:
    p (Iterable[int]): point to check.
    t1: list[int]: first translation vector.
    t2: list[int]: second translation vector.

  Returns:
    bool: True if the point lies in the space, False if not.

  """
  U = change_base(base_to_point(p), base_to_point(t1), base_to_point(t2))
  return ((U[0] >= -EPS) and (U[0] < 1 - EPS) and 
          (U[1] >= -EPS) and (U[1] < 1 - EPS))


def get_mesh(tiling_code:str) -> ArchimedeanMesh:
  """Build and return the tiling mesh from the specified code.

  Args:
    tiling_code (str): key of tiling in one of the two JSON catalogs.

  Returns:
    ArchimedeanMesh: desired mesh populated with vectors, tiles, and dual tiles.

  """
  if tiling_code not in Symbols:
    return "Unknown code for regular tiling."
  if tiling_code in ["t1002", "j"]:
    return get_488()
  tvec1 = Symbols[tiling_code]["T1"]
  tvec2 = Symbols[tiling_code]["T2"]
  seed = Symbols[tiling_code]["Seed"]
  vertices = get_patch(seed, tvec1, tvec2, 0, 1)
  patch = get_patch(seed, tvec1, tvec2, -1, 2)
  tiles = [] # tile polygons
  centroids = set()
  dual_tiles = []
  for vertex in vertices:
    star = get_star(vertex, patch)
    these_faces = star_to_faces(star, vertex)
    for face in these_faces:
      centroid = tuple(int(6 * p / len(face)) for p in vsum(*face))
      # times-6 factor keeps coords as integers, and is enforced to stop
      # floating point errors creeping in, and so that centroid can be safely
      # used as dictionary key
      if centroid not in centroids and \
          is_in_parallelogram(vprod(1/6, centroid), tvec1, tvec2):
        centroids.add(centroid)
        tiles.append(face)
    if is_in_parallelogram(vertex, tvec1, tvec2):
      dual_tiles.append(get_dual_face(star, vertex))
  mesh = ArchimedeanMesh()
  mesh.tvec1 = tvec1
  mesh.tvec2 = tvec2
  mesh.tiles = tiles
  mesh.dual_tiles = dual_tiles
  return mesh


def get_488() -> ArchimedeanMesh:
  """Return mesh for the square-octagon tiling.

  4.8.8 vertex only occurs in one regular tiling and does not align with
  30º lattice - so hard code its results.

  Returns:
    ArchimedeanMesh: 4.8.8 mesh, populated with vectors, tiles, and dual tiles.

  """
  mesh = ArchimedeanMesh()
  D = R2 / 2  # key unit vector at 45º is (0.707, 0.707)
  mesh.tvec1 = [  1 + D, 0, 0, 1 + D]
  mesh.tvec2 = [2 + 2*D, 0, 0,     0]
  E, N, NE, SE = [1, 0, 0, 0], [0, 0, 0, 1], [D, 0, 0, D], [D, 0, 0, -D]
  mesh.tiles = [
    [[0, 0, 0, 0], E, vsum(E, N), N],
     [E, vsum(E, N), vsum(E, N, NE), vsum(E, N, NE, E),
      vsum(E, N, NE, E, SE), vsum(E, SE, E, NE), vsum(E, SE, E), vsum(E, SE)]]
  origin      = [      3, 0, 0,       3]
  right, left = [9 + 6*D, 0, 0,       3], [-3 - 6*D, 0, 0,        3]
  up, down    = [      3, 0, 0, 9 + 6*D], [       3, 0, 0, -3 - 6*D]
  mesh.dual_tiles = [[origin, right,  up], [origin, up,    left],
                     [origin, left, down], [origin, down, right]]
  return mesh
