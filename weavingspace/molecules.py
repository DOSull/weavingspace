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

"""Functions for making sets of polygons around a specified vertex type."""

SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)

def get_base_length(n:int, scale:float = 100.0) -> float|str|None:
  match n:
    case 12:
      return scale
    case 8:
      return scale * (1 + SQRT2) / (2 + SQRT3)
    case 6:
      return scale * SQRT3 / (2 + SQRT3)
    case 4:
      return scale / (2 + SQRT3)
    case 3:
      return scale * SQRT3 / 2 / (2 + SQRT3)
    case _:
      return "Not supported"
  return None


def get_molecule(code:str, scale:float = 100.0) -> list[geom.Polygon]|None:
  fn_name = f"get_molecule_{code.replace('.', '_')}"
  if fn_name in globals():
    return globals()[fn_name](scale)
  return None


def get_molecule_3_3_3_3_3_3(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  return [affine.rotate(triangle, a, (0, 0)) for a in range(0, 360, 60)]


def get_molecule_3_3_3_3_6(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  h = get_base_length(6, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangles = [affine.rotate(triangle, a, (0, 0)) for a in (0, 60, 240, 300)]
  hexagon = affine.translate(tiling_utils.get_regular_polygon(h, 6), -s, 0)
  return [*triangles[:2], hexagon, *triangles[-2:]]


def get_molecule_3_3_3_4_4(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangles = [affine.rotate(triangle, a, (0, 0)) for a in range(0, 180, 60)]
  square = tiling_utils.get_regular_polygon(s, 4)
  squares = [affine.translate(square, dx, -s/2) for dx in (-s/2, s/2)]
  return [*triangles, *squares]


def get_molecule_3_3_4_3_4(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangles = [affine.rotate(triangle, a, (0, 0)) for a in (30, 90, 240)]
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), s/2, s/2)
  squares = [affine.rotate(square, a, (0, 0)) for a in (150, 300)]
  return [*triangles[:2], squares[0], triangles[2], squares[1]]


def get_molecule_3_3_4_12(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  d = get_base_length(12, scale)
  dodecagon = affine.translate(tiling_utils.get_regular_polygon(d, 12), d/2, s/2)
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), -s/2, s/2)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangles = [affine.rotate(triangle, a, (0, 0)) for a in (180, 240)]
  return [dodecagon, square, *triangles]


def get_molecule_3_3_6_6(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  h = get_base_length(6, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangles = [affine.rotate(triangle, a, (0, 0)) for a in (30, 90)]
  hexagon = affine.translate(tiling_utils.get_regular_polygon(h, 6), -s, 0)
  hexagons = [affine.rotate(hexagon, a, (0, 0)) for a in (30, 150)]
  return [*triangles, *hexagons]


def get_molecule_3_4_3_12(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  d = get_base_length(12, scale)
  dodecagon = affine.translate(tiling_utils.get_regular_polygon(d, 12), d/2, s/2)
  dodecagon = affine.rotate(dodecagon, -15, (0, 0))
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), s/2, s/2)
  square = affine.rotate(square, 135, (0, 0))
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangles = [affine.rotate(triangle, a, (0, 0)) for a in (75, 225)]
  return [dodecagon, triangles[0], square, triangles[1]]


def get_molecule_3_4_4_6(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  h = get_base_length(6, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangle = affine.rotate(triangle, 240, (0, 0))
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), s/2, s/2)
  squares = [affine.rotate(square, a, (0, 0)) for a in (60, 150)]
  hexagon = affine.translate(tiling_utils.get_regular_polygon(h, 6), s, 0)
  return [hexagon, *squares, triangle]


def get_molecule_3_4_6_4(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  h = get_base_length(6, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangle = affine.rotate(triangle, 150, (0, 0))
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), s/2, s/2)
  squares = [affine.rotate(square, a, (0, 0)) for a in (60, 210)]
  hexagon = affine.translate(tiling_utils.get_regular_polygon(h, 6), s, 0)
  return [hexagon, squares[0], triangle, squares[1]]


def get_molecule_3_6_3_6(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  h = get_base_length(6, scale)
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangles = [affine.rotate(triangle, a, (0, 0)) for a in (60, 240)]
  hexagon = tiling_utils.get_regular_polygon(h, 6)
  hexagons = [affine.translate(hexagon, dx, 0) for dx in (s, -s)]
  return [hexagons[0], triangles[0], hexagons[1], triangles[1]]


def get_molecule_3_12_12(scale:float) -> list[geom.Polygon]:
  t = get_base_length(3, scale)
  s = get_base_length(4, scale)
  d = get_base_length(12, scale)
  dodecagon = affine.translate(tiling_utils.get_regular_polygon(d, 12), d/2, s/2)
  dodecagons = [affine.rotate(dodecagon, a, (0, 0)) for a in (0, 150)]
  triangle = affine.translate(tiling_utils.get_regular_polygon(t, 3), s/2, t/3)
  triangle = affine.rotate(triangle, 240, (0, 0))
  return [*dodecagons, triangle]


def get_molecule_4_4_4_4(scale:float) -> list[geom.Polygon]:
  s = get_base_length(4, scale)
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), s/2, s/2)
  return [affine.rotate(square, a, (0, 0)) for a in range(0, 360, 90)]


def get_molecule_4_6_12(scale:float) -> list[geom.Polygon]:
  s = get_base_length(4, scale)
  h = get_base_length(6, scale)
  d = get_base_length(12, scale)
  dodecagon = affine.translate(tiling_utils.get_regular_polygon(d, 12), d/2, s/2)
  hexagon = affine.rotate(tiling_utils.get_regular_polygon(h, 6), 30)
  hexagon = affine.translate(hexagon, -h/2, s/2)
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), s/2, s/2)
  square = affine.rotate(square, 210, (0, 0))
  return [dodecagon, hexagon, square]


def get_molecule_4_8_8(scale:float) -> list[geom.Polygon]:
  s = get_base_length(4, scale)
  o = get_base_length(8, scale)
  square = affine.translate(tiling_utils.get_regular_polygon(s, 4), s/2, s/2)
  square = affine.rotate(square, 45, (0, 0))
  octagon = affine.translate(tiling_utils.get_regular_polygon(o, 8), o/2, -s/2)
  octagons = [affine.rotate(octagon, a, (0, 0)) for a in (0, 225)]
  return [octagons[0], square, octagons[1]]


def get_molecule_6_6_6(scale:float) -> list[geom.Polygon]:
  s = get_base_length(4, scale)
  h = get_base_length(6, scale)
  hexagon = affine.translate(tiling_utils.get_regular_polygon(h, 6), s, 0)
  return [affine.rotate(hexagon, a, (0, 0)) for a in range(-30, 330, 120)]