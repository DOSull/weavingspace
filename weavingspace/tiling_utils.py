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

import copy
import itertools
import re
import string
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity as affine
import shapely.geometry as geom
import shapely.ops
from shapely.algorithms import polylabel

if TYPE_CHECKING:
  from collections.abc import Iterable

"""Utility functions for the `weavingspace` module.

Most are geometric convenience functions for commonly applied
operations.
"""

PRECISION = 6
RESOLUTION = 1e-6
TILE_IDS = [
  *string.ascii_lowercase,
  *["".join(x) for x in itertools.product(
    string.ascii_lowercase, string.ascii_lowercase)]
]

def _parse_strand_label(s:str) -> list[str]:
  """Break a strand label specification in to a list of labels.

  Args:
    s (str): see get_strand_ids() for details

  Returns:
    list[str]: list of strand labels.

  """
  # replace multiple open and close parentheses with single parentheses
  clean_s = re.sub("[(]+", "(", re.sub("[)]+", ")", s))
  is_combo = False
  output = []
  current = ""
  for c in clean_s:
    if is_combo:
      if c == ")":
        # end of a combination
        output.append(current)
        current = ""
        is_combo = False
      else:
        current = current + c
    elif c == "(":
      # start of a combination
      is_combo = True
    else:
      output.append(c)
  return output


def get_strand_ids(strands_spec: str) -> tuple[list[str]]:
  """Convert strands specification string to list of lists of strand labels.

  String format is "a|bc|(de)f" where | separates strands in each
  direction and () designates combining labels into a single strand that
  will be sliced lengthwise. Example output:

        "a|bc|(de)f" -> (["a"], ["b", "c"], ["de", "f"])

  Superflous parentheses are removed, but no other error-checks are
  applied.

  Args:
    strands_spec (str): the strands specification to be parsed.

  Returns:
    tuple[str]: tuple of lists of labels for each set of strands.

  """
  strand_ids = [_parse_strand_label(s) for s in strands_spec.split("|")]
  strand_ids = strand_ids if len(strand_ids) == 3 else [*strand_ids, [""]]
  return tuple(strand_ids)


def get_regular_polygon(spacing:float, n:int) -> geom.Polygon:
  r"""Return regular n-gon centered on (0, 0) with height specified.

  n-gon has horizontal base and height given by spacing. Generation is based on
  equally spaced points around a circle with a radius calculated slightly
  differently, depending on whether the number of sides is even or odd.

            Even         Odd
        \  /    \  /      |
         \/   ___\/___    |
         /\      /\      / \
        /  \    /  \    /   \

  The returned polygon is gridified. Note also that shapely silently reorders
  the coordinates clockwise from the lower left corner, regardless of the CCW
  order in which we supply them.

  Args:
    spacing (_type_): required height.
    n (int): number of sides.

  Returns:
    geom.Polygon: required geom.Polygon.

  """
  if n % 2 == 0:
    R = spacing / np.cos(np.pi / n) / 2
  else:
    R = spacing / (1 + np.cos(np.pi / n))
  angle_0 = -np.pi / 2 + np.pi / n
  a_diff = 2 * np.pi / n
  angles = [angle_0 + a_diff * i for i in range(n)]
  corners = [(R * np.cos(angle), R * np.sin(angle)) for angle in angles]
  return geom.Polygon(corners)


def get_corners(shape:geom.Polygon,
                repeat_first:bool = True) -> list[geom.Point]:
  """Return list of geom.Points on boundary of a polygon.

  The first point is optionally repeated. No simplification is carried out
  (e.g. if a line segment has a 'corner' along its length, it is NOT removed;
  see get_clean_polygon for that). Points have precision set to the
  packagedefault tiling_utils.RESOLUTION.

  Args:
    shape (geom.Polygon): polygon whose corners are required.
    repeat_first (bool, optional): if True the first corner is repeated in the
      returned list, if False it is omitted. Defaults to True.

  Returns:
    list[geom.Point]: list of geom.Point vertices of the polygon.

  """
  corners = [geom.Point(pt) for pt in shape.exterior.coords]
  return corners if repeat_first else corners[:-1]


def get_sides(shape:geom.Polygon) -> list[geom.LineString]:
  """Return polygon sides as list of geom.LineStrings.

  Resolution is set to the package default tiling_utils.RESOLUTION. No
  simplification for successive colinear sides is carried out.

  Args:
    shape (geom.Polygon): polygon whose edges are required.

  Returns:
    list[geom.LineString]: list of geom.LineString sides of the polygon.

  """
  corners = get_corners(shape)
  return [geom.LineString([p1, p2])
          for p1, p2 in zip(corners[:-1], corners[1:], strict = True)]


def get_side_lengths(shape:geom.Polygon) -> list[float]:
  """Return list of lengths of polygon sides.

  No simplification for corners along sides is carried out.

  Args:
    shape (geom.Polygon): polygon whose edge lengths are required.

  Returns:
    list[float]: list of side lengths.

  """
  corners = get_corners(shape)
  return [
    p1.distance(p2) for p1, p2 in zip(corners[:-1], corners[1:], strict = True)]


def get_side_bearings(shape:geom.Polygon) -> tuple[float,...]:
  """Return list of angles between sides of polygon and the positive x-axis.

  Angle is in degrees and calculated when proceeding from the first point in
  each side to its end point. This should usually be CW around the polygon.

  Args:
    shape (geom.Polygon): polygon whose side bearings are required.

  Returns:
    tuple[float,...]: tuple of bearings of each edge.

  """
  return tuple([np.degrees(np.arctan2(
    e.coords[1][1] - e.coords[0][1], e.coords[1][0] - e.coords[0][0]))
    for e in get_sides(shape)])


def get_interior_angles(shape:geom.Polygon) -> list[float]:
  """Return angles (in degrees) between successive edges of a polygon.

  No polygon simplification is carried out so some angles may be 180 (i.e. a
  'corner' along a side, such that successive sides are colinear).

  Args:
    shape (geom.Polygon): polygon whose angles are required.

  Returns:
    list[float]: list of angles.

  """
  corners = get_corners(shape, repeat_first = False)
  wrap_corners = corners[-1:] + corners + corners[:1]
  triples = zip(wrap_corners[:-2],
                wrap_corners[1:-1],
                wrap_corners[2:], strict = True)
  return [get_inner_angle(p1, p2, p3) for p1, p2, p3 in triples]


def get_clean_polygon(
      shape:geom.MultiPolygon|geom.Polygon|gpd.GeoSeries,
    ) ->  geom.MultiPolygon|geom.Polygon|gpd.GeoSeries:
  """Return polygon with successive co-linear or very close corners removed.

  Particularly useful for tidying polygons weave tilings that have been
  assembled from multiple 'cells' in the weave grid.

  Args:
    shape (geom.MultiPolygon|geom.Polygon): polygon to clean.

  Returns:
    geom.Polygon: cleaned polygon.

  """
  if isinstance(shape, geom.MultiPolygon):
    return geom.MultiPolygon([get_clean_polygon(p) for p in shape.geoms])
  if isinstance(shape, geom.GeometryCollection):
    elements = [p for p in shape.geoms if isinstance(p, geom.Polygon)]
    if len(elements) > 1:
      return get_clean_polygon(geom.MultiPolygon(elements))
    return get_clean_polygon(elements[0])
  corners = get_corners(shape, repeat_first = False)
  # first remove any 'near neighbour' corners
  distances = get_side_lengths(shape)
  to_remove = [np.isclose(d, 0, rtol = RESOLUTION, atol = 10 * RESOLUTION)
                for d in distances]
  corners = [c for c, r in zip(corners, to_remove, strict = True) if not r]
  # next remove any that are colinear
  p = geom.Polygon(corners)
  corners = get_corners(p, repeat_first = False)
  angles = get_interior_angles(p)
  to_remove = [np.isclose(a,   0, rtol = RESOLUTION, atol = RESOLUTION) or
               np.isclose(a, 180, rtol = RESOLUTION, atol = RESOLUTION)
               for a in angles]
  corners = [c for c, r in zip(corners, to_remove, strict = True) if not r]
  return gridify(geom.Polygon(corners))


def rotate_preserving_order(
    polygon:geom.Polygon,
    angle:float,
    centre:geom.Point = None,
  ) -> geom.Polygon:
  """Return supplied polygon rotated with order of corners preserved.

  Order of polygon corners is not guaranteed to be preserved by shapely.
  affinity.rotate).

  Args:
    polygon (geom.Polygon): polygon to rotate.
    angle (float): desired angle of rotation (in degrees).
    centre (geom.Point): the rotation centre (passed on to shapely.affinity.
      rotate).

  Returns:
    geom.Polygon: rotated polygon.

  """
  ctr = "center" if centre is None else centre
  corners = get_corners(polygon, repeat_first = False)
  return geom.Polygon([affine.rotate(c, angle, ctr) for c in corners])


def ensure_cw(shape:geom.Polygon) -> geom.Polygon:
  """Return the polygon with its outer boundary vertices in clockwise order.

  It is important to note that shapely.set_precision() imposes clockwise order
  on polygons, and since it is used widely throughout theses modules, it makes
  sense to impose this order.

  Args:
    shape (geom.Polygon): the polygon.

  Returns:
    geom.Polygon: the polygon in clockwise order.

  """
  if geom.LinearRing(shape.exterior.coords).is_ccw:
    return shape.reverse()
  return shape


def get_inner_angle(p1:geom.Point, p2:geom.Point, p3:geom.Point) -> float:
  r"""Return angle between line p1-p2 and p2-p3, i.e., the angle A below.

            p2
           / \
          / A \
        p1     p3

  Args:
    p1 (geom.Point): first point.
    p2 (geom.Point): second 'corner' point.
    p3 (geom.Point): third point.

  Returns:
    float: angle in degrees.

  """
  return np.degrees(np.arctan2(p3.y - p2.y, p3.x - p2.x) -
                    np.arctan2(p1.y - p2.y, p1.x - p2.x)) % 360


def is_regular_polygon(shape:geom.Polygon) -> bool:
  """Test if supplied polygon is regular (i.e. equal sides and angles).

  Args:
    shape (geom.Polygon): polygon to test.

  Returns:
    bool: True if polygon is regular, False if not.

  """
  side_lengths = get_side_lengths(shape)
  angles = get_interior_angles(shape)
  return all(np.isclose(side_lengths, side_lengths[0])) \
     and all(np.isclose(angles, angles[0]))


def is_tangential(shape:geom.Polygon) -> bool:
  """Determine if supplied polygon is tangential.

  A tangential polygon can have a circle inscribed tangential to all its sides.

  NOTE: not currently used but retained in case incentre is restored to use
  'home made' code rather than polylabel.polylabel.

  Note that this will fail for polygons with successive colinear sides, meaning
  that polygons should be fully simplified...

  Args:
    shape (geom.Polygon): polygon to test.

  Returns:
    bool: True if polygon is tangential, False if not.

  """
  if is_regular_polygon(shape):
    return True
  side_lengths = get_side_lengths(shape)
  n = len(side_lengths)
  if n % 2 == 1:
    if n == 3: #triangles are easy!
      return True
    # odd number of sides there is a nice solvable system  of equations see
    # https://math.stackexchange.com/questions/4065370/tangential-polygons-conditions-on-edge-lengths
    #
    # TODO: this is not reliable for reasons I don't fully understand. There is
    # a corresponding crude catch exception in incentre... but this needs a
    # more complete investigation
    mat = np.identity(n) + np.roll(np.identity(n), 1, axis = 1)
    fractions = np.linalg.inv(mat) @ side_lengths / side_lengths
    if not(np.isclose(np.mean(fractions),
                      0.5, rtol= RESOLUTION, atol = RESOLUTION)):
      return False
    ones = [np.isclose(f, 1, rtol = RESOLUTION, atol = RESOLUTION)
            for f in fractions]
    zeros = [np.isclose(f, 0, rtol = RESOLUTION, atol = RESOLUTION)
             for f in fractions]
    negatives = [f < 0 for f in fractions]
    return not (any(ones) or any(zeros) or any(negatives))
  if n == 4:
    # for quadrilaterals odd and even side lengths are equal
    return np.isclose(sum(side_lengths[0::2]),
                      sum(side_lengths[1::2]), rtol = RESOLUTION)
  # other even numbers of sides... hard to avoid brute force
  bisectors = [get_angle_bisector(shape, i) for i in range(n)]
  intersects = [b1.intersection(b2) for b1, b2 in
                zip(bisectors, bisectors[1:] + bisectors[:1], strict = False)
                if b1.intersects(b2)]
  distances = [i1.distance(i2) for i1, i2 in
               zip(intersects, intersects[1:] + intersects[:1],
                   strict = False)]
  return all([d <= 2 * RESOLUTION for d in distances])  # noqa: C419


def get_incentre(shape:geom.Polygon) -> geom.Point:
  """Get a polygon incentre.

  See https://en.wikipedia.org/wiki/Incenter.

  This method relies on the polygon being tangential, i.e. there is an
  inscribed circle to which all sides of the polygon are tangent. It will work
  on all the polygons encountered in the Laves tilings, but is not guaranteed
  to work on all polygons.

  Given that the polygon is tangential, the radius of the inscribed circle is
  the [apothem of the polygon](https://en.wikipedia.org/wiki/Apothem) given by
  2 x Area / Perimeter. We apply a parallel offset of this size to two sides of
  the polygon and find their intersection to determine the centre of the circle
  givng the incentre of the polygon.

  Args:
    shape (geom.Polygon): the polygon.

  Returns:
    geom.Point: the incentre of the polygon.

  """
  if is_regular_polygon(shape):
    return shape.centroid
  # default tolerance of 1.0 is much too high for our purposes
  return polylabel.polylabel(shape, tolerance = np.sqrt(shape.area) * 1e-9)


def get_incircle(shape:geom.Polygon) -> geom.Polygon:
  """Return incircle of supplied polygon.

  Args:
    shape(geom.Polygon): polygon for which incircle is required.

  Returns:
    geom.Polygon: a polygon representation of the incircle

  """
  c = get_incentre(shape)
  r = min([c.distance(e) for e in get_sides(shape)])
  return c.buffer(r)


def get_apothem(shape:geom.Polygon) -> float:
  """Return length of the apothem of supplied polygon.

  Args:
    shape(geom.Polygon): polygon for which apothem length is required.

  Returns:
    float: an estimate of the apothem length.

  """
  if is_regular_polygon(shape):
    return 2 * shape.area / geom.LineString(shape.exterior.coords).length
  c = polylabel.polylabel(shape, np.sqrt(shape.area) * 1e-9)
  return min([c.distance(e) for e in get_sides(shape)])


def get_angle_bisector(shape:geom.Polygon, v:int = 0) -> geom.LineString:
  """Return angle bisector of specified corner of supplied polygon.

  Args:
    shape (geom.Polygon): the polygon
    v (int, optional): index of the corner whose bisector is required.
    Defaults to 0.

  Returns:
    geom.LineString: line which bisects the specified corner.

  """
  pts = get_corners(shape, repeat_first = False)
  n = len(pts)
  p0 = pts[v % n]
  p1 = pts[(v + 1) % n]
  p2 = pts[(v - 1) % n]
  d01 = p0.distance(p1)
  d02 = p0.distance(p2)
  if d01 < d02:
    p2 = geom.LineString([p0, p2]).interpolate(d01)
  else:
    p1 = geom.LineString([p0, p1]).interpolate(d02)
  m = geom.Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
  # we have to scale this to much larger in case of an angle near 180
  return affine.scale(geom.LineString([p0, m]), 10, 10, origin = p0)


def _get_interior_vertices(tiles:gpd.GeoDataFrame) -> gpd.GeoSeries:
  """Return points not on outer boundary of supplied set of tiles.

  Args:
    tiles (gpd.GeoDataFrame): a set of polygons.

  Returns:
    gpd.GeoSeries: interior vertices of the set of polygons.

  """
  tiles:gpd.GeoSeries = gridify(tiles.geometry)
  uu:geom.Polygon = safe_union(tiles, as_polygon = True)
  interior_pts = set()
  for tile in tiles:
    for corner in tile.exterior.coords:
      if uu.contains(geom.Point(corner).buffer(1e-3, cap_style = "square")):
        interior_pts.add(corner)
  return gridify(gpd.GeoSeries([geom.Point(p) for p in interior_pts]))


def gridify(
    gs:gpd.GeoSeries | gpd.GeoDataFrame | shapely.Geometry,
  ) -> gpd.GeoSeries | gpd.GeoDataFrame | shapely.Geometry:
  """Return supplied GeoSeries rounded to tiling_utils.RESOLUTION.

  Args:
    gs (gpd.GeoSeries): geometries to gridify.
    precision (int, optional): digits of precision. Defaults to 6.

  Returns:
    gpd.GeoSeries: the rounded geometries.

  """
  if isinstance(gs, shapely.Geometry) :
    return shapely.set_precision(gs, grid_size = RESOLUTION)
  if isinstance(gs, gpd.GeoSeries):
    return gpd.GeoSeries(
      [shapely.set_precision(g, grid_size = RESOLUTION) for g in list(gs)],
      crs = gs.crs)
  if isinstance(gs, gpd.GeoDataFrame):
    gs.geometry = gpd.GeoSeries(
      [shapely.set_precision(g, grid_size = RESOLUTION)
       for g in list(gs.geometry)], crs = gs.crs)
    return gs
  return None


def get_dual_tile_unit(unit) -> gpd.GeoDataFrame:
  """Convert supplied TileUnit to candidate GeoDataFrame of its dual TileUnit.

  NOTE: this is complicated and not remotely guaranteed to work!
  a particular issue is that where to place the vertices of the faces
  of the dual with respect to the tiles in the original is ill-defined.
  This is because the dual process is topologically not metrically defined,
  so that exact vertex locations are ambiguous. Tiling duality is defined in
  Section 4.2 of Grunbaum B, Shephard G C, 1987 _Tilings and Patterns_ (W. H.
  Freeman and Company, New York)

  NOTE: In general, this method will work only if all supplied tiles are
  regular polygons. A known exception is if the only non-regular polygons are
  triangles.

  NOTE: 'clean' polygons are required. If supplied polygons have messy
  vertices with multiple points where there is only one proper point, bad
  things are likely to happen! Consider using `clean_polygon()` on the
  tile geometries.

  Because of the above limitations, we only return a GeoDataFrame
  for inspection. However some `weavingspace.tile_unit.TileUnit` setup
  methods in `weavingspace.tiling_geometries` use this method, where we are
  confident the returned dual is valid.

  Args:
    unit (TileUnit): the tiling for which the dual is required.

  Returns:
    gpd.GeoDataFrame: GeoDataFrame that could be the tiles attribute for
      a TileUnit of the dual tiling.

  """
  # get a local patch of this Tiling
  local_patch:gpd.GeoDataFrame = unit.get_local_patch(r = 1, include_0 = True)
  # Find the interior points of these tiles - these will be guaranteed
  # to have a sequence of surrounding tiles incident on them
  interior_pts = _get_interior_vertices(local_patch)
  # Compile a list of the polygons incident on the interior points
  cycles = [{poly_id for poly_id, p in enumerate(local_patch.geometry)
             if pt.distance(p) < RESOLUTION * 2} for pt in interior_pts]
  dual_faces = []
  for cycle in cycles:
    ids, pts = [], []
    for poly_id in cycle:
      ids.append(local_patch.tile_id[poly_id])
      poly = local_patch.geometry[poly_id]
      pts.append(get_incentre(poly))
    # sort them into CW order so they are well formed
    sorted_coords = _sort_cw([(pt.x, pt.y, ID)
                              for pt, ID in zip(pts, ids, strict = True)])
    dual_faces.append(
      (geom.Polygon([(pt_id[0], pt_id[1]) for pt_id in sorted_coords]),
       "".join([pt_id[2] for pt_id in sorted_coords])))
  # ensure the resulting face centroids are inside the original tile
  # displaced a little to avoid uncertainties at corners/edges
  # This is a guess at something that _could_ work, which turned out well!
  dual_faces = [(f, ID) for f, ID in dual_faces
                if affine.translate(
                  unit.get_prototile_from_vectors().loc[0, "geometry"],
                  RESOLUTION * 10, RESOLUTION * 10).contains(f.centroid)]
  gdf = gpd.GeoDataFrame(
    data = {"tile_id": [f[1] for f in dual_faces]}, crs = unit.crs,
    geometry = gpd.GeoSeries([f[0] for f in dual_faces]))
  # ensure no duplicates
  gdf = gdf.dissolve(by = "tile_id", as_index = False).explode(
    index_parts = False, ignore_index = True)
  gdf.tile_id = _relabel(gdf.tile_id)
  return gdf


def _relabel(data:Iterable) -> list:
  """Return supplied data reassigned with unique values from TILE_IDS.

  Args:
    data (Iterable): the data to relabel

  Returns:
    list: the reassigned data

  """
  new_data = {}
  d_count = 0
  for d in data:
    if d not in new_data:
      new_data[d] = TILE_IDS[d_count]
      d_count = d_count + 1
  return [new_data[d] for d in data]


def _sort_cw(
  pts_ids:list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
  """Sort supplied x,y,ID tuple into CW order relative to their mean centre.

  Args:
    pts_ids (list[tuple[float, float, str]]): A tuple of a pair of
      floats and a string.

  Returns:
    list: a list in the same format as supplied sorted into
      clockwise order of the point locations.

  """
  x = [p[0] for p in pts_ids]
  y = [p[1] for p in pts_ids]
  cx, cy = np.mean(x), np.mean(y)
  dx = [_ - cx for _ in x]
  dy = [_ - cy for _ in y]
  angles = [np.arctan2(dy, dx) for dx, dy in zip(dx, dy, strict = True)]
  d = dict(zip(angles, pts_ids, strict = True))
  return [pt_id for angle, pt_id in sorted(d.items(), reverse = True)]


def write_map_to_layers(gdf:gpd.GeoDataFrame,
                        fname:str = "output.gpkg",
                        tile_var:str = "tile_id") -> None:
  """Write supplied GeoDataFrame to a GPKG file.

  Geopackage file layers will be based on the `tile_var` attribute.

  Args:
    gdf (gpd.GeoDataFrame): the GeoDataFrame.
    fname (str, optional): filename to write.
    tile_var (str, optional): the attribute to use to separate
      output file into layers. Defaults to "tile_id".

  """
  grouped = gdf.groupby(tile_var, as_index = False)
  for e in pd.Series.unique(gdf[tile_var]):
    grouped.get_group(e).to_file(fname, layer = e, driver = "GPKG")


def get_collapse_distance(geometry:geom.Polygon) -> float:
  """Return distance under which polygon will disappear if negatively buffered.

  Performs a binary search between an upper bound based on the radius of circle
  of equal area to the polygon, and 0.

  Args:
    geometry (geom.Polygon): the polygon.

  Returns:
    float: its collapse distance.

  """
  if is_regular_polygon(geometry):
    return get_apothem(geometry)
  radius = np.sqrt(geometry.area / np.pi)
  lower = 0
  upper = radius
  delta = 1e12
  stop_delta = radius / 1000
  while delta > stop_delta:
    new_r = (lower + upper) / 2
    if geometry.buffer(-new_r).area > 0:
      lower = new_r
      delta = upper - new_r
    else:
      upper = new_r
      delta = new_r - lower
  return new_r


def get_largest_polygon(polygons:gpd.GeoSeries) -> gpd.GeoSeries:
  """Return largest polygon in a GeoSeries as a GeoSeries of one polygon.

  NOT CURRENTLY USED ANYWHERE IN MODULE.

  Args:
    polygons (gpd.GeoSeries): the set of polygons to pick from.

  Returns:
    gpd.GeoSeries: the largest polygon.

  """
  actual_polygons = [p for p in polygons.geometry
                     if isinstance(p, (geom.Polygon, geom.MultiPolygon))]
  areas = sorted(actual_polygons, key = lambda x: x.area, reverse = True)
  return gpd.GeoSeries(areas[:1])


def touch_along_an_edge(p1:geom.Polygon, p2:geom.Polygon) -> bool:
  """Test if two polygons touch along an edge.

  Checks that the intersection area of the two polygons buffered by
  a small amount is large enough to indicate that they neighbour at more
  than a corner.

  Args:
    p1 (geom.Polygon): First polygon
    p2 (geom.Polygon): Second polygon

  Returns:
    bool: True if they neighbour along an edge

  """
  return (
    p1.buffer(RESOLUTION, join_style = "mitre", cap_style = "square")
      .intersection(p2.buffer(RESOLUTION, join_style = "mitre",
                                          cap_style = "square"))
      .area > 16 * RESOLUTION ** 2)


def get_width_height_left_bottom(
    gs:gpd.GeoSeries,
  ) -> tuple[float,float,float,float]:
  """Return width, height, left and bottom limits of a GeoSeries.

  Args:
    gs (geopandas.GeoSeries): GeoSeries for which limits are required.

  Returns:
    tuple: four float values of width, height, left and bottom of gs.

  """
  bb = gs.total_bounds
  return (bb[2] - bb[0], bb[3] - bb[1], bb[0], bb[1])


def get_bounding_ellipse(
    shapes:gpd.GeoSeries,
    mag:float = 1.0,
  ) -> gpd.GeoSeries:
  """Return an ellipse containing the supplied shapes.

  The method used is to calculate the size of square that would contain
  the shapes, if they had an aspect ratio 1, then stretch the circle in
  the x, y directions according to the actual aspect ratio of the shapes.

  Args:
    shapes (gpd.GeoSeries): the shapes to be contained.
    mag (float, optional): optionally increase the size of the returned
      ellipse by this scale factor. Defaults to 1.0.

  Returns:
    gpd.GeoSeries: the set of shapes.

  """
  width, height, left, bottom = get_width_height_left_bottom(shapes)
  c = geom.Point(left + width / 2, bottom + height / 2)
  r = min(width, height) * np.sqrt(2)
  circle = [c.buffer(r)]
  return gpd.GeoSeries(circle, crs = shapes.crs).scale(
    width / r * mag / np.sqrt(2), height / r * mag / np.sqrt(2), origin = c)


def get_tiling_edges(tiles:gpd.GeoSeries) -> gpd.GeoSeries:
  """Return linestring GeoSeries from supplied polygon GeoSeries.

  This is used to allow display of edges of tiles in legend when they are
  masked by an ellipse (if we instead clip polygons then the ellipse edge
  will also show in the result.)

  Args:
    tiles (gpd.GeoSeries): Polygons to convert.

  Returns:
    gpd.GeoSeries: LineStrings from the supplied Polygons.

  """
  tiles = tiles.explode(ignore_index = True)
  edges = [geom.LineString(p.exterior.coords) for p in tiles]
  return gpd.GeoSeries(edges, crs = tiles.crs)


def get_polygon_sector(
    shape:geom.Polygon,
    start:float = 0.0,
    end:float = 1.0,
  ) -> geom.Polygon:
  """Return a sector of the provided Polygon.

  The returned sector is a section of the polygon boundary between the
  normalized start and end positions, and including the polygon centroid.
  Should (probably) only be applied to convex polygons.

  Args:
    shape (geom.Polygon): the Polygon.
    start (float): normalized start position along the boundary. Defaults to
      0.
    end (float): normalized start position along the boundary. Defaults to
      1.

  Returns:
    geom.Polygon: the requested polygon sector.

  """
  if start == end:
    # must return a null polygon since the calling context
    # expects to get something back... which most likely
    # is needed to align with other data
    return geom.Polygon()
  if start * end < 0:
    # either side of 0/1 so assume required sector includes 0
    e1, e2 = min(start, end), max(start, end)
    arc1 = shapely.ops.substring(geom.LineString(shape.exterior.coords),
                                 np.mod(e1, 1), 1, normalized = True)
    arc2 = shapely.ops.substring(geom.LineString(shape.exterior.coords),
                                 0, e2, normalized = True)
    sector = geom.Polygon(
      [shape.centroid, *list(arc1.coords), *list(arc2.coords)[1:]])
  else:
    arc = shapely.ops.substring(geom.LineString(shape.exterior.coords),
                                start, end, normalized = True)
    sector = geom.Polygon([shape.centroid, *list(arc.coords)])
  return sector


def repair_polygon(
    polygon:geom.Polygon|geom.MultiPolygon|gpd.GeoSeries,
    shrink_then_grow:bool = True,
  ) -> geom.Polygon|gpd.GeoSeries:
  """Repair a polyon or GeoSeries by double buffering.

  This is method is often unofficially recommended (on stackexchange etc.)
  even in the shapely docs, to resolve topology issues and extraneous
  additional vertices appearing when spatial operations are repeatedly
  applied.

  Optionally the buffer may be applied in the opposite order (i.e. grow then
  shrink). This operation may also convert a MultiPolygon that has some 'stray'
  parts to a Polygon.

  Args:
    polygon(geom.Polygon|gpd.GeoSeries): Polygon or GeoSeries to clean.
    res (float, optional): buffer size to use. Defaults to 1e-3.
    shrink_then_grow (bool, optional): if True the negative buffer is
      applied first, otherwise the buffer operations are applied in
      reverse. Defaults to True.

  Returns:
    geom.Polygon|gpd.GeoSeries: the cleaned Polygon or GeoSeries.

  """
  if type(polygon) is gpd.GeoSeries:
    return gpd.GeoSeries([repair_polygon(p, shrink_then_grow)
                          for p in polygon]).set_crs(polygon.crs)
  if polygon is None:
    return None
  if shrink_then_grow:
    return polygon.buffer(
      -RESOLUTION * 10, join_style = "mitre", cap_style = "square").buffer(
       RESOLUTION * 10, join_style = "mitre", cap_style = "square")
  return polygon.buffer(
     RESOLUTION * 10, join_style = "mitre", cap_style = "square").buffer(
    -RESOLUTION * 10, join_style = "mitre", cap_style = "square")


def safe_union(
    gs:gpd.GeoSeries,
    as_polygon:bool = False,
  ) -> gpd.GeoSeries|geom.Polygon:
  """Union supplied GeoSeries buffering to avoid gaps and internal edges.

  Optionally returns a Polygon or a GeoSeries.

  NOTE: APPEARS NOT TO BE VERY SAFE... find workarounds for the few occasions
  on which it is used.

  Frequently when unioning polygons that are ostensibly adjacent 'rogue'
  internal boundaries remain in the result. We can avoid this by buffering the
  polygons before unioning them, then reversing the buffer on the unioned
  shape.

  Args:
    gs (gpd.GeoSeries): the Polygons to union.
    res (float, optional): size of the buffer to use. Defaults to 1e-3.
    as_polygon (bool, optional): if True returns a Polygon, otherwise
      returns a one Polygon GeoSeries. Defaults to False.

  Returns:
    gpd.GeoSeries|geom.Polygon: the resulting union of supplied
      polygons.

  """
  r = RESOLUTION * 10
  union = gpd.GeoSeries(
    [p.buffer(r, join_style = "mitre", cap_style = "square") for p in gs],
    crs = gs.crs).union_all().buffer(-r, join_style = "mitre",
                                         cap_style = "square")
  if as_polygon:
    return union
  return gpd.GeoSeries([union], crs = gs.crs)


def get_translation_transform(dx:float, dy:float) -> tuple[float,...]:
  """Return shapely affine transform tuple for a translation.

  Args:
      dx (float): translation distance in x direction.
      dy (float): translation distance in y direction.

  Returns:
    list[float]: a six item list of floats, per the shapely.affinity.
    affine_transform method, see
      https://shapely.readthedocs.io/en/stable/manual.html#affine-transformations

  """
  return (1, 0, 0, 1, dx, dy)


def get_rotation_transform(
    angle:float, centre:tuple[float,...]|None = None) -> tuple[float,...]:
  """Return the shapely affine transform tuple for a rotation.

  Rotation is optionally about a supplied centre point.

  Args:
    angle (float): the angle of rotation (in degrees).
    centre (tuple[float], optional): An option centre location. Defaults to
    None, which will in turn be converted to (0, 0).

  Returns:
    list[float]: a six item list of floats, per the shapely.affinity.
      affine_transform method, see
        https://shapely.readthedocs.io/en/stable/manual.html#affine-transformations

  """
  if centre is None or np.allclose((0, 0), centre, atol = RESOLUTION):
    a = np.radians(angle)
    return (np.cos(a), -np.sin(a), np.sin(a), np.cos(a), 0, 0)
  dx, dy = centre
  t1 = get_translation_transform(-dx, -dy)
  r = get_rotation_transform(angle)
  t2 = get_translation_transform(dx, dy)
  return combine_transforms([t1, r, t2])


def get_reflection_transform(
    angle:float, centre:tuple[float,...]|None = None) -> tuple[float,...]:
  """Return a shapely affine reflection transform.

  Reflection is in a line at the specified angle, optionally through a
  specified centre point.

  Args:
    angle (float): angle to the x-axis of the line of reflection.
    centre (tuple[float], optional): point through which the line of
      reflection passes. Defaults to None, which
      will in turn be converted to (0, 0).

  Returns:
    list[float]: a six item list of floats, per the shapely.affinity.
      affine_transform method, see
        https://shapely.readthedocs.io/en/stable/manual.html#affine-transformations

  """
  if centre is None or np.allclose((0, 0), centre, atol = RESOLUTION):
    A = 2 * np.radians(angle)
    return (np.cos(A), np.sin(A), np.sin(A), -np.cos(A), 0, 0)
  dx, dy = centre
  t1 = get_translation_transform(-dx, -dy)
  r = get_reflection_transform(angle)
  t2 = get_translation_transform(dx, dy)
  return combine_transforms([t1, r, t2])


def combine_transforms(transforms:list[tuple[float,...]]) -> tuple[float,...]:
  """Return shapely affine transform that combines supplied list of transforms.

  Transforms are applied in the order of the supplied list

  Args:
    transforms (list[list[float]]): sequence of transforms to combine.

  Returns:
    list[float]: a transform tuple combining the supplied transforms applied
      in order, see
        https://shapely.readthedocs.io/en/stable/manual.html#affine-transformations

  """
  result = np.identity(3)
  for t in transforms:
    result = as_numpy_matrix(t) @ result
  return as_shapely_transform(result)


def reverse_transform(transform:tuple[float,...]) -> tuple[float,...]:
  """Return inverse shapely affine transform of the supplied transform.

  Args:
    transform (list[float]): the transform for which the inverse is desired.

  Returns:
    list[float]: shapely affine transform tuple that will invert the supplied
      transform.

  """
  return as_shapely_transform(
    np.linalg.inv(as_numpy_matrix(transform)))


def as_shapely_transform(arr:np.ndarray) -> tuple[float,...]:
  """Return shapely affine transform equivalent to supplied numpy matrix.

  The numpy matrix is a conventional augmented (i.e. 3x3) affine transform
  matrix.

  Args:
    arr (np.array): augmented affine transform matrix of the desired
      transform.

  Returns:
    list[float]: desired shapely affine transform list of floats.

  """
  return (arr[0][0], arr[0][1], arr[1][0], arr[1][1], arr[0][2], arr[1][2])


def as_numpy_matrix(transform:tuple[float,...]) -> np.ndarray:
  """Convert shapely affine transform to augmented affine transform matrix.

  Args:
    transform (list[float]): the transform in shapely format.

  Returns:
    np.array: the transform in numpy matrix format.

  """
  return np.array([[transform[0], transform[1], transform[4]],
                   [transform[2], transform[3], transform[5]],
                   [           0,            0,            1]])


class StraightLine(NamedTuple):
  """Straight line in canonical Ax+By+C=0 form."""

  A: float
  B: float
  C: float


def get_straight_line(
    p1:geom.Point, p2:geom.Point,
    perpendicular:bool = False,
  ) -> StraightLine:
  """Return StraightLine p1-p2, optionally perpendicular through the midpoint.

  Args:
    p1 (geom.Point): First point.
    p2 (geom.Point): Second point
    perpendicular (bool, optional): If True the perpendicular bisector is
      returned. Defaults to False.

  Returns:
    StraightLine: _description_

  """
  if perpendicular:
    ls = affine.rotate(geom.LineString([p1, p2]), 90)
    pts = list(ls.coords)
    x1, y1 = pts[0]
    x2, y2 = pts[1]
  else:
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
  return StraightLine(y1 - y2, x2 - x1, x1 * y2 - x2 * y1)


def get_intersection(line1:StraightLine,
                     line2:StraightLine) -> geom.Point|None:
  """Return point of intersection of straight lines.

  Args:
    line1 (StraightLine): First straight line.
    line2 (StraightLine): Second straight line.

  Returns:
    geom.Point|None: Point is returned if the lines intersect, otherwise None.

  """
  x_set, y_set = False, False
  denominator = line1.A * line2.B - line2.A * line1.B
  if np.isclose(line1.A, 0, atol = 1e-4, rtol = 1e-4):
    y = -line1.C / line1.B
    y_set = True
  elif np.isclose(line2.A, 0, atol = 1e-4, rtol = 1e-4):
    y = -line2.C / line2.B
    y_set = True
  if np.isclose(line1.B, 0, atol = 1e-4, rtol = 1e-4):
    x = -line1.C / line1.A
    x_set = True
  elif np.isclose(line2.B, 0, atol = 1e-4, rtol = 1e-4):
    x = -line2.C / line2.A
    x_set = True
  if np.isclose(denominator, 0, atol = 1e-4, rtol = 1e-4):
    return None
  x = x if x_set else (line1.B * line2.C - line2.B * line1.C) / denominator
  y = y if y_set else (line1.C * line2.A - line2.C * line1.A) / denominator
  return geom.Point(x, y)


def are_parallel(v1:Iterable, v2:Iterable) -> np.bool:
  """Report true if supplied vectors are parallel.

  Args:
    v1 (Iterable): sequence of floats.
    v2 (Iterable): sequences of floats.

  Returns:
    bool: True if the supplied vectors are parallel.

  """
  return np.isclose(np.linalg.det(np.array([v1, v2])), 0)


def are_orthogonal(v1:Iterable, v2:Iterable) -> np.bool:
  """Report true if supplied vectors are orthogonal.

  Args:
    v1 (Iterable): sequence of floats.
    v2 (Iterable): sequences of floats.

  Returns:
    bool: True if the supplied vectors are orthogonal.

  """
  l1 = np.hypot(*v1)
  l2 = np.hypot(*v2)
  return np.isclose(np.hypot(v2[0] - v1[0], v2[1] - v1[1]),
                    np.hypot(l1, l2))

def geometry_matches(geom1:geom.Polygon, geom2:geom.Polygon) -> bool:
  """Report true if supplied polygons are identical.

  Args:
    geom1 (geom.Polygon): first polygon to check.
    geom2 (geom.Polygon): second polygon to check.

  Returns:
    bool: True if polygons match, else False.

  """
  a = geom1.area
  return np.isclose(a, geom1.intersection(geom2).area,
                    rtol = RESOLUTION * 100,
                    atol = RESOLUTION * 100)


def get_counts(seq:Iterable) -> defaultdict[int]:
  """Count distinct items in supplied iterable and compile to dictionary.

  Args:
    seq (Iterable): iterable to count items in.

  Returns:
    defaultdict[int]: dictionary of counts keyed by item.

  """
  counts = defaultdict(int)
  for x in seq:
    counts[x] += 1
  return counts


def minimise_weave_unit(tileable: "WeaveUnit") -> "WeaveUnit":
  """Return a minimised version of supplied WeaveUnit.

  The WeaveUnit constructor often produces non-minimal units that could be
  simplified further. This method attempts to reduce the supplied unit to a
  minimal form.

  Args:
    tileable (WeaveUnit): the WeaveUnit to minimise.

  Returns:
    WeaveUnit: a new WeaveUnit that will yield the same pattern when tiled but
      which consists of fewer elements.

  """
  n_axes = 3 if tileable.weave_type in ["cube", "hex"] else 2
  new_unit = copy.deepcopy(tileable)
  # if any tile_id is singleton then already minimal
  if any(n == 1 for n in get_counts(tileable.tiles.tile_id).values()):
    return new_unit
  # for biaxial only proceed if #strands is same in each direction
  n_strands = [len(s) for s in tileable.strands.split("|")]
  if n_axes == 2 and n_strands[0] != n_strands[1]:
    return new_unit
  # next get to grips with different sized elements due to aspect < 1
  areas = [p.area for p in new_unit.tiles.geometry]
  min_area = min(areas)
  relative_areas = [int(a / min_area) for a in areas]
  pieces_by_id_and_area = defaultdict(list)
  for ID, r_area, tile in zip(new_unit.tiles.tile_id,
                              relative_areas,
                              new_unit.tiles.geometry, strict = True):
    pieces_by_id_and_area[(ID, r_area)].append(tile)
  v_ij = defaultdict(list)
  for group in pieces_by_id_and_area.values():
    for i, tile_i in enumerate(group):
      for j, tile_j in enumerate(group):
        if i < j:
          v = (tile_j.centroid.x - tile_i.centroid.x,
              tile_j.centroid.y - tile_i.centroid.y)
          if geometry_matches(
            tile_j, affine.translate(tile_i, v[0], v[1])):
              v_ij[v].append((i, j))
              v_ij[v].append((j, i))
  v_ij = dict(sorted(v_ij.items(), key = lambda x: len(x[1]), reverse = True))
  v_ij = list(v_ij.keys())
  combos = itertools.combinations(v_ij, n_axes)
  if n_axes == 3:
    # three vectors that sum to (0, 0)
    result = [c for c in combos
              if np.isclose(sum([x for x, y in c]), 0, atol = 1e-6) and
                 np.isclose(sum([y for x, y in c]), 0, atol = 1e-6)]
    result = sorted(result,
                    key = lambda vecs: sum([dx**2 + dy**2 for dx, dy in vecs]))
  else:
    # two non-parallel vectors
    result = [c for c in combos if are_orthogonal(*c) and
              np.isclose(np.hypot(*c[0]), np.hypot(*c[1]))]
  new_unit.setup_vectors(*result[0])
  new_unit.prototile = new_unit.get_prototile_from_vectors()
  ptile = new_unit.prototile.geometry.loc[0]
  new_tiles = [
    (ID, t) for ID, t in
    zip(new_unit.tiles.tile_id, new_unit.tiles.geometry, strict = True)
    if t.intersection(ptile).area > 0]
  new_unit.tiles = gpd.GeoDataFrame({
    "geometry": gpd.GeoSeries([t[1] for t in new_tiles]),
    "tile_id": [t[0] for t in new_tiles]})
  new_unit._setup_regularised_prototile()
  return new_unit