"""
MIT License

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

import itertools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity as affine
import shapely.geometry as geom
import shapely.ops

from weavingspace import (
  Loom,
  Tileable,
  TileShape,
  WeaveGrid,
  tiling_utils,
  weave_matrices,
)

if TYPE_CHECKING:
  from collections.abc import Iterable

"""The `WeaveUnit` subclass of `weavingspace.tileable.Tileable`.

Implements tileable geometric patterns constructed by specifying 2- and 3-axial
weaves.

Examples:
  Explain usage here...

"""

@dataclass
class WeaveUnit(Tileable):
  """Extend Tileable to allow for tiles that appear like woven patterns."""

  weave_type:str = "plain"
  """type of weave pattern, one of `plain`, `twill`, `basket`, `cube`, `hex` or
  `this`. Defaults to `plain`."""
  aspect:float = 1.
  """width of strands relative to the `spacing`. Defaults to 1.0."""
  n:int|tuple[int] = (2, 2)
  """number of over-under strands in biaxial weaves. Only one item is
  required in a plain weave. Twill and basket patterns expect an even number of
  entries in the tuple."""
  strands:str = "a|b|c"
  """specification of the strand labels along each axis. Defaults to `a|b|c`."""
  _tie_up:np.ndarray = None
  """optional tie-up array to pass through for `this` weave type."""
  _tr:np.ndarray = None
  """optional treadling array to pass through for `this` weave type."""
  _th:np.ndarray = None
  """optional threading array to pass through for `this` weave type."""

  def __init__(self, **kwargs:float|str) -> None:
    super().__init__(**kwargs)


  def _setup_tiles(self) -> None:
    """Set up weave unit."""
    if self.weave_type in ("hex", "cube"):
      self.base_shape = TileShape.HEXAGON
      self._setup_triaxial_weave_unit()
    else:
      self.base_shape = TileShape.RECTANGLE
      self._setup_biaxial_weave_unit()


  def _setup_regularised_prototile(self) -> None:
    """Set up regularised prototile fully containing all tile elements.

    The work is carried out by the two methods  that follow _regularise_tiles()
    and _merge_fragments().
    """
    self._regularise_tiles()
    # it's prudent to do some cleanup given all the manipulation of geometries
    # carried out to generate the regularised prototile. But note that the
    # regularised prototile has no functional purpose, so it's OK if it has,
    # for example, additional points along line segments.
    self.regularised_prototile.geometry = tiling_utils.repair_polygon(
      self.regularised_prototile.geometry)


  def _setup_biaxial_weave_unit(self) -> None:
    """Set up weave tiles GeoDataFrame and tile GeoDataFrame."""
    warp_threads, weft_threads, _ = \
      tiling_utils.get_strand_ids(self.strands)
    if self.weave_type == "basket" and isinstance(self.n, (list, tuple)):
      self.n = self.n[0]
    p = weave_matrices.get_weave_pattern_matrix(
      weave_type = self.weave_type, n = self.n, warp = warp_threads,
      weft = weft_threads, tie_up = self._tie_up, tr = self._tr,
      th = self._th)
    self._make_shapes_from_coded_weave_matrix(
      Loom(p), strand_labels = [weft_threads, warp_threads, []])
    bb = self.tiles.total_bounds
    w = (bb[2] - bb[0]) // self.spacing * self.spacing
    h = (bb[3] - bb[1]) // self.spacing * self.spacing
    self.setup_vectors((0, h), (w, 0))


  def _get_triaxial_weave_matrices(self,
      strands_1:list[str]|tuple[str] = ("a",),
      strands_2:list[str]|tuple[str] = ("b",),
      strands_3:list[str]|tuple[str] = ("c",),
    ) -> Loom:
    """Return encoded weave pattern matrix as Loom of three biaxial matrices.

    Allowed weave_types: "cube" or "hex".

    "hex" is not flexible and will fail with any strand label lists that are
    not length 3 or include more than one non-blank "-" item. You can
    generate the "hex" weave with the default settings in any case!

    Strand lists should be length 3 or length 1. "cube" tolerates more
    than "hex" for the items in the strand lists.

    Defaults will produce 'mad weave'.

    Args:
      strands_1 (list[str]|tuple[str], optional): list of labels
      for warp strands. Defaults to ["a"].
      strands_2 (list[str]|tuple[str], optional): list of labels
      for weft strands. Defaults to ["b"].
      strands_3 (list[str]|tuple[str], optional): list of labels
      for weft strands. Defaults to ["c"].

    Returns:
      Loom: which combines the three biaxial weaves 12, 23 and 31 implied
      by the strand label lists.

    """
    if self.weave_type == "hex":
      loom = Loom(
        weave_matrices.get_weave_pattern_matrix(
          weave_type = "this", tie_up = np.ones((6, 6)),
          warp = strands_1, weft = strands_2),
        weave_matrices.get_weave_pattern_matrix(
          weave_type = "this", tie_up = np.ones((6, 6)),
          warp = strands_2, weft = strands_3),
        weave_matrices.get_weave_pattern_matrix(
          weave_type = "this", tie_up = np.ones((6, 6)),
          warp = strands_3, weft = strands_1))
    else: # "cube"
      loom = Loom(
      # Note n = (1,2,1,2) is required here to force 6x6 twill
        weave_matrices.get_weave_pattern_matrix(
          weave_type = "twill", n = (1, 2, 1, 2),
          warp = strands_1, weft = strands_2),
        weave_matrices.get_weave_pattern_matrix(
          weave_type = "twill", n = (1, 2, 1, 2),
          warp = strands_2, weft = strands_3),
        weave_matrices.get_weave_pattern_matrix(
          weave_type = "twill", n = (1, 2, 1, 2),
          warp = strands_3, weft = strands_1))
    return loom


  def _setup_triaxial_weave_unit(self) -> None:
    """Return weave tiles GeoDataFrame and tile GeoDataFrame."""
    strands_1, strands_2, strands_3 = \
      tiling_utils.get_strand_ids(self.strands)
    loom = self._get_triaxial_weave_matrices(
      strands_1 = strands_1, strands_2 = strands_2, strands_3 = strands_3)
    self._make_shapes_from_coded_weave_matrix(
      loom, strand_labels = [strands_1, strands_2, strands_3])
    self.setup_vectors((0, 6 * self.spacing),
                       (3 * self.spacing * np.sqrt(3),  3 * self.spacing),
                       (3 * self.spacing * np.sqrt(3), -3 * self.spacing))


  def _make_shapes_from_coded_weave_matrix(
      self,
      loom:Loom, strand_labels:list[list[str]] = (["a"], ["b"], ["c"]),
    ) -> None:
    """Set up weave tiles and prototile GeoDataFrames in a dictionary.

    Builds the geometries associated with a given weave supplied as
    'loom' containing the coordinates in an appropriate grid (Cartesian or
    triangular) and the orderings of the strands at each coordinate location

    Args:
      loom (Loom): matrix or stack of matrices representing the weave
      pattern.
      strand_labels (list[list[str]], optional): list of lists of labels
      for strands in each direction. Defaults to [["a"], ["b"], ["c"]]

    """
    grid = WeaveGrid(loom.n_axes, loom.orientations, self.spacing)
    # expand the list of strand labels if needed in each direction
    labels = [thread * int(np.ceil(dim // len(thread)))
              for dim, thread in zip(loom.dimensions, strand_labels,
                                     strict = False)]
    weave_polys = []
    strand_ids = []
    cells = []
    for k, strand_order in zip(loom.indices, loom.orderings, strict = True):
      IDs = [thread[coord] for coord, thread in zip(k, labels, strict = True)]
      cells.append(grid.get_grid_cell_at(k))
      if strand_order is None:
        continue  # No strands present
      if strand_order == "NA":
        continue  # Inconsistency in layer order
      n_slices = [len(ID) for ID in IDs]
      next_polys = grid.get_visible_cell_strands(
        width = self.aspect, coords = k,
        strand_order = strand_order, n_slices = n_slices)
      weave_polys.extend(next_polys)
      next_labels = [list(IDs[i]) for i in strand_order]  # list of lists
      next_labels = list(itertools.chain(*next_labels))   # flatten
      strand_ids.extend(next_labels)
    # sometimes empty polygons make it to here, so
    # filter those out along with the associated IDs
    real_polys = [not p.is_empty for p in weave_polys]
    weave_polys = [
      p for p, b in zip(weave_polys, real_polys, strict = True) if b]
    strand_ids = [
      ID for ID, b in zip(strand_ids, real_polys, strict = True) if b]
    # note that the tile is important for the biaxial case, which makes it
    # a little hard to understand why the behaviour is so different in
    # the biaxial and triaxial cases; however below seems to work...
    if loom.n_axes == 3:
      tile = grid.get_tile_from_cells(cells)
      shift = (0, 0)
    else:
      tile = tiling_utils.safe_union(gpd.GeoSeries(cells), as_polygon = True)
      shift = (-tile.centroid.x, -tile.centroid.y)
      tile = grid.get_tile_from_cells(tile)
    self.tiles = self._get_weave_tiles_gdf(weave_polys, strand_ids, shift)
    self.prototile = gpd.GeoDataFrame(
      geometry = gpd.GeoSeries([tile]), crs = self.crs)


  def _get_weave_tiles_gdf(
      self, polys:list[geom.Polygon], strand_ids:list[str],
      offset:tuple[float]) -> gpd.GeoDataFrame:
    """Make a GeoDataFrame from weave tile polygons, labels, etc.

    Args:
      polys (list[Polygon | MultiPolygon]): list of weave tile
        polygons.
      strand_ids (list[str]): list of strand labels.
      offset (tuple[float]): offset to centre the weave tiles on the
        tile.

    Returns:
      geopandas.GeoDataFrame: GeoDataFrame clipped to the tile, with
        margin applied.

    """
    weave = gpd.GeoDataFrame(
      data = {"tile_id": strand_ids},
      geometry = gpd.GeoSeries([affine.translate(p, offset[0], offset[1])
                                for p in polys]))
    weave = weave[weave.tile_id != "-"]
    weave.geometry = gpd.GeoSeries(
      [tiling_utils.get_clean_polygon(p) for p in weave.geometry])

    # some buffering is required if aspect is 1 to safely dissolve and
    # explode weave unit tiles that meet at corners
    if self.aspect == 1:
      # grow for dissolve
      weave.geometry = weave.geometry.buffer(
        # self.spacing * tiling_utils.RESOLUTION,
        tiling_utils.RESOLUTION,
        join_style = "mitre", cap_style = "square")
      weave = weave.dissolve(by = "tile_id", as_index = False)
      # shrink by more to explode into separate polygons
      weave.geometry = weave.geometry.buffer(
        # -2 * self.spacing * tiling_utils.RESOLUTION,
        -2 * tiling_utils.RESOLUTION,
        join_style = "mitre", cap_style = "square")
      weave = weave.explode(ignore_index = True)
      weave.geometry = weave.geometry.buffer(
        # self.spacing * tiling_utils.RESOLUTION,
        tiling_utils.RESOLUTION,
        join_style = "mitre", cap_style = "square")
    else: # aspect < 1 is fine without buffering
      weave = weave.dissolve(by = "tile_id", as_index = False)
      weave = weave.explode(ignore_index = True)
    weave.geometry = gpd.GeoSeries(
      [tiling_utils.get_clean_polygon(p) for p in weave.geometry])
    return weave.set_crs(self.crs)


  def _get_axis_from_label(self, label:str = "a", strands:str = None):
    """Determine the axis of a tile_id from the strands spec string.

    Args:
      label (str, optional): the tile_id. Defaults to "a".
      strands (str, optional): the strand spec. Defaults to the WeaveUnit
      strands attribute.

    Returns:
      _type_: the axis in which the supplied tile is found.

    """
    if strands == None:
      strands = self.strands
    index = strands.index(label)
    return strands[:index].count("|")


  def _get_legend_tiles(self) -> gpd.GeoDataFrame:
    """Return tiles suitable for use in a legend representation.

    One tile for each tile_id value will be chosen, close to the
    centre of the prototile extent, and not among the smallest tiles present
    (for example not a short length of strand mostly hidden by other
    strands)

    Returns:
      gpd.GeoDataFrame: the chosen tiles.

    """
    angles = ((0, 240, 120)
              if self.weave_type in ("hex", "cube")
              else (90, 0))
    tile_ids = pd.Series.unique(self.tiles.tile_id)
    groups = self.tiles.groupby("tile_id")
    tiles, rotations = [], []
    for id in tile_ids:
      candidates = groups.get_group(id)
      axis = self._get_axis_from_label(id, self.strands)
      tiles.append(
        self._get_most_central_large_tile(candidates, tiles))
      rotations.append(-angles[axis] + self.rotation)
    return gpd.GeoDataFrame(
      data = {"tile_id": tile_ids, "rotation": rotations},
      crs = self.crs,
      geometry = gpd.GeoSeries(tiles))


  def _get_most_central_large_tile(self, tiles:gpd.GeoDataFrame,
          other_tiles:list[geom.Polygon]) -> geom.Polygon:
    """Get a large tile close to the centre of the WeaveUnit.

    Args:
      tiles (gpd.GeoDataFrame): the set of tiles to choose from.

    Returns:
      geom.Polygon: the chosen, large central tile.

    """
    areas = [g.area for g in tiles.geometry]
    min_area, max_area = min(areas), max(areas)
    if min_area / max_area > 0.5:
      geoms = list(tiles.geometry)
    else:
      mean_log_a = np.mean(np.log(areas))
      geoms = [g for g, a in zip(tiles.geometry, areas)
              if np.log(a) > mean_log_a]
    if len(other_tiles) == 0 or self.weave_type in ("cube", "hex"):
      d = [g.centroid.distance(geom.Point(0, 0)) for g in geoms]
    else:
      c = geom.MultiPolygon(other_tiles).centroid
      d = [geom.MultiPolygon([g] + other_tiles).centroid.distance(c)
          for g in geoms]
    return geoms[d.index(min(d))]


  def _get_legend_key_shapes(
      self,
      polygon:geom.Polygon,
      counts:Iterable = [1] * 25,
      angle:float = 0,
      radial:bool = False,
    ) -> list[geom.Polygon]:
    """Return polygons obtained by slicing polygon crosswise into n slices.

    Orientation of the polygon is indicated by the angle.

    The returned list of polygons can be used to form a colour ramp in a
    legend.

    Args:
      polygon (geom.Polygon): the weave strand polygon to slice.
      counts (Iterable, optional): an iterable list of the numbers of
        slices in each category. Defaults to [1] * 25.
      angle (float, optional): orientation of the polygon. Defaults to 0.
      radial (bool, optional): ignored by WeaveUnit.

    Returns:
      list[geom.Polygon]: a list of polygons.

    """
    c = polygon.centroid
    g = affine.rotate(polygon, -angle, origin = c)
    width, height, left, bottom = \
      tiling_utils.get_width_height_left_bottom(gpd.GeoSeries([g]))
    margin = width / 50
    total = sum(counts)
    cuts = list(np.cumsum(counts))
    cuts = [0] + [c / total for c in cuts]
    cuts = [left + c * width for c in cuts]
    bottom = bottom - margin
    top = bottom + height + 2 * margin
    slices = []
    for l, r in zip(cuts[:-1], cuts[1:], strict = True):
      # we add a margin to left and right so that they overplot; otherwise in
      # rendering matplotlib leaves small gaps which give a washed out look
      # to the fill colour!
      slice = geom.Polygon([(l - margin, bottom), (r + margin, bottom),
                            (r + margin,    top), (l - margin,    top)])
      slices.append(slice.intersection(g))
    return [affine.rotate(s, angle, origin = c) for s in slices]
