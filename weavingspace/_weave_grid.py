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

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import shapely
import shapely.affinity as affine
import shapely.geometry as geom

from weavingspace import tiling_utils


@dataclass(slots=True)
class WeaveGrid:
  """Class to encapsulate generation of sites where strands intersect.

  Can generate both triangular and square grid 'cells' as visual
  representation of overlaps of two or three strands at a site.

  NOTE: there is minimal 'gridification' or 'cleaning' in the geometry of
  this code - we leave that to steps further down the pipeline, preferring
  to perform this low level geometry with maximum precision. The only limited
  break with this is in get_visible_cell_strands, where strands in higher
  layers are buffered by a small amount to avoid closely aligned polygon
  edges producing slivers etc. This is likely only required in the triaxial
  case.

  Atrributes:
    n_axes (int): the number of axes in the weave, 2 or 3.
      Defaults to 2.
    orientations (tuple[float]: orientations of the two or three
      axes either (0, -90) or (0, 120, 240). Defaults to (0, -90).
    spacing (float): spacing of the strands. Defaults to 10_000.
    basis (np.ndarray): matrix to calculate x,y coordinates of a
      site from its (integer) grid coordinates,
    grid_cell (geom.Polygon): the base triangle or square of the grid.
  """

  n_axes:int # = 2
  orientations:tuple[float,...] # = (0, -90)
  spacing:float # = 10000
  basis:np.ndarray #|None = None
  grid_cell:geom.Polygon #|None = None

  def __init__(
      self,
      n_axes:int,
      orientations:tuple[float,...],
      spacing:float = 10000,
    ) -> None:
    """Initialise a WeaveGrid."""
    self.n_axes = n_axes
    self.orientations = orientations
    self.spacing = spacing
    self.basis = self.setup_basis()
    self.grid_cell = tiling_utils \
      .get_regular_polygon(self.spacing, 4 if self.n_axes == 2 else 3)


  def setup_basis(self) -> np.ndarray:
    """Set up basis of the grid coordinate generation matrix.

    The underlying mathematics of the triangular grid case is more complicated
    than you'd imagine. See

    Nagy BN. 2003. Shortest Paths in Triangular Grids with Neighbourhood
    Sequences. Journal of Computing and Information Technology 11(2)111.
    https://doi.org/10.2498/cit.2003.02.04

    Returns:
      np.nd.array: self.n_axes x 2 matrix to generate float coordinates
        of cells in Cartesian space from integer grid coordinates of
        cells.

    """
    # angles are perpendicular to the direction of their respective strands
    # so e.g. 2nd strand in 3 axis case is at 120, and perpendicular is 210
    if self.n_axes == 2:
      angles = (np.pi / 2, 0)
      dx = [self.spacing * np.cos(a) for a in angles]  # [0, 1]
      dy = [self.spacing * np.sin(a) for a in angles]  # [1, 0]
    else: # self.n_axes == 3:
      angles = [np.pi / 6 * x for x in range(3, 12, 4)]  # [90, 210, 330]
      dx = [self.spacing * 2 / 3 * np.cos(a) for a in angles]
      dy = [self.spacing * 2 / 3 * np.sin(a) for a in angles]
    return np.array(dx + dy).reshape((2, self.n_axes))


  def get_coordinates(self,
                      coords:tuple[int,...],
                      ) -> np.ndarray:
    """Return Cartesian coordinates of cell centroid from grid coordinates.

    Args:
      coords (tuple[int]): integer coordinate pair of grid location.

    Returns:
      np.ndarray: float coordinate pair of grid cell centroid.

    """
    return self.basis @ coords


  def get_grid_cell_at(self,
                       coords:tuple[int,...]|None = None,
                       ) -> geom.Polygon:
    """Return grid cell polygon centred on coords.

    Args:
      coords (tuple[int,...], optional): _description_. Defaults to None.

    Returns:
      geom.Polygon: square or triangle centred at the specified
        coordinates.

    """
    if coords is None:
      coords = tuple([0] * self.n_axes)
    if self.grid_cell is None:
      polygon = tiling_utils \
        .get_regular_polygon(self.spacing, 4 if self.n_axes == 2 else 3)
    else:
      polygon = self.grid_cell
      xy = self.get_coordinates(coords)
      polygon = affine.translate(polygon, xy[0], xy[1])
    if self.n_axes == 2 or sum(coords) % 2 == 0:
      return polygon
    # triaxial case where triangle cell has to be flipped
    return affine.rotate(polygon, 180, origin = polygon.centroid)


  def _get_angles(self,
                  n:int = 4,
                  ) -> list[float]:
    """Return angles to corners of n-gon, with one side parallel to x-axis.

    To determine angles start at 6 o'clock (3pi/2) and add (pi/n), then
    subtract a series of n - 1 2pi/n steps. Note we subtract due to the CW
    polygon winding order convention of shapely.

    Args:
      n (int, optional): Number of sides. Defaults to 4.

    Returns:
      list[float]: angles in radians.

    """
    return [(3 * np.pi/2) + (np.pi/n) - (i/n * 2 * np.pi) for i in range(n)]


  def _get_grid_cell_slices(self,
                            slice_l:float,
                            w:float = 1.0,
                            n_slices:int = 1,
                            ) -> list[geom.Polygon]:
    r"""Get list of rectangular polygons represneting 'slices' across cell.

    Returns 'slices' across grid cell (i.e. horizontally) centred vertically
    relative to the cell, ie

             /\
            /  \
    +------------------+
    |     /      \     |
    +------------------+
    |   /          \   |
    +------------------+
      /              \
     /________________\

    Horizontal extent is l, total width of the strips is W * self.spacing,
    they are 'sliced' horizontally in n_slices slices of equal width.

    Args:
      slice_l (float, optional): length of slices. Defaults to 0.
      w (float, optional): width of slices relative to grid spacing.
        Defaults to 1.
      n_slices (int, optional): number of slices to divide strands into
        along their length. Defaults to 1.

    Returns:
      list[geom.Polygon]: _description_

    """
    slice_l = self.spacing if slice_l == 0 else slice_l
    # note that strand width is based on self.spacing, not L because L
    # may be larger if generating slices for aspect < 1 when strands
    # will extend outside grid cell.
    strand_w = self.spacing * w
    slice_w = strand_w / n_slices
    odd_numbers = list(range(1, 2 * n_slices, 2))
    slice_offsets = [(slice_w * o / 2) - (strand_w / 2) for o in odd_numbers]
    base_slice = geom.Polygon([
      (-slice_l/2, -slice_w/2), (-slice_l/2,  slice_w/2),
      ( slice_l/2,  slice_w/2), ( slice_l/2, -slice_w/2)])
    return [affine.translate(base_slice, 0, offset)
            for offset in slice_offsets]


  def _get_cell_strands(self,
                        width:float = 1.0,
                        coords:tuple[int,...]|None = None,
                        orientation:float = 0.0,
                        n_slices:int = 1,
                        ) -> list[shapely.Geometry]:
    """Get cells strands across grid cell at coordinates and orientation.

    The strands will have specified total width across the cell.

    Args:
      width (float, optional): total width of strands relative to
        self.spacing. Defaults to 1.0.
      coords (tuple[int], optional): integer grid coordinates of
        cell. Defaults to None.
      orientation (int, optional): orientation of the strands.
        Defaults to 0.
      n_slices (int, optional): number of length-wise slices to cut
        strands into. Defaults to 1.

    Returns:
      list[geom.Polygon|geom.MultiPolygon]: polygons representing the
        strands.

    """
    cell = self.get_grid_cell_at(coords)
    # when aspect is <1 strands extend outside cell by some scale factor
    sf = 2 - width if self.n_axes == 2 else (5 - 3 * width) / 2
    expanded_cell = affine.scale(cell, sf, sf, origin = cell.centroid)
    big_l = (sf * self.spacing       ## rectangular case is simple
             if self.n_axes == 2     ## triangular less so!
             else sf * self.spacing * 2 / np.sqrt(3) * (3 - width) / 2)
    strands = geom.MultiPolygon(
      self._get_grid_cell_slices(slice_l = big_l, w = width,
                                 n_slices = n_slices))
    # we need centre of cell bounding box to shift strands to
    # vertical center of triangular cells. In rectangular case
    # this will be (0, 0).
    cell_offset = cell.envelope.centroid
    strands = affine.translate(strands, cell_offset.x, cell_offset.y)
    strands = geom.MultiPolygon(
        [expanded_cell.intersection(s) for s in strands.geoms])
    strands = affine.rotate(strands, orientation, origin = cell.centroid)
    return [tiling_utils.gridify(s) for s in strands.geoms]


  def get_visible_cell_strands(self,
                               width:float= 1.0,
                               coords:tuple[int,...]|None = None,
                               strand_order:tuple[int,...] = (0, 1, 2),
                               n_slices:tuple[int,...] = (1, 1, 1),
                               ) -> list[geom.Polygon|geom.MultiPolygon]:
    """Return visible strands in grid cell based on layer order.

    Returns visible parts of the strands in a grid cell, given strand width,
    strand layer order and the number of slices in each direction.

    Args:
      width (float): total width of strands relative to self.spacing.
        Defaults to 1.0.
      coords (tuple[int], optional): grid cell coordinates. Defaults
        to None.
      strand_order (tuple[int], optional): order of the layers from top,
        at this cell site. Defaults to (0, 1, 2).
      n_slices (tuple[int], optional): number of slices in each layer
        at this cell site. Defaults to (1, 1, 1).

    Returns:
      list[geom.Polygon|geom.MultiPolygon]: those polygons that
        will be visible at this site given requested strand order from
        the top.

    """
    all_polys = []
    for order in strand_order[:self.n_axes]:
        next_polys = self._get_cell_strands(
          width, coords, self.orientations[order], n_slices[order])
        if all_polys == []:
            all_polys.extend(next_polys)
            mask = shapely.union_all(next_polys)
        else:
            # buffering the mask cleans up many issues with closely
            # aligned polygon edges in overlayed layers
            all_polys.extend([p.difference(
              mask.buffer(tiling_utils.RESOLUTION,
                          join_style = "mitre", cap_style = "square"))
              for p in next_polys])
            mask = mask.union(shapely.union_all(next_polys))
    return all_polys


  def get_tile_from_cells(self,
                          approx_tile:geom.Polygon|list[geom.Polygon],
                          ) -> geom.Polygon:
    """Return rectangle or hexagon derived from bounds of supplied approx tile.

    This is required because we know the required tile is an exact 4 or
    6 cornered polygon, but the MultiPolygon formed by unary_union may have
    many more corners than this (due to buffering etc during construction).

    Args:
      approx_tile (geom.Polygon): MultiPolygon formed from the cells of
        the tile.

    Returns:
      geom.Polygon (geom.Polygon): rectangle or hexagon geom.Polygon.

    """
    if isinstance(approx_tile, Iterable):
      xmin, ymin, xmax, ymax = \
        1000_000_000, 1000_000_000, -1000_000_000, -1000_000_000
      for p in approx_tile:
        xmn, ymn, xmx, ymx = p.bounds
        xmin, ymin, xmax, ymax = \
          min(xmin, xmn), min(ymin, ymn), max(xmx, xmax), max(ymx, ymax)
    else:
      xmin, ymin, xmax, ymax = approx_tile.bounds
    w = xmax - xmin
    h = ymax - ymin
    if self.n_axes == 2:
      return geom.Polygon([(-w/2, -h/2), (-w/2,  h/2),
                           ( w/2,  h/2), ( w/2, -h/2)])
    return geom.Polygon([( w/4, -h/2), (-w/4, -h/2), (-w/2, 0),
                           (-w/4,  h/2), ( w/4,  h/2), ( w/2, 0)])
