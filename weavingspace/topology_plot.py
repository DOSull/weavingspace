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

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
  from weavingspace import (
    Topology,
    Transform,
  )

"""A collection of functions for plotting `weavingspace.Topology` objects. The
two public functions are `plot()` and `plot_tiling_symmetries()`. All other
functions are helpers.
"""

def plot(
    topo: Topology,
    show_original_tiles: bool = True,
    show_tile_centres: bool = False,
    show_vertex_labels: bool = True,
    show_vertex_ids: bool = False,
    show_edges: bool = True,
    offset_edges: bool = True,
    show_edge_labels: bool = False,
    show_dual_tiles: bool = False,
  ) -> plt.Axes:
  """Delegate plotting of requested elements and return plt.Axes.

  Args:
    topo (Topology): the Topology to plot.
    show_original_tiles (bool, optional): if True show the tiles. Defaults to
      True.
    show_tile_centres (bool, optional): if True show tile centres with upper
      case alphabetical labels. Defaults to False.
    show_vertex_labels (bool, optional): if True show tiling vertices labelled
      to show their equivalence classes. Defaults to True.
    show_vertex_ids (bool, optional): if True show vertex IDs (i.e., sequence
      numbers) which is useful for debugging. Defaults to False.
    show_edges (bool, optional): if True show tiling edges (not tile sides).
      Defaults to True.
    offset_edges (bool, optional): if True offset edges a little from and
      parallel to their geometric position. Defaults to True.
    show_edge_labels (bool, optional): if true show lower case alphabetical
      labels identifying edge equivalence classes. Defaults to False.
    show_dual_tiles (bool, optional): if True show a candidate set of dual
      tiles as an overlay. Defaults to False.

  Returns:
    plt.Axes: a plot of the Topology as requested.

  """
  fig = plt.figure(figsize = (10, 10))
  ax = fig.add_axes(111)
  extent = gpd.GeoSeries([t.shape for t in topo.tiles]).total_bounds
  dist = max([extent[2] - extent[0], extent[3] - extent[1]]) / 100
  if show_original_tiles:
    _plot_tiles(topo, ax)
  if show_tile_centres:
    _plot_tile_centres(topo, ax)
  if show_vertex_labels:
    _plot_vertex_labels(topo, ax, show_vertex_ids)
  if show_edge_labels or show_edges:
    _plot_edges(topo, ax, show_edges, show_edge_labels, dist, offset_edges)
  if show_dual_tiles:
    _plot_dual_tiles(topo, ax, dist)
  plt.axis("off")
  return ax


def _plot_tiles(topo: Topology, ax: plt.Axes) -> plt.Axes:
  """Plot Topology's tiles on supplied Axes.

  Tiles are coloured by equivalence class.

  Args:
    topo (Topology): the Topology to plot.
    ax (plt.Axes): Axes on which to plot.

  Returns:
    plt.Axes: the Axes.

  """
  topo._get_tile_geoms().plot(column = "transitivity_class",
    ax = ax, ec = "#444444", lw = 0.5, alpha = 0.25, cmap = "Greys")
  return ax


def _plot_tile_centres(topo: Topology, ax: plt.Axes) -> plt.Axes:
  """Print tile transitivity class at each tile centroid.

  Args:
    topo (Topology): the Topology to plot.
    ax (plt.Axes): Axes on which to plot.

  Returns:
    plt.Axes: the Axes.

  """
  for tile in topo.tiles:
    ax.annotate(tile.transitivity_class, xy = (tile.centre.x, tile.centre.y),
                ha = "center", va = "center")
  return ax


def _plot_vertex_labels(
    topo: Topology,
    ax: plt.Axes,
    show_vertex_ids: bool = False,
  ) -> plt.Axes:
  """Plot either the Vertex transitivity class label or its sequential ID.

  Args:
    topo (Topology): the Topology to plot.
    ax (plt.Axes): Axes on which to plot.
    show_vertex_ids (bool, optional): If True plots the ID, else plots the
      transitivity class. Defaults to False.

  Returns:
    plt.Axes: the Axes.

  """
  for v in topo.points.values():
    ax.annotate(v.ID if show_vertex_ids else v.label,
                xy = (v.point.x, v.point.y), color = "k",
                ha = "center", va = "center")
  return ax


def _plot_edges(
    topo: Topology,
    ax: plt.Axes,
    show_edges: bool = False,
    show_edge_labels: bool = False,
    dist: float = 0.0,
    offset_edges: bool = True,
  ) -> plt.Axes:
  """Plot edges, including an offset if specified and labels if requested.

  Can also be called to only plot the labels.

  Args:
    topo (Topology): the Topology to plot.
    ax (plt.Axes): Axes on which to plot.
    show_edges (bool, optional): if True includes the edges as. a dotted blue
      line, optionally offset (for clarity) from the tile boundary. Defaults
      to False.
    show_edge_labels (bool, optional): if True shows an edge label. Defaults
      to False.
    dist (float, optional): a distance by which to offset the dotted line for
      the edge from the tile boundary. Defaults to 0.0.
    offset_edges (bool, optional): if True applies the edge drawing offset,
      if False the edge is drawn as a single line segment between its end
      vertices (and may not align with the sides of tiles. Defaults to True.

  Returns:
    plt.Axes: the Axes.

  """
  if show_edges:
    edges = topo._get_edge_geoms(dist if offset_edges else 0).geometry
    edges.plot(ax = ax, color = "dodgerblue", ls = ":")
  else:
    edges = topo._get_edge_geoms().geometry
  if show_edge_labels:
    for ls, e in zip(edges, topo.edges.values(), strict = True):
      c = ls.centroid
      ax.annotate(e.label, xy = (c.x, c.y), color = "k",
                  ha = "center", va = "center")
  return ax


def _plot_dual_tiles(
    topo: Topology,
    ax: plt.Axes,
    dist: float = 0.0,
  ) -> plt.Axes:
  gpd.GeoSeries(topo.dual_tiles).buffer(
    -dist / 4, join_style = "mitre", cap_style = "square").plot(
      ax = ax, fc = "g", alpha = 0.25)
  return ax


def plot_tiling_symmetries(topo: Topology, **kwargs: dict[str:float]) -> None:
  """Plot the symmetries of Topology's tiling.

  Most of the work here is delegated to `_plot_tiling_symmetry` which is run
  once per symmetry on a grid of plt.Axes built by this function.

  Args:
    topo (Topology): the Topology to plot.
    kwargs: passed through to `_plot_tiling_symmetry`

  """
  n = min(len(topo.tile_matching_transforms), 24)
  nc = int(np.ceil(np.sqrt(n)))
  nr = int(np.ceil(n / nc))
  fig = plt.figure(figsize = (12, 12 * nr / nc))
  for i, tr in enumerate(list(topo.tile_matching_transforms.values())[:n]):
    ax = fig.add_subplot(nr, nc, i + 1)
    _plot_tiling_symmetry(topo, tr, ax, **kwargs)


def _plot_tiling_symmetry(
    topo: Topology,
    transform: Transform,
    ax: plt.Axes,
    **kwargs: dict[str:float],
  ) -> None:
  """Plot the supplied Transform on the supplied plt.Axes.

  Args:
    topo (Topology): the Topology to plot.
    transform (Transform): the Transform to plot.
    ax (plt.Axes): the Axes on which to plot.
    kwargs: passed through to plt.plot functions.

  """
  tiles = gpd.GeoSeries([t.shape for t in topo.tiles])
  base_tiles = tiles[:topo.n_tiles]
  tiles.plot(ax = ax, fc = "k", alpha = .15, ec = "k", lw = .5)
  base_tiles.plot(ax = ax, fc = "#00000000", ec = "w", lw = 1, zorder = 2)
  transformed_base_tiles = gpd.GeoSeries(
    [transform.apply(g) for g in base_tiles])
  transformed_base_tiles.plot(ax = ax, fc = "k", alpha = .2, lw = 0, ec = "k")
  transform.draw(ax, **kwargs) # delegate to Transform.draw
  plt.axis("off")

