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

import copy
import itertools
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import geopandas as gpd
import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shapely.affinity as affine
import shapely.geometry as geom
import shapely.ops

from weavingspace import Tileable, TileUnit, tiling_utils

if typing.TYPE_CHECKING:
  from matplotlib.figure import Figure

"""Classes for tiling maps.

`weavingspace.tile_map.Tiling` and
`weavingspace.tile_map.TiledMap` are exposed in the  public API and
respectively enable creation of a tiling and plotting of the tiling as a
multivariate map.
"""

CMAPS_SEQUENTIAL = list(itertools.chain(*[[x, x + "_r"] for x in
  ["Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
   "viridis", "plasma", "inferno", "magma", "cividis",
   "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
   "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
   "binary", "gist_yarg", "gist_gray", "gray", "bone",
   "pink", "spring", "summer", "autumn", "winter", "cool",
   "Wistia", "hot", "afmhot", "gist_heat", "copper"]]))

CMAPS_DIVERGING = list(itertools.chain(*[[x, x + "_r"] for x in
  ["PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu",
   "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic"]]))
# these ones not yet in MPL 3.8.4: "berlin", "managua", "vanimo"

CMAPS_CATEGORICAL = list(itertools.chain(*[[x, x + "_r"] for x in
  ["Pastel1", "Pastel2", "Paired", "Accent", "Dark2",
   "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b",
   "tab20c"]]))

@dataclass(slots=True, init=False)
class _TileGrid:
  """A class to represent the translation centres of a tiling.

  We store the grid as a GeoSeries of Point objects to make it simple to plot
  in map views if required.

  Implementation relies on transforming the translation vectors into a square
  space where the tile spacing is unit squares, then transforming this back
  into the original map space. Some member variables of the class are in
  the transformed grid generation space, some in the map space. We store the
  translation vectors as shapely Points, and the extent of the grid in grid
  space as a shapely Polygon so we can more easily visualise if needed.
  """

  tile_unit:Tileable
  """the base tile in map space."""
  oriented_rect_to_tile:geom.Polygon
  """oriented rectangular region in map space that is to be tiled."""
  to_grid_space:tuple[float,...]
  """the forward transformation from map space to tiling grid generation space,
  stored as a shapely.affinity transform tuple of 6 floats."""
  to_map_space:tuple[float,...]
  """the inverse transform from tiling grid space to map space."""
  extent_in_grid_space:geom.Polygon
  """geometry of the circular extent of the tiling transformed into grid
  generation space."""
  centre:geom.Point
  """centre point of the grid in map space - this is required for later
  rotations of a generated tiling in map space"""
  points:gpd.GeoSeries
  """geom.Points recording translation vectors of the tiling in map space."""

  def __init__(
      self,
      tile_unit:Tileable,
      to_tile:gpd.GeoSeries,
      at_centroids:bool = False) -> None:
    self.tile_unit = tile_unit
    self.oriented_rect_to_tile = self._get_rect_to_tile(to_tile)
    self.to_map_space, self.to_grid_space = self._get_transforms()
    self._set_centre_in_map_space()
    self._set_extent_in_grid_space()
    if at_centroids:
      self.points = to_tile.representative_point()
    else:
      self.points = self._get_grid()
    self.points.crs = self.tile_unit.crs


  def _get_rect_to_tile(
      self,
      region_to_tile:gpd.GeoSeries) -> geom.Polygon:
    """Generate an oriented rectangle that encompasses the region to be tiled.

    This will be rotated to generate a circular are that will actually be tiled
    so that if new orientations of the tiled pattern are requested they can be
    generated quickly without much recalculation.

    Args:
      region_to_tile (gpd.GeoSeries): the region to be tiled.

    Returns:
      geom.Polygon: an oriented rectangle encompassing the region to be tiled
        with some buffering to avoid cutting off tiling elements.

    """
    # buffer the region by an amount dictated by the size of the tile unit
    bb = self.tile_unit.tiles.total_bounds
    diagonal = np.hypot(bb[2] - bb[0], bb[3] - bb[1])
    # note interpolated convex_hull operations to speed this step up
    return (region_to_tile
            .convex_hull
            .union_all("coverage")
            .convex_hull
            .buffer(diagonal)
            .minimum_rotated_rectangle)


  def _get_transforms(self) -> tuple[tuple[float,...],tuple[float,...]]:
    """Return forward and inverse transforms from map space to grid space.

    In grid generation space the translation vectors are (1, 0) and (0, 1)
    so we simply form a matrix from two translation vectors to get the
    transform from grid space to map space, and invert it to get transform from
    map space to grid space.

    See pages 18-22 in Kaplan CS, 2009. Introductory Tiling Theory for Computer
    Graphics (Morgan & Claypool) for the logic of this approach.

    The results are returned as shapely affine transform tuples.

    Returns:
      tuple[tuple[float,...],tuple[float,...]]: shapely affine transform tuples
        (a, b, d, e, dx, dy).

    """
    v = self.tile_unit.get_vectors()
    # unpack the first two vectors and funnel them into a 2x2 matrix
    grid_to_map = np.array([[v[0][0], v[1][0]], [v[0][1], v[1][1]]])
    map_to_grid = np.linalg.inv(grid_to_map)
    return (self._np_to_shapely_transform(grid_to_map),
            self._np_to_shapely_transform(map_to_grid))


  def _set_centre_in_map_space(self) -> None:
    """Set the tiling centre in map space."""
    self.centre = self.oriented_rect_to_tile.centroid


  def _set_extent_in_grid_space(self) -> None:
    """Set extent of the grid in grid generation space."""
    corner = geom.Point(self.oriented_rect_to_tile.exterior.coords[0])
    radius = self.centre.distance(corner)
    self.extent_in_grid_space = \
      affine.affine_transform(self.centre.buffer(radius), self.to_grid_space)


  def _get_grid(self) -> gpd.GeoSeries:
    """Generate the grid transformed into map space.

    Obtain dimensions of the transformed region, then set down a uniform grid.
    Grid generation is greatly accelerated by using the numpy meshgrid method
    which we can do because we're working in a grid-generation space where
    tile units have been transformed to the unit square.

    Returns:
      gpd.GeoSeries: the grid as a collection of geom.Points.

    """
    bb = self.extent_in_grid_space.bounds
    _w, _h, _l, _b = bb[2] - bb[0], bb[3] - bb[1], bb[0], bb[1]
    w = int(np.ceil(_w))
    h = int(np.ceil(_h))
    l = _l - (w - _w) / 2
    b = _b - (h - _h) / 2
    xs, ys = np.array(np.meshgrid(np.arange(w) + l,
                                  np.arange(h) + b)).reshape((2, w * h))
    pts = [geom.Point(x, y) for x, y in zip(xs, ys, strict = True)]
    return (gpd.GeoSeries(
      [p for p in pts if p.within(self.extent_in_grid_space)])
        .affine_transform(self.to_map_space))


  def _np_to_shapely_transform(
      self,
      array:np.ndarray) -> tuple[float,...]:
    """Convert a numpy affine transform matrix to shapely format.

        [[a b c]
         [d e f] --> (a b d e c f)
         [g h i]]

    There is no translation in transforms in this use-case so c == f == 0

    Args:
      array (np.ndarray): numpy affine transform array to convert.

    Returns:
      tuple[float,...]: shapely affine transform tuple

    """
    return (*list(array[:2, :2].flatten()), 0, 0)


@dataclass(slots=True, init=False)
class Tiling:
  """Class that applies a `Tileable` object to a region to be mapped.

  The result of the tiling procedure is stored in the `tiles` variable and
  covers a region sufficient that the tiling can be rotated to any desired
  angle. Rotation can be requested when the render method is called.
  """

  tileable:Tileable
  """Tileable on which the tiling is based."""
  region:gpd.GeoDataFrame
  """the region to be tiled."""
  region_union:geom.Polygon
  """a single polygon of all the areas in the region to be tiled"""
  grid:_TileGrid
  """the grid which will be used to apply the tiling."""
  tiles:gpd.GeoDataFrame
  """the tiles after tiling has been carried out."""
  prototiles:gpd.GeoDataFrame
  """the prototiles after tiling has been carried out."""
  reg_prototiles:gpd.GeoDataFrame
  """the regularised prototiles after tiling has been carried out."""
  rotation:float
  """additional rotation applied to the tiling beyond any that might have
  been 'baked in' to the Tileable."""

  def __init__(
      self,
      tileable:Tileable,
      region:gpd.GeoDataFrame,
      as_icons:bool = False,
    ) -> None:
    """Construct a tiling by polygons extending beyond supplied region.

    The tiling is extended sufficiently to allow for its application at any
    rotation.

    Args:
      tileable (Tileable): the TileUnit or WeaveUnit to use.
      region (gpd.GeoDataFrame): the region to be tiled.
      as_icons (bool, optional): if True prototiles will only be placed at the
        region's zone centroids, one per zone. Defaults to False.

    """
    self.tileable = tileable
    self.rotation = 0
    self.region = region
    self.region.sindex # this probably speeds up overlay
    self.region_union = self.region.geometry.union_all()
    self.grid = _TileGrid(
      self.tileable,
      self.region.geometry if as_icons else gpd.GeoSeries([self.region_union]),
      as_icons)
    self.tiles, self.prototiles, self.reg_prototiles = self.make_tiling()
    self.tiles.sindex # again this probably speeds up overlay


  def get_tiled_map(
      self,
      rotation:float = 0.,
      join_on_prototiles:bool = False,
      retain_tileables:bool = False,
      prioritise_tiles:bool = True,
      ragged_edges:bool = True,
      use_centroid_lookup_approximation:bool = False,
      debug:bool = False,
    ) -> TiledMap:
    """Return a `TiledMap` filling a region at the requested rotation.

    HERE BE DRAGONS! This function took a lot of trial and error to get right,
    so modify with CAUTION!

    The `proritise_tiles = True` option means that the tiling will not break
    up the tiles in `TileUnit`s at the boundaries between areas in the mapped
    region, but will instead ensure that tiles remain complete, picking up
    their data from the region zone which they overlap the most.

    The exact order in which operations are performed affects performance. For
    example, the final clipping to self.region when ragged_edges = False is
    _much_ slower if it is carried out before the dissolving of tiles into the
    region zones. So... again... modify CAREFULLY!

    Args:
      rotation (float, optional): Optional rotation to apply. Defaults to 0.
      join_on_prototiles (bool, optional): if True data from the region dataset
        are joined to tiles based on the prototile to which they belong. If
        False the join is based on the tiles in relation to the region areas.
        For weave-based tilings False is probably to be preferred. Defaults to
        False.
      retain_tileables (bool, optional): if True complete tileable units will
        be retained. If False tile unit elements that do not overlap the map
        area will be discarded.
      prioritise_tiles (bool, optional): if True tiles will not be broken at
        boundaries in the region dataset. Defaults to True.
      ragged_edges (bool, optional): if True tiles at the edge of the region
        will not be cut by the region extent - ignored if prioritise_tiles is
        False when edges will always be clipped to the region extent. Defaults
        to True.
      use_centroid_lookup_approximation (bool, optional): if True use tile
        centroids for lookup of region data - ignored if prioritise_tiles is
        False when it is irrelevant. Defaults to False.
      debug (bool, optional): if True prints timing messages. Defaults to
        False.

    Returns:
      TiledMap: a TiledMap of the region with attributes attached to tiles.

    """
    # this to avoid clashes with 'area' which is a plausible column name
    area_name = "my_ridiculous_area_name_42"
    if debug:
      t1 = perf_counter()

    id_var = self._setup_region_DZID()
    if join_on_prototiles:
      if rotation == 0:
        tiled_map, join_layer = self.tiles, self.prototiles
      else:
        tiled_map, join_layer = self.rotated(rotation)
      tiled_map["joinUID"] = self.tiles["prototile_id"]
    else:
      tiled_map = self.tiles if rotation == 0 else self.rotated(rotation)[0]
      tiled_map["joinUID"] = self.tiles["tile_id"]
      join_layer = tiled_map
    join_layer["joinUID"] = list(range(join_layer.shape[0]))

    # compile a list of the variable names we are NOT going to change
    # i.e. everything except the geometry and the id_var
    region_vars = [column for column in self.region.columns
                   if "geom" not in column and column != id_var]

    if debug:
      t2 = perf_counter()
      print(f"STEP 1: prep data (rotation if requested): {t2 - t1:.3f}")

    if prioritise_tiles:
      # maintain tile continuity across zone boundaries
      # so we have to do more work than a simple overlay
      if use_centroid_lookup_approximation:
        t5 = perf_counter()
        tile_pts = copy.deepcopy(join_layer)
        tile_pts.geometry = tile_pts.centroid
        lookup = tile_pts.sjoin(
          self.region, how = "inner")[["joinUID", id_var]]
      else:
        # determine areas of overlapping tiles and drop the data we join the
        # data back later, so dropping makes that easier overlaying in region.
        # overlay(tiles) seems to be faster??
        # TODO: also... this part is performance-critical, think about fixes --
        # possibly including the above centroid-based approximation
        overlaps = self.region.overlay(join_layer, make_valid = False)
        if debug:
          t3 = perf_counter()
          print(f"STEP A2: overlay zones with tiling: {t3 - t2:.3f}")
        overlaps[area_name] = overlaps.geometry.area
        if debug:
          t4 = perf_counter()
          print(f"STEP A3: calculate areas: {t4 - t3:.3f}")
        overlaps = overlaps.drop(columns = region_vars)
        if debug:
          t5 = perf_counter()
          print(f"STEP A4: drop columns prior to join: {t5 - t4:.3f}")
        # make a lookup by largest area tile to region id
        lookup = overlaps \
          .iloc[overlaps.groupby("joinUID")[area_name] \
          .agg(pd.Series.idxmax)][["joinUID", id_var]]
      # now join the lookup and from there the region data
      if debug:
        t6 = perf_counter()
        print(f"STEP A5: build lookup for join: {t6 - t5:.3f}")
      tiled_map = tiled_map \
        .merge(lookup, on = "joinUID") \
        .merge(self.region.drop(columns = ["geometry"]), on = id_var)
      if debug:
        t7 = perf_counter()
        print(f"STEP A6: perform lookup join: {t7 - t6:.3f}")
      tiled_map = tiled_map.drop(columns = ["joinUID"])

    else:  
      # here it's a simple overlay
      tiled_map = self.region.overlay(tiled_map)
      t7 = perf_counter()
      if debug:
        print(f"STEP B2: overlay tiling with zones: {t7 - t2:.3f}")

    if not retain_tileables:
      tiled_map = tiled_map.loc[
        shapely.intersects(self.region_union, np.array(tiled_map.geometry)), :]
    
    # inplace changes considered unsafe, BUT not dropping id_var in this
    # causes it to persist in the tiled_map and region dataframes!
    tiled_map.drop(columns = [id_var], inplace = True)
    self.region.drop(columns = [id_var], inplace = True)

    # if we've retained tiles and want 'clean' edges, then clip
    # note that this step is slow: geopandas unary_unions the clip layer
    if prioritise_tiles and not ragged_edges:
      tiled_map.sindex
      tiled_map = tiled_map.clip(self.region_union)
      if debug:
        print(f"""STEP A7/B3: clip map to region: {perf_counter() - t7:.3f}""")

    tm = TiledMap()
    tm.tiling = self
    tm.map = tiled_map
    return tm


  def _setup_region_DZID(self) -> str:
    """Return guaranteed-unique new attribute name for self.region dataframe.

    Avoids a name clash with any existing attribute in the dataframe.

    Returns:
      str: name of the added attribute.

    """
    dzid = "DZID"
    i = 0
    while dzid in self.region.columns:
      dzid = "DZID" + str(i)
      i = i + 1
    self.region[dzid] = list(range(self.region.shape[0]))
    return dzid


  def make_tiling(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Tile the region with a tile unit, returning a GeoDataFrame.

    Returns:
      geopandas.GeoDataFrame: a GeoDataFrame of the region tiled with the
        tile unit.

    """
    # we assume the geometry column is called geometry so make it so...
    if self.region.geometry.name != "geometry":
      self.region = self.region.rename_geometry("geometry")

    # chain list of lists of GeoSeries geometries to list of geometries
    tiles = itertools.chain(*[
      self.tileable.tiles.geometry.translate(p.x, p.y)
      for p in self.grid.points])
    prototiles = itertools.chain(*[
      self.tileable.prototile.geometry.translate(p.x, p.y)
      for p in self.grid.points])
    reg_prototiles = itertools.chain(*[
      self.tileable.regularised_prototile.geometry.translate(p.x, p.y)
      for p in self.grid.points])
    # replicate the tile ids
    tile_ids = list(self.tileable.tiles.tile_id) * len(self.grid.points)
    prototile_ids = list(range(len(self.grid.points)))
    tile_prototile_ids = sorted(prototile_ids * self.tileable.tiles.shape[0])
    tiles_gs = gpd.GeoSeries(list(tiles))
    prototiles_gs = gpd.GeoSeries(list(prototiles))
    reg_prototiles_gs = gpd.GeoSeries(list(reg_prototiles))
    # assemble and return as GeoDataFrames
    tiles_gdf = gpd.GeoDataFrame(
      data = {"tile_id": tile_ids, "prototile_id": tile_prototile_ids},
      geometry = tiles_gs, crs = self.tileable.crs)
    prototiles_gdf = gpd.GeoDataFrame(
      data = {"prototile_id": prototile_ids},
      geometry = prototiles_gs, crs = self.tileable.crs)
    reg_prototiles_gdf = gpd.GeoDataFrame(
      data = {"prototile_id": prototile_ids},
      geometry = reg_prototiles_gs, crs = self.tileable.crs)
    return tiles_gdf, prototiles_gdf, reg_prototiles_gdf


  def get_prototiles_background(self):
    return gpd.GeoSeries([p for p in self.prototiles.geometry
                          if p.intersects(self.region_union)])


  def get_regularised_prototiles_background(self):
    return gpd.GeoSeries([p for p in self.reg_prototiles.geometry
                          if p.intersects(self.region_union)])

  def get_merged_tiles_background(self):
    return gpd.GeoSeries([
      p for p in self.tiles.dissolve(by = "prototile_id").geometry
      if p.intersects(self.region_union)])


  def rotated(self,
              rotation:float = 0.0,
              ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return the stored tiling rotated.

    The stored tiling never changes and
    if it was originally made with a Tileable that was rotated it will retain
    that rotation. The requested rotation is _additional_ to that baseline
    rotation.

    Args:
      rotation (float, optional): Rotation angle in degrees.
        Defaults to None.

    Returns:
      gpd.GeoDataFrame: Rotated tiling.

    """
    if self.tiles is None:
      self.tiles = self.make_tiling()[0]
    if rotation == 0:
      return self.tiles, self.prototiles
    tiles = gpd.GeoDataFrame(
      data = {"tile_id": self.tiles.tile_id,
              "prototile_id": self.tiles.tile_id},
      crs = self.tiles.crs,
      geometry = self.tiles.geometry.rotate(
        rotation, origin = self.grid.centre))
    prototiles = gpd.GeoDataFrame(
      data = {"prototile_id": self.prototiles.prototile_id},
      crs = self.prototiles.crs,
      geometry = self.prototiles.geometry.rotate(
        rotation, origin = self.grid.centre))
    self.rotation = rotation
    return tiles, prototiles


@dataclass(slots=True)
class TiledMap:
  """Class representing a tiled map.

  Should not be accessed directly, but will
  be created by calling `Tiling.get_tiled_map()`. After creation the variables
  and colourmaps attributes can be set, and then `TiledMap.render()` called to
  make a map. Settable attributes are explained in documentation of the
  `TiledMap.render()` method.

  Examples:
    Recommended usage is as follows. First, make a `TiledMap` from a `Tiling`
    object:

      `tm = tiling.get_tiled_map(...)`

    Some options in the `Tiling` constructor affect the map appearance. See
    `Tiling` for details.

    Once a `TiledMap` object exists, set options on it, either when calling
    `TiledMap.render()` or explicitly, i.e.

      tm.render(opt1 = val1, opt2 = val2, ...)

    or

      tm.opt1 = val1
      tm.opt2 = val2
      tm.render()

    Option settings are persistent, i.e. unless a new `TiledMap` object is
    created the option settings have to be explicitly reset to new values on
    subsequent calls to `TiledMap.render()`.

    The most important options are the `vars_map` and `colors_to_use` settings.

    `vars_to_map` is a lost of the dataset variable names to match with
    `weavingspace.tileable.Tileable` elements with corresponding (ordered)
    tile_ids (usually "a", "b", etc.). If you need to control the match, then
    also supply `ids_to_map` in matching order. E.g.

      tm.ids_to_map = ['d', 'c', 'b', 'a']
      tm.vars_to_map = ['x1', 'x2', 'x3', 'x4']

    Note that this means that if you really want more than one element in the
    tiling to represent the same variable more than once, you can do that.

    `colors_to_use` is a parallel list of named matplotlib colormaps,

      tm.colors_to_use = ["Reds", "Blues", "Greys", "Purples"]

    Similarly, you can specify the classification `schemes_to_use` (such as
    'quantiles') and the number of classes `n_classes` in each.

    If data are categorical, this is flagged in the `categoricals` list of
    booleans, in which case an appropriate colour map should be used. There is
    currently no provision for control of which colour in a categorical
    colour map is applied to which variable level.

    TODO: better control of categorical mapping schemes.

  """

  # these will be set at instantion by Tiling.get_tiled_map()
  tiling:Tiling = None
  """the Tiling with the required tiles"""
  map:gpd.GeoDataFrame = None
  """the GeoDataFrame on which this map is based"""
  ids_to_map:list[str] = None
  """tile_ids that are to be used to represent data"""
  vars_to_map:list[str] = None
  """dataset variables that are to be symbolised"""
  colors_to_use:list[str|list[str]] = None
  """list of matplotlib colormap names."""
  categoricals:list[bool] = None
  """list specifying if each variable is -- or is to be treated as --
  categorical"""
  schemes_to_use:list[str|None] = None
  """mapclassify schemes to use for each variable."""
  n_classes:list[int|None] = None
  """number of classes to apply; if set to 0 will be unclassed."""
  _colourspecs:dict[str,dict] = None
  """dictionary of dictionaries keyed by the items in `ids_to_use` with each
  dictionary forming additional kwargs to be supplied to geopandas.plot()."""
  vmins:[int|float|None] = None
  """list of vmin settings for each variable"""
  vmaxs:[int|float|None] = None
  """list of vmax settings for each variable"""
  add_buffer:bool = False
  """if True then include a buffer of all tiles as a background"""
  buffer_colour:str = "grey"
  """colour of any include buffer layer"""

  # the below parameters can be set either before calling self.render() or
  # passed in as parameters to self.render(). These are solely
  # `TiledMap.render()` options not geopandas plot options.
  legend:bool = True
  """whether or not to show a legend"""
  legend_zoom:float = 1.0
  """<1 zooms out from legend to show more context"""
  legend_dx:float = 0.
  """x shift of legend relative to the map"""
  legend_dy:float = 0.
  """y shift of legend relative to the map"""
  use_ellipse:bool = False
  """if True clips legend with an ellipse"""
  ellipse_magnification:float = 1.0
  """magnification to apply to clip ellipse"""
  radial_key:bool = False
  """if True use radial key even for ordinal/ratio data (normally these will be
  shown by concentric tile geometries)"""
  draft_mode:bool = False
  """if True plot the map coloured by tile_id"""

  # the parameters below are geopandas.plot options which we intercept to
  # ensure they are applied appropriately when we plot a GDF
  figsize:tuple[float,float] = (20, 15)
  """maptlotlib figsize"""
  dpi:float = 72
  """dpi for bitmap formats"""

  def render(
      self,
      **kwargs,
    ) -> Figure:
    """Render the current state to a map.

    Note that TiledMap objects will usually be created by calling
    `Tiling.get_tiled_map()`.

    Args:
      ids_to_map (list[str]): tile_ids to be used in the map. Defaults to None.
      vars_to_map (list[str]): dataset columns to be mapped. Defaults to None.
      colors_to_use (list[str]): list of matplotlib colormaps to be used.
        Defaults to None.
      categoricals (list[bool]): list of flags indicating if associated variable
        should be treated as categorical. Defaults to None.
      schemes_to_use (list[str]): list of strings indicating the mapclassify
        scheme to use e.g. 'equalinterval' or 'quantiles'. Defaults to None.
      n_classes (list[int]): list of ints indicating number of classes to use in
        map classification. Defaults to None.
      legend (bool): If True a legend will be drawn. Defaults to True.
      legend_zoom (float): Zoom factor to apply to the legend. Values <1
        will show more of the tile context. Defaults to 1.0.
      legend_dx (float): x shift to apply to the legend position in plot area
        relative units, i.e. 1.0 is full width of plot. Defaults to 0.0.
      legend_dy (float): x and y shift to apply to the legend position in plot
        area relative units, i.e. 1.0 is full height of plot. Defaults to 0.0.
      use_ellipse (bool): If True applies an elliptical clip to the legend.
        Defaults to False.
      ellipse_magnification (float): Magnification to apply to ellipse clipped
        legend. Defaults to 1.0.
      radial_key (bool): If True legend key for TileUnit maps will be based on
        radially dissecting the tiles, i.e. pie slices. Defaults to False.
      draft_mode (bool): If True a map of the tiled map coloured by tile_ids
        (and with no legend) is returned. Defaults to False.
      figsize (tuple[float,floar]): plot dimensions passed to geopandas.
        plot. Defaults to (20, 15).
      dpi (float): passed to pyplot.plot. Defaults to 72.
      **kwargs: other settings to pass to pyplot/geopandas.plot.

    Returns:
      matplotlib.figure.Figure: figure on which map is plotted.

    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["pdf.use14corefonts"] = True
    matplotlib.rcParams["pdf.fonttype"] = 42

    to_remove = set()  # keep track of kwargs we use to setup TiledMap
    # kwargs with no corresponding class attribute will be discarded
    # because we are using slots, we have to use setattr() here
    for k, v in kwargs.items():
      if k in self.__slots__:
        setattr(self, k, v)
        to_remove.add(k)
    # remove any them so we don't pass them on to pyplot and get errors
    for k in to_remove:
      del kwargs[k]

    if self.draft_mode:
      fig = plt.figure(figsize = self.figsize)
      ax = fig.add_subplot(111)
      self.map.plot(ax = ax, column = "tile_id", cmap = "tab20", **kwargs)
      ax.set_axis_off()
      return fig

    if self.legend:
      # this sizing stuff is rough and ready for now, possibly forever...
      reg_w, reg_h, *_ = \
        tiling_utils.get_width_height_left_bottom(self.map.geometry)
      tile_w, tile_h, *_ = \
        tiling_utils.get_width_height_left_bottom(
          self.tiling.tileable._get_legend_tiles().rotate(
            self.tiling.rotation, origin = (0, 0)))
      sf_w, sf_h = reg_w / tile_w / 3, reg_h / tile_h / 3
      gskw = {"height_ratios": [sf_h * tile_h, reg_h - sf_h * tile_h],
              "width_ratios":  [reg_w, sf_w * tile_w]}

      fig, axes = plt.subplot_mosaic([["map", "legend"], ["map", "."]],
                                        gridspec_kw = gskw,
                                        figsize = self.figsize,
                                        layout = "constrained", **kwargs)
    else:
      fig, axes = plt.subplots(1, 1, figsize = self.figsize,
                               layout = "constrained", **kwargs)
    
    self._plot_map(axes, **kwargs)
    return fig


  def _plot_map(
      self,
      ax:plt.Axes,
      **kwargs,
    ) -> None:
    """Plot map to the supplied axes.

    Args:
      axes (plt.Axes): axes on which maps will be drawn.
      kwargs (dict): additional parameters to be passed to plot.

    """
    self._set_colourspecs()
    bb = self.map.geometry.total_bounds
    if self.legend:
      if self.add_buffer:
        self._plot_buffer(ax["map"], **kwargs)
      if (self.legend_dx != 0 or self.legend_dx != 0):
        box = ax["legend"].get_position()
        box.x0 += self.legend_dx
        box.x1 += self.legend_dx
        box.y0 += self.legend_dy
        box.y1 += self.legend_dy
        ax["legend"].set_position(box)
      ax["map"].set_axis_off()
      ax["map"].set_xlim(bb[0], bb[2])
      ax["map"].set_ylim(bb[1], bb[3])
      self._plot_subsetted_gdf(ax["map"], self.map, **kwargs)
      self.plot_legend(ax = ax["legend"], **kwargs)
    else:
      if self.add_buffer:
        self._plot_buffer(ax, **kwargs)
      ax.set_axis_off()
      ax.set_xlim(bb[0], bb[2])
      ax.set_ylim(bb[1], bb[3])
      self._plot_subsetted_gdf(ax, self.map, **kwargs)


  def _plot_subsetted_gdf(
      self,
      ax:plt.Axes,
      gdf:gpd.GeoDataFrame,
      grouping_var:str = "tile_id",
      **kwargs,
    ) -> None:
    """Repeatedly plot a gpd.GeoDataFrame based on a subsetting attribute.

    NOTE: used to plot both the main map _and_ the legend, which is why
    a separate GeoDataframe is supplied and we don't just use self.map.

    Args:
      ax (pyplot.Axes): axes to plot to.
      gdf (gpd.GeoDataFrame): the GeoDataFrame to plot.

    Raises:
      Exception: if self.colourmaps cannot be parsed exception is raised.

    """
    groups = gdf.groupby(grouping_var)
    for ID, cspec in self._colourspecs.items():
      subset = groups.get_group(ID)
      n_values = len(subset[cspec["column"]].unique())
      if not cspec["categorical"] and n_values == 1:
        print(f"""
              Only one level in variable {cspec['column']}, replacing requested
              colour map with single colour fill.
              """)
        cspec["color"] = matplotlib.colormaps.get(cspec["cmap"])(0.5)
        del cspec["column"]
        del cspec["cmap"]
        del cspec["scheme"]
      elif not cspec["categorical"] and cspec["scheme"] is not None and n_values < cspec["k"]:
        cspec["k"] = n_values
      elif cspec["categorical"]:
        del cspec["scheme"]
      subset.plot(ax = ax, **cspec, **kwargs)


  def to_file(self, fname:str) -> None:
    """Output the tiled map to a layered GPKG file.

    Currently delegates to `weavingspace.tiling_utils.write_map_to_layers()`.

    Args:
      fname (str): Filename to write. Defaults to None.

    """
    tiling_utils.write_map_to_layers(self.map, fname)


  def _plot_buffer(self, ax) -> None:
    buffer = self.map.geometry \
      .buffer(10, cap_style = "square", join_style = "mitre", resolution = 1) \
      .union_all()
    gdf = gpd.GeoDataFrame(
      geometry = gpd.GeoSeries([buffer]), crs = self.map.crs)
    gdf.plot(ax = ax, fc = self.buffer_colour)


  def plot_legend(self, ax, **kwargs) -> None:
    """Plot a legend for this tiled map.

    Args:
      ax (pyplot.Axes, optional): axes to draw legend. Defaults to None.

    """
    ax.set_axis_off()
    legend_tiles = self.tiling.tileable._get_legend_tiles()
    # this is a bit hacky, but we will apply the rotation to text
    # annotation so for TileUnits which don't need it, reverse that now
    if isinstance(self.tiling.tileable, TileUnit):
      # note that this confuses type hinting because pandas silently assigns 
      # a scalar value to a Series
      legend_tiles.rotation = -self.tiling.rotation

    legend_key = self._get_legend_key_gdf(legend_tiles)
    legend_tiles.geometry = legend_tiles.geometry.rotate(
      self.tiling.rotation, origin = (0, 0))

    if self.use_ellipse:
      ellipse = tiling_utils.get_bounding_ellipse(
        legend_tiles.geometry, mag = self.ellipse_magnification)
      bb = ellipse.total_bounds
      c = ellipse.union_all().centroid
    else:
      bb = legend_tiles.geometry.total_bounds
      c = legend_tiles.geometry.union_all().centroid

    # apply legend zoom - NOTE that this must be applied even
    # if self.legend_zoom is == 1...
    ax.set_xlim(c.x + (bb[0] - c.x) / self.legend_zoom,
                c.x + (bb[2] - c.x) / self.legend_zoom)
    ax.set_ylim(c.y + (bb[1] - c.y) / self.legend_zoom,
                c.y + (bb[3] - c.y) / self.legend_zoom)

    for cs, tile, rotn in zip(self._colourspecs.values(),
                              legend_tiles.geometry,
                              legend_tiles.rotation):
      c = tile.centroid
      ax.annotate(cs["column"], xy = (c.x, c.y),
                  ha = "center", va = "center",
                  rotation_mode = "anchor",
                  # adjust rotation to favour text reading left to right
                  rotation = (rotn + self.tiling.rotation + 90) % 180 - 90,
                  bbox = {"lw": 0, "fc": "#ffffff60"})

    # now plot background; we include the central tiles, since in
    # the weave case these may not match the legend tiles
    context_tiles = self.tiling.tileable \
      .get_local_patch(r = 2, include_0 = True) \
      .geometry.rotate(self.tiling.rotation, origin = (0, 0))
    if self.use_ellipse:
      context_tiles.clip(ellipse, keep_geom_type = False).plot(
        ax = ax, fc = "#9F9F9F3F", lw = .35)
      tiling_utils.get_tiling_edges(context_tiles.geometry).clip(
        ellipse, keep_geom_type = True).plot(
          ax = ax, ec = "#5F5F5F", lw = .35)
    else:
      context_tiles.plot(ax = ax, fc = "#9F9F9F3F",
                         ec = "#5F5F5F", lw = .35)
      tiling_utils.get_tiling_edges(context_tiles.geometry).plot(
        ax = ax, ec = "#5F5F5F", lw = .35)

    # plot the legend key tiles (which include the data)
    self._plot_subsetted_gdf(ax, legend_key, **kwargs)


  def _get_legend_key_gdf(self, tiles:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return tiles dissected and with data assigned for use as a legend.

    'Dissection' is handled differently by `WeaveUnit` and `TileUnit`
    objects and delegated to either `WeaveUnit._get_legend_key_shapes()`
    or `TileUnit._get_legend_key_shapes()`.

    Args:
      tiles (gpd.GeoDataFrame): the legend tiles.

    Returns:
      gpd.GeoDataFrame:  with tile_id, variables and rotation
        attributes, and geometries of Tileable tiles sliced into a
        colour ramp or set of nested tiles.

    """
    key_tiles = []   # set of tiles to form a colour key (e.g. a ramp)
    ids = []         # tile_ids applied to the keys
    unique_ids = []  # list of each tile_id used in order
    vals = []        # the data assigned to the key tiles
    rots = []        # rotation of each key tile
    # subsets = self.map.groupby("tile_id")
    for (id, cspec), geom, rot in zip(self._colourspecs.items(),
                                      tiles.geometry, 
                                      tiles.rotation):
      d = list(self.map.loc[self.map.tile_id == id][cspec["column"]])
      # if the data are categorical then it's complicated...
      # if cs["categorical"]:
      #   radial = True and self.radial_key
      #   # desired order of categorical variable is the
      #   # color maps dictionary keys
      #   num_cats = len(cmap)
      #   val_order = dict(zip(cmap.keys(), range(num_cats)))
      #   # compile counts of each category
      #   freqs = [0] * num_cats
      #   for v in list(d):
      #     freqs[val_order[v]] += 1
      #   # make list of the categories containing appropriate
      #   # counts of each in the order needed using a reverse lookup
      #   data_vals = list(val_order.keys())
      #   data_vals = [data_vals[i] for i, f in enumerate(freqs) if f > 0]
      # else: # any other data is easy!
      #   data_vals = sorted(d)
      #   freqs = [1] * len(data_vals)
      data_vals = sorted(d)
      key = self.tiling.tileable._get_legend_key_shapes( # type: ignore
        geom, [1] * len(data_vals), rot, False)
      key_tiles.extend(key)
      vals.extend(data_vals)
      n = len(data_vals)
      ids.extend([id] * n)
      unique_ids.append(id)
      rots.extend([rot] * n)
    # finally make up a data table with all the data in all the columns. This
    # allows us to reuse the tiling_utils.plot_subsetted_gdf() function. To be
    # clear: all the data from all variables are added in all columns but when
    # sent for plotting only the subset associated with each tile_id will get
    # plotted. It's wasteful of space... but note that the same is true of the
    # original data - each tile_id has data for all the variables even if it's
    # not being used to plot them: tables gonna table!
    key_data = {}
    for ID in unique_ids:
      key_data[self.vars_to_map[self.ids_to_map.index(ID)]] = vals
    key_gdf = gpd.GeoDataFrame(
      data = key_data | {"tile_id": ids, "rotation": rots},
      crs = self.map.crs,
      geometry = gpd.GeoSeries(key_tiles))
    key_gdf.geometry = key_gdf.rotate(self.tiling.rotation, origin = (0, 0))
    return key_gdf


  def explore(self) -> None:
    """TODO: add wrapper to make tiled web map via geopandas.explore.
    """
    return None


  def _set_colourspecs(self) -> None:
    """Set _colourspecs dictionary based on instance member variables.

    Set up is to the extent it is possible to follow user requested variables.
    Each requested `ids_to_map` item keys a dictionary in `_colourspecs` which
    contains the `column`, `cmap`, `scheme`, `categorical`, and `k` parameters
    to be passed on for use by the `geopandas.GeoDataFrame.plot()` calls in the
    _plot_subsetted_gdf()` method.

    This is the place to make 'smart' adjustments to how user requests for map
    styling are handled.
    """
    numeric_columns = list(self.map.select_dtypes(
      include = ("float", "int")).columns)
    # note that some numeric columns can be considered categorical
    categorical_columns = list(self.map.select_dtypes(
      include = ("category", "int")).columns)
    try:
      if isinstance(self.ids_to_map, str):
        # wrap a single string in a list - this would be an unusual request...
        if self.ids_to_map in list(self.map.tile_id):
          print("""You have only requested a single attribute to map.
                That's fine, but perhaps not what you intended?""")
          self.ids_to_map = [self.ids_to_map]
        else:
          raise KeyError(
            """You have requested a single non-existent attribute to map!""")
      elif self.ids_to_map is None or not isinstance(self.ids_to_map, Iterable):
        # default to using all of them in order
        print("""No tile ids provided: setting all of them!""")
        self.ids_to_map = sorted(list(set(self.map.tile_id)))

      if self.vars_to_map is None or not isinstance(self.vars_to_map, Iterable):
        self.vars_to_map = []
        if len(numeric_columns) == 0:
          # if there are none then we can't do it
          raise IndexError("""Attempting to set default variables, but
                          there are no numeric columns in the data!""")
        if len(numeric_columns) < len(self.ids_to_map):
          # if there are fewer available than we need then repeat some
          print("""Fewer numeric columns in the data than elements in the 
                tile unit. Reusing as many as needed to make up the numbers""")
          reps = len(self.ids_to_map) // len(numeric_columns) + 1
          self.vars_to_map = (numeric_columns * reps)[:len(self.ids_to_map)]
        elif len(numeric_columns) > len(self.ids_to_map):
          # if there are more than we need let the user know, but trim list
          print("""Note that you have supplied more variables to map than 
                there are distinct elements in the tile unit. Ignoring the
                extras.""")
          self.vars_to_map = numeric_columns[:len(self.ids_to_map)]
        else:
          self.vars_to_map = numeric_columns
      # print(f"{self.vars_to_map=}")

      if self.categoricals is None or not isinstance(self.categoricals, Iterable):
        # provide a set of defaults
        self.categoricals = [col not in numeric_columns for col in self.vars_to_map]
      # print(f"{self.categoricals=}")
      
      if isinstance(self.schemes_to_use, str):
        self.schemes_to_use = [self.schemes_to_use] * len(self.ids_to_map)
      elif self.schemes_to_use is None or not isinstance(self.schemes_to_use, Iterable):
        # provide a set of defaults
        self.schemes_to_use = [None] * len(self.ids_to_map)
      # print(f"{self.schemes_to_use=}")
      
      if isinstance(self.colors_to_use, str):
        self.colors_to_use = [self.colors_to_use] * len(self.ids_to_map)
      elif self.colors_to_use is None or not isinstance(self.colors_to_use, Iterable):
        # provide starter defaults
        print(f"""No colour maps provided! Setting some defaults.""")
        self.colors_to_use = ["Set1" if cat else "Reds" for cat in self.categoricals]
      for i, (col, cat) in enumerate(zip(self.colors_to_use, self.categoricals)):
        if cat and col not in CMAPS_CATEGORICAL:
          self.colors_to_use[i] = CMAPS_CATEGORICAL[i]
        # we'll allow diverging schemes for now...
        elif not cat and col not in CMAPS_SEQUENTIAL and col not in CMAPS_DIVERGING:
          self.colors_to_use[i] = CMAPS_SEQUENTIAL[i]
      # print(f"{self.colors_to_use=}")

      if isinstance(self.n_classes, int):
        if self.n_classes == 0:
          self.n_classes = [None] * len(self.ids_to_map)
        else:
          self.n_classes = [self.n_classes] * len(self.ids_to_map)
      elif self.n_classes is None or not isinstance(self.n_classes, Iterable):
        # provide a set of defaults
        self.n_classes = [None if cat else None for cat in self.categoricals] 
      # print(f"{self.n_classes=}")

      if not isinstance(self.vmins, Iterable):
        self.vmins = [None] * len(self.ids_to_map)

      if not isinstance(self.vmaxs, Iterable):
        self.vmaxs = [None] * len(self.ids_to_map)

    except IndexError as e:
      e.add_note("""One or more of the supplied lists of mapping settings is 
                 an inappropriate length""")
      raise

    self._colourspecs = {
      ID: {"column": v,
           "cmap": c,
           "categorical": cat,
           "scheme": s,
           "k": k,
           "vmin": vmin,
           "vmax": vmax}
      for ID, v, c, cat, s, k, vmin, vmax 
      in zip(self.ids_to_map,
             self.vars_to_map,
             self.colors_to_use,
             self.categoricals,
             self.schemes_to_use,
             self.n_classes, 
             self.vmins,
             self.vmaxs, strict = False)}
