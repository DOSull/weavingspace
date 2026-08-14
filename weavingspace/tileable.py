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

import copy
from dataclasses import dataclass
from enum import Enum

import geopandas as gpd
import numpy as np
import shapely.affinity as affine
import shapely.geometry as geom
from matplotlib import pyplot as plt

from weavingspace import tiling_utils

"""Implements Tileable and TileShape.

`Tileable` should not be called directly, but is instead accessed from the
`weavingspace.tile_unit.TileUnit` or `weavingspace.weave_unit.WeaveUnit`
constructors.

Several methods of `weavingspace.tileable.Tileable` are generally useful and
can be accessed through its subclasses.
"""

class TileShape(Enum):
  """The available base tile shapes.

  NOTE: the TRIANGLE type does not persist, but should be converted to a
  DIAMOND or HEXAGON type during `Tileable` construction.
  """

  RECTANGLE = "rectangle"
  HEXAGON = "hexagon"
  TRIANGLE = "triangle"
  DIAMOND = "diamond"


@dataclass
class Tileable:
  """Class to represent a tileable set of tile geometries."""

  tiles:gpd.GeoDataFrame|None = None
  """the geometries with associated `title_id` attribute encoding their
  different colouring."""
  prototile:gpd.GeoDataFrame|None = None
  """the tileable polygon (rectangle, hexagon or diamond)"""
  spacing:float = 1000.0
  """the tile spacing effectively the resolution of the tiling. Defaults to
  1000"""
  base_shape:TileShape = TileShape.RECTANGLE
  """the tile shape. Defaults to 'RECTANGLE'"""
  vectors:dict[tuple[int,...],tuple[float,...]]|None = None
  """translation vector symmetries of the tiling"""
  regularised_prototile:gpd.GeoDataFrame|None = None
  """polygon containing the tiles of this tileable, usually a union of its
  tile polygons"""
  crs:int = 3857
  """coordinate reference system of the tile. Most often an ESPG code but
  any valid geopandas CRS specification is valid. Defaults to 3857 (i.e. Web
  Mercator)."""
  rotation:float = 0.0
  """cumulative rotation of the tileable."""
  debug:bool = False
  """if True prints debug messages. Defaults to False."""

  # Tileable constructor called by subclasses - should not be used directly
  def __init__(self,      # noqa: D107
               **kwargs,  # noqa: ANN003
              ) -> None:
    for k, v in kwargs.items():
      if isinstance(v, str):
        # Make any string arguments lower case. Note: this means WeaveUnits can
        # support 'only' 26 distinct tile_id values.
        self.__dict__[k] = v.lower()
      else:
        self.__dict__[k] = v
    # delegate making the tiles back to the subclass _setup_tiles() method
    # which is implemented differently by TileUnit and WeavUnit. It will return
    # a message if there's a problem.
    # There might be a try... except... way to do this more 'properly', but we
    # prefer to return something even if it's not what was requested - along
    # with an explanation / suggestion
    message = self._setup_tiles()
    if message is not None: # there was a problem
      print(message)
      self._setup_default_tileable()
    else:
      self.prototile = self.get_prototile_from_vectors()
      self._setup_regularised_prototile()


  def setup_vectors(
        self,
        *args,  # noqa: ANN002
      ) -> None:
    """Set up translation vectors of a Tileable.

    Initialised from either two or three supplied tuples. Two non-parallel
    vectors are sufficient for a tiling to work, but usually three will be
    supplied for tiles with a hexagonal base tile. We also store the reverse
    vectors - this is for convenience when finding a 'local patch'. This method
    is preferred during Tileable initialisation.

    The vectors are stored in a dictionary indexed by their
    coordinates, e.g.

      {( 1,  0): ( 100, 0), ( 0,  1): (0,  100),
       (-1,  0): (-100, 0), ( 0, -1): (0, -100)}

    For a tileable of type `TileShape.HEXAGON`, the indexing tuples
    have three components. See https://www.redblobgames.com/grids/hexagons/
    """
    vectors = list(args)
    # extend list to include the inverse vectors too
    for v in list(args):
      vectors = [*vectors, (-v[0], -v[1])]
    if len(vectors) == 4:
      i = [1, 0, -1,  0]
      j = [0, 1,  0, -1]
      self.vectors = {
        (i, j): v for i, j, v in zip(i, j, vectors, strict = True)}
    else:
      i = [ 0,  1,  1,  0, -1, -1]
      j = [ 1,  0, -1, -1,  0,  1]
      k = [-1, -1,  0,  1,  1,  0]
      self.vectors = {
        (i, j, k): v for i, j, k, v in zip(i, j, k, vectors, strict = True)}


  def get_vectors(
      self,
      as_dict: bool = False,
      ) -> list[tuple[float,...]]|dict[tuple[int,...],tuple[float,...]]:
    """Return symmetry translation vectors as floating point pairs.

    Optionally returns the vectors in a dictionary indexed by offsets in grid
    coordinates, e.g.

      {( 1,  0): ( 100, 0), ( 0,  1): (0,  100),
       (-1,  0): (-100, 0), ( 0, -1): (0, -100)}

    Returns:
      dict[tuple[int],tuple[float]]|list[tuple[float]]: either the vectors as a
        list of float tuples, or a dictionary of those vectors indexed by
        integer coordinate tuples.

    """
    if as_dict:
      return self.vectors
    return list(self.vectors.values())


  def get_prototile_from_vectors(self) -> gpd.GeoDataFrame:
    r"""Contruct and returns a prototile unit based on vectors of the Tileable.

    For rectangular tilings the prototile is formed by points
    at diagonal corners defined by the halved vectors. By inspection, each edge
    of the prototile is the resultant of adding two of the four vectors.

      ----
     |\  /|
     | \/ |
     | /\ |
     |/  \|
      ----

    In the hexagonal case we form three such quadrilaterals (but don't halve the
    vectors, because we need the extended length) and intersect them to find a
    hexagonal shape. This guarantees that each vector will connect two opposite
    faces of the hexagon, as desired. This seems the most elegant approach by
    geometric construction.

    The prototile is not uniquely defined. The shape returned by this method is
    not guaranteed to be the most 'obvious' one that a human might construct!

    Returns:
      gpd.GeoDataFrame: A suitable prototile shape for the tiling wrapped in a
        GeoDataFrame.

    """
    vecs = self.get_vectors()
    if len(vecs) == 4:
      v1, v2, v3, v4 = [(x / 2, y / 2) for (x, y) in vecs]
      prototile = geom.Polygon([(v1[0] + v2[0], v1[1] + v2[1]),
                                (v2[0] + v3[0], v2[1] + v3[1]),
                                (v3[0] + v4[0], v3[1] + v4[1]),
                                (v4[0] + v1[0], v4[1] + v1[1])])
    else:
      v1, v2, v3, v4, v5, v6 = vecs
      q1 = geom.Polygon([v1, v2, v4, v5])
      q2 = geom.Polygon([v2, v3, v5, v6])
      q3 = geom.Polygon([v3, v4, v6, v1])
      prototile = q3.intersection(q2).intersection(q1)
    return gpd.GeoDataFrame(
      geometry = gpd.GeoSeries([prototile]),
      crs = self.crs)


  def _regularise_tiles(self) -> None:
    """Combine tiles with same tile_id into single tiles.

    This is used where some tile fragments might be on opposite sides of the
    initial Tileable, but would end up touching after tiling. Most likely to be
    applicable to WeaveUnit Tileables.

    Also adjusts the `Tileable.regularised_prototile` attribute accordingly.
    """
    self.regularised_prototile = copy.deepcopy(self.prototile)
    # This preserves order while finding uniques, unlike list(set()).
    # Reordering ids might cause confusion when colour palettes
    # are not assigned explicitly to each id, but in the order
    # encountered in the tile_id Series of the GeoDataFrame.
    tiles, tile_ids = [], []
    ids = list(self.tiles.tile_id.unique())
    for ID in ids:
      fragment_set = list(
        self.tiles[self.tiles.tile_id == ID].geometry)
      merge_result = self._merge_fragments(fragment_set)
      tiles.extend(merge_result)
      tile_ids.extend([ID] * len(merge_result))

    self.tiles = gpd.GeoDataFrame(
      data = {"tile_id": tile_ids},
      crs = self.crs,
      geometry = gpd.GeoSeries([tiling_utils.get_clean_polygon(t)
                                for t in tiles]))

    self.regularised_prototile = \
      self.regularised_prototile.explode(ignore_index = True)
    if self.regularised_prototile.shape[0] > 1:
      self.regularised_prototile.geometry = tiling_utils.get_largest_polygon(
        self.regularised_prototile.geometry)


  def _merge_fragments(
      self,
      fragments:list[geom.Polygon],
    ) -> list[geom.Polygon]:
    """Merge a set of polygons if they touch under the translation vectors.

    Called by `regularise_tiles()` to combine tiles in a tile unit that
    may be fragmented as supplied but will combine after tiling into single
    tiles. This step makes for more efficient implementation of the
    tiling of map regions, and also adds to the woven look in particular
    where it means that 'threads' go beyond the edges of the tile shapes.

    Args:
      fragments (list[geom.Polygon]): A set of polygons to merge.

    Returns:
      list[geom.Polygon]: A minimal list of merged polygons.

    """
    if len(fragments) == 1:
      return [f for f in fragments if not f.is_empty]
    fragments = [f for f in fragments if not f.is_empty]
    prototile = self.prototile.loc[0, "geometry"]
    reg_prototile = copy.deepcopy(
      self.regularised_prototile.loc[0, "geometry"])
    changes_made = True
    while changes_made:
      changes_made = False
      for v in self.vectors.values():
        # empty list to collect the new fragments
        # assembled in this iteration
        next_frags = []
        t_frags = [affine.translate(f, v[0], v[1]) for f in fragments]
        # build a set of any near matching pairs of
        # fragments and their translated copies
        matches = set()
        for i, f1 in enumerate(fragments):
          for j, f2, in enumerate(t_frags):
            if i < j and tiling_utils.touch_along_an_edge(f1, f2):
              matches.add((i, j))
        # determine which of these when unioned has the larger area in common
        # with the prototile
        done_frags = set()
        for i, j in matches:
          f1, f2 = fragments[i], t_frags[j]
          u1 = f1.buffer(tiling_utils.RESOLUTION * 2,
                         join_style = "mitre", cap_style = "square").union(
            f2.buffer(tiling_utils.RESOLUTION * 2,
                      join_style = "mitre", cap_style = "square"))
          u2 = affine.translate(u1, -v[0], -v[1])
          if prototile.intersection(u1).area > prototile.intersection(u2).area:
            u1 = u1.buffer(-tiling_utils.RESOLUTION * 2,
                           join_style = "mitre", cap_style = "square")
            u2 = u2.buffer(-tiling_utils.RESOLUTION * 2,
                           join_style = "mitre", cap_style = "square")
            next_frags.append(u1)
            reg_prototile = reg_prototile.union(u1).difference(u2)
          else:
            u1 = u1.buffer(-tiling_utils.RESOLUTION * 2,
                           join_style = "mitre", cap_style = "square")
            u2 = u2.buffer(-tiling_utils.RESOLUTION * 2,
                           join_style = "mitre", cap_style = "square")
            next_frags.append(u2)
            reg_prototile = reg_prototile.union(u2).difference(u1)
          changes_made = True
          done_frags.add(i)
          done_frags.add(j)
        fragments = [f for i, f in enumerate(fragments)
                     if i not in done_frags] + next_frags
    self.regularised_prototile.loc[0, "geometry"] = reg_prototile
    return [f for f in fragments if not f.is_empty] # don't return any duds


  def _augment_vectors(self) -> list[tuple[float]]:
    """Get vectors with 'diagonal' and inverse vectors added to current set.

    This may be useful if code is changed to work with only two translation
    vectors for all tilings, particularly in the _merge_fragments code, which
    has to check for fragments that merge due to 'diagonal' translation
    neighbours.

    Note: not currently used.

    Returns:
      list(tuple[float]): the augmented set of vectors.

    """
    v = list(self.vectors.values())
    v0 = v[0]
    v1 = v[1]
    v2 = tuple(x + y for x, y in zip(v1, v0, strict = True))
    v3 = tuple(x - y for x, y in zip(v1, v0, strict = True))
    return [
      v0, v1, v2, v3,
      tuple(-x for x in v0),
      tuple(-x for x in v1),
      tuple(-x for x in v2),
      tuple(-x for x in v3)
    ]


  def get_local_patch(
      self,
      r:int = 1,
      include_0:bool = False,
    ) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with translated copies of the Tileable.

    The geodataframe takes the same form as the `Tileable.tile` attribute.

    Args:
      r (int, optional): the number of 'layers' out from the unit to
        which the translate copies will extendt. Defaults to `1`.
      include_0 (bool, optional): If True includes the Tileable itself at
        (0, 0). Defaults to `False`.

    Returns:
      gpd.GeoDataFrame: A GeoDataframe of the tiles extended by a number
        of 'layers'.

    """
    # a dictionary of all the vectors we need, starting with (0, 0)
    three_vecs = len(next(iter(self.vectors.keys()))) == 3
    vecs = {(0, 0, 0): (0, 0)} if three_vecs else {(0, 0): (0, 0)}
    steps = r if three_vecs else r * 2
    # a dictionary of the last 'layer' of added vectors
    last_vecs = copy.deepcopy(vecs)
    # get the translation vectors in a dictionary indexed by coordinates
    # we keep track of the sum of vectors using the (integer) coordinates
    # to avoid duplication of moves due to floating point inaccuracies
    vectors = self.get_vectors(as_dict = True)
    for i in range(steps):
      new_vecs = {}
      for k1, v1 in last_vecs.items():
        for k2, v2 in vectors.items():
          # add the coordinates to make a new key...
          new_key = tuple([k1[i] + k2[i] for i in range(len(k1))])
          # if we haven't reached here before store the actual vector
          if new_key not in vecs:
            new_vecs[new_key] = (v1[0] + v2[0], v1[1] + v2[1])
      # extend the vectors and set the last layer to the set just added
      vecs = vecs | new_vecs
      last_vecs = new_vecs
    if not include_0:  # throw away the identity vector
      vecs.pop((0, 0, 0) if three_vecs else (0, 0))
    ids, tiles = [], []
    # we need to add the translated prototiles in order of their distance from
    # tile 0, esp. in the square case, i.e. something like this:
    #
    #      5 4 3 4 5
    #      4 2 1 2 4
    #      3 1 0 1 3
    #      4 2 1 2 4
    #      5 4 3 4 5
    #
    # this is important for topology detection, where filtering back to the
    # local patch of radius 1 is simplified if prototiles have been added in
    # this order. We use the vector index tuples not the euclidean distances
    # because this is more resistant to odd effects for non-convex tiles
    extent = self.get_prototile_from_vectors().loc[0, "geometry"]
    extent = affine.scale(extent,
                          2 * r + tiling_utils.RESOLUTION,
                          2 * r + tiling_utils.RESOLUTION)
    vector_lengths = {index: np.sqrt(sum([_ ** 2 for _ in index]))
                      for index in vecs}
    ordered_vector_keys = [k for k, v in sorted(vector_lengths.items(),
                                                key = lambda item: item[1])]
    for k in ordered_vector_keys:
      v = vecs[k]
      if geom.Point(v[0], v[1]).within(extent):
        ids.extend(self.tiles.tile_id)
        tiles.extend(
          self.tiles.geometry.apply(affine.translate, xoff = v[0], yoff = v[1]))
    return gpd.GeoDataFrame(
      data = {"tile_id": ids}, crs=self.crs,
      geometry = gpd.GeoSeries(tiles))


  # applicable to both TileUnits and WeaveUnits
  def inset_tiles(
      self,
      inset:float = 0) -> "Tileable":
    """Return a new Tileable with an inset applied around the tiles.

    Works by applying a negative buffer of specfied size to all tiles.
    Tiles that collapse to zero area are removed and the tile_id
    attribute updated accordingly.

    NOTE: this method is likely to not preserve the relative area of tiles.

    Args:
      inset (float, optional): The distance to inset. Defaults to `0`.

    Returns:
      "Tileable": the new inset Tileable.

    """
    inset_tiles, inset_ids = [], []
    for p, ID in zip(self.tiles.geometry, self.tiles.tile_id, strict = True):
      b = p.buffer(-inset, join_style = "mitre", cap_style = "square")
      if not b.area <= 0:
        inset_tiles.append(b)
        inset_ids.append(ID)
    result = copy.deepcopy(self)
    result.tiles = gpd.GeoDataFrame(
      data={"tile_id": inset_ids},
      crs=self.crs,
      geometry=gpd.GeoSeries(inset_tiles),
    )
    return result


  def scale_tiles(
      self,
      sf:float = 1,
      individually:bool = False,
    ) -> "Tileable":
    """Scales the tiles by the specified factor, centred on (0, 0).

    Args:
      sf (float, optional): scale factor to apply. Defaults to 1.
      individually (bool, optional): if True scaling is applied to each tiling
        element centred on its centre, rather than with respect to the Tileable.
        Defaults to False.

    Returns:
      TileUnit: the scaled TileUnit.

    """
    if individually:
      self.tiles.geometry = gpd.GeoSeries(
        [affine.scale(g, sf, sf) for g in self.tiles.geometry])
    else:
      self.tiles.geometry = self.tiles.geometry.scale(sf, sf, origin = (0, 0))
    return self


  def transform_scale(
      self,
      xscale:float = 1.0,
      yscale:float = 1.0,
      independent_of_tiling:bool = False,
    ) -> "Tileable":
    """Transform tileable by scaling.

    Args:
      xscale (float, optional): x scale factor. Defaults to 1.0.
      yscale (float, optional): y scale factor. Defaults to 1.0.
      independent_of_tiling (bool, optional): if True Tileable is scaled while
        leaving the translation vectors untouched, so that it can change size
        independent from its spacing when tiled. Defaults to False.

    Returns:
      Tileable: the transformed Tileable.

    """
    result = copy.deepcopy(self)
    result.tiles.geometry = self.tiles.geometry.scale(
      xscale, yscale, origin=(0, 0))
    if not independent_of_tiling:
      result.prototile.geometry = self.prototile.geometry.scale(
        xscale, yscale, origin=(0, 0))
      result.regularised_prototile.geometry = \
        self.regularised_prototile.geometry.scale(xscale, yscale, origin=(0, 0))
      result._set_vectors_from_prototile()
    return result


  def transform_rotate(
      self,
      angle:float = 0.0,
      independent_of_tiling:bool = False,
    ) -> "Tileable":
    """Transform tiling by rotation.

    Args:
      angle (float, optional): angle to rotate by. Defaults to 0.0.
      independent_of_tiling (bool, optional): if True Tileable is rotated while
        leaving the translation vectors untouched, so that it can change
        orientation independent from its position when tiled. Defaults to False.

    Returns:
      Tileable: the transformed Tileable.

    """
    result = copy.deepcopy(self)
    result.tiles.geometry = self.tiles.geometry.rotate(angle, origin=(0, 0))
    if not independent_of_tiling:
      result.prototile.geometry = \
        self.prototile.geometry.rotate(angle, origin=(0, 0))
      result.regularised_prototile.geometry = \
        self.regularised_prototile.geometry.rotate(angle, origin=(0, 0))
      result._set_vectors_from_prototile()
      result.rotation = self.rotation + angle
    return result


  def transform_skew(
      self,
      xa:float = 0.0,
      ya:float = 0.0,
      independent_of_tiling:bool = False,
      rescale:bool = True
    ) -> "Tileable":
    """Transform tiling by skewing.

    Args:
      xa (float, optional): x direction skew. Defaults to 0.0.
      ya (float, optional): y direction skew. Defaults to 0.0.
      independent_of_tiling (bool, optional): if True Tileable is skewed while
        leaving the translation vectors untouched, so that it can change shape
        independent from its situation when tiled. Defaults to False.
      rescale (bool, optional): if True rescales the result so that overall
        size and extent is more or less unaffected by the skew, otherwise the
        simultaneous application of both x and y shears may dramatically alter
        the size of tiling. Defaults to True.

    Returns:
      Tileable: the transformed Tileable.

    """
    result = copy.deepcopy(self)
    area_0 = result.prototile.geometry[0].area
    result.tiles.geometry = (self.tiles.geometry
                             .skew(xa, ya, origin=(0, 0)))
    if not independent_of_tiling:
      result.prototile.geometry = (self.prototile.geometry
                                   .skew(xa, ya, origin=(0, 0)))
      result.regularised_prototile.geometry = (
        self.regularised_prototile.geometry
            .skew(xa, ya, origin=(0, 0)))
      result._set_vectors_from_prototile()
    if rescale and xa != 0 and ya != 0:
      area_1 = result.prototile.geometry[0].area
      sf = np.sqrt(area_0 / area_1)
      return result.transform_scale(sf, sf, independent_of_tiling)
    return result


  def _set_vectors_from_prototile(self) -> None:
    """Set translation vectors by derivation from a prototile shape.

    Intended to be used internally, after a transform_scale, _skew, or _rotate.

    These are 'face to face' vectors of the prototile, so that a hexagonal tile
    will have 3 vectors, not the minimal parallelogram pair. Also sets the
    inverse vectors. See `Tileable.setup_vectors()` for details.
    """
    t = self.prototile.loc[0, "geometry"]
    points = list(t.exterior.coords)[:-1] # each point once only no wrap
    n_pts = len(points)
    vec_dict = {}
    if n_pts == 4:
      vecs = [(q[0] - p[0], q[1] - p[1])
          for p, q in zip(points, points[1:] + points[:1], strict = True)]
      i = [1, 0, -1,  0]
      j = [0, 1,  0, -1]
      vec_dict = {(i, j): v for i, j, v in zip(i, j, vecs, strict = True)}
    elif n_pts == 6:
      vecs = [(q[0] - p[0], q[1] - p[1])
          for p, q in zip(points, points[2:] + points[:2], strict = True)]
      # hex grid coordinates associated with each of the vectors
      i = [ 0,  1,  1,  0, -1, -1]
      j = [ 1,  0, -1, -1,  0,  1]
      k = [-1, -1,  0,  1,  1,  0]
      vec_dict = {
        (i, j, k): v for i, j, k, v in zip(i, j, k, vecs, strict = True)}
    self.vectors = vec_dict


  def plot(
      self,
      ax:plt.Axes = None,
      show_prototile:bool = True,
      show_reg_prototile:bool = True,
      show_ids:str|bool = "tile_id",
      show_vectors:bool = False,
      r:int = 0,
      prototile_edgecolour:str = "k",
      reg_prototile_edgecolour:str = "r",
      vector_edgecolour:str = "k",
      alpha:float = 1.0,
      r_alpha:float = 0.5,
      cmap:list[str]|str|None = None,
      figsize:tuple[float] = (8, 8),
      **kwargs:dict,
    ) -> plt.Axes:
    """Plot Tileable on the supplied axis.

    **kwargs are passed on to matplotlib.plot()

    Args:
      ax (_type_, optional): matplotlib axis to draw to. Defaults to None.
      show_prototile (bool, optional): if `True` show the tile outline.
        Defaults to `True`.
      show_reg_prototile (bool, optional): if `True` show the regularised tile
        outline. Defaults to `True`.
      show_ids (str, optional): if `tile_id` show the tile_ids. If
        `id` show index number. If None or `''` don't label tiles.
        Defaults to `tile_id`.
      show_vectors (bool, optional): if `True` show the translation
        vectors (not the minimal pair, but those used by
        `get_local_patch()`). Defaults to `False`.
      r (int, optional): passed to `get_local_patch()` to show context if
        greater than 0. Defaults to `0`.
      r_alpha (float, optional): alpha setting for units other than the
        central one. Defaults to 0.5.
      prototile_edgecolour (str, optional): outline colour for the tile.
        Defaults to `"k"`.
      reg_prototile_edgecolour (str, optional): outline colour for the
        regularised. Defaults to `"r"`.
      vector_edgecolour (str, optional): colour for the translation vectors.
        Defaults to `"k"`.
      cmap (list[str], optional): colour map to apply to the central
        tiles. Defaults to `None`.
      figsize (tuple[float], optional): size of the figure.
        Defaults to `(8, 8)`.

    Returns:
      pyplot.axes: to which calling context may add things.

    """
    w = self.prototile.loc[0, "geometry"].bounds[2] - \
      self.prototile.loc[0, "geometry"].bounds[0]
    n_cols = len(set(self.tiles.tile_id))
    if cmap is None:
      if n_cols > 12:
        cm = "Spectral"
      elif n_cols > 8:
        cm = "Paired"
      else:
        cm = "Dark2"
    else:
      cm = cmap
    if ax is None:
      ax = self.tiles.plot(
        column="tile_id", cmap=cm, figsize=figsize, alpha = alpha, **kwargs)
    else:
      self.tiles.plot(
        ax=ax, column="tile_id", cmap=cm, figsize=figsize, **kwargs)
    if show_ids not in [None, ""]:
      do_label = True
      if show_ids in ["tile_id", True]:
        labels = self.tiles.tile_id
      elif show_ids == "id":
        labels = [str(i) for i in range(self.tiles.shape[0])]
      else:
        do_label = False
      if do_label:
        for ID, tile in zip(labels, self.tiles.geometry, strict = True):
          ax.annotate(ID, (tile.centroid.x, tile.centroid.y),
            ha = "center", va = "center", bbox = {"lw": 0, "fc": "#ffffff40"})
    if r > 0:
      self.get_local_patch(r=r).plot(
        ax = ax, column = "tile_id", alpha = r_alpha, cmap = cm, **kwargs)
    if show_prototile:
      self.prototile.plot(ax = ax, ec = prototile_edgecolour, lw = 0.5,
                          fc = "#00000000", **kwargs)
    if show_vectors:  # note that arrows in mpl are dimensioned in plotspace
      vecs = self.get_vectors()
      for v in vecs[: len(vecs) // 2]:
        ax.arrow(0, 0, v[0], v[1], color = vector_edgecolour, width = w * 0.002,
          head_width = w * 0.05, length_includes_head = True, zorder = 3)
    if show_reg_prototile:
      self.regularised_prototile.plot(
        ax = ax, ec = reg_prototile_edgecolour, fc = "#00000000",
        lw = 1.5, zorder = 2, **kwargs)
    return ax


  def _get_legend_tiles(self) -> gpd.GeoDataFrame:
    """Return the tiles augmented by a rotation column.

    This base implementation may be overridden by specific tile unit types.
    In particular see `weavingspace.weave_unit.WeaveUnit._get_legend_tiles()`.

    Returns:
      gpd.GeoDataFrame: the tiles GeoDataFrame with a rotation column added.

    """
    tiles = copy.deepcopy(self.tiles)
    tiles["rotation"] = 0
    return tiles


  def _setup_default_tileable(self) -> None:
    """Set up a default Tileable for when the TileUnit generators fail."""
    # if we've somehow got to here without a base_shape, make it rectangular
    if self.base_shape is None:
      self.base_shape = TileShape.RECTANGLE
    if self.spacing is None:
      self.spacing = 1000
    match self.base_shape:
      case TileShape.HEXAGON:
        ids = ["a"]
        tiles = [tiling_utils.get_regular_polygon(self.spacing, 6)]
        self.setup_vectors(*self._get_hex_vectors())
      case TileShape.DIAMOND:
        ids = ["a"]
        tiles = [( self.spacing/2, 0), (0,  self.spacing * np.sqrt(3)/2),
                 (-self.spacing/2, 0), (0, -self.spacing * np.sqrt(3)/2)]
        self.setup_vectors(*self._get_diamond_vectors())
      case TileShape.TRIANGLE:
        ids = ["a", "b"]
        t1 = tiling_utils.get_regular_polygon(self.spacing, 3)
        t1 = affine.translate(t1, 0, -t1.bounds[1])
        t2 = affine.rotate(t1, 180, origin = (0, 0))
        tiles = [t1, t2]
        self.setup_vectors(*self._get_diamond_vectors())
      case _:
        ids = ["a"]
        tiles = [tiling_utils.get_regular_polygon(self.spacing, 4)]
        self.setup_vectors(*self._get_square_vectors())
    self.tiles = gpd.GeoDataFrame(data = {"tile_id": ids},
                                  geometry = gpd.GeoSeries(tiles),
                                  crs = 3857)
    self.crs = 3857
    self.prototile = self.get_prototile_from_vectors()
    self.regularised_prototile = copy.deepcopy(self.prototile)


  def _get_hex_vectors(self) -> list[tuple[float,float]]:
    """Return three vectors for a hexagonal tiling.

    Returns:
        list[tuple[float]]: Translation vectors of a hexagonal tiling.

    """
    return [(v[0] * self.spacing, v[1] * self.spacing)
            for v in [(0, 1), (np.sqrt(3)/2, 1/2), (np.sqrt(3)/2, -1/2)]]


  def _get_square_vectors(self) -> list[tuple[float,float]]:
    """Return two vectors for a square tiling.

    Returns:
        list[tuple[float]]: Translation vectors of a square tiling.

    """
    return [(v[0] * self.spacing, v[1] * self.spacing)
            for v in [(1, 0), (0, 1)]]


  def _get_diamond_vectors(self) -> list[tuple[float,float]]:
    """Return two vectors for a diamond or triangular tiling.

    Returns:
        list[tuple[float]]: Translation vectors of a square tiling.

    """
    return [(v[0] * self.spacing, v[1] * self.spacing)
            for v in [(1/np.sqrt(3), 1), (1/np.sqrt(3), -1)]]
