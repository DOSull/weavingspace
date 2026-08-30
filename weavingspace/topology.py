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
import inspect
import itertools
import string
from collections import defaultdict
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely.affinity as affine
import shapely.geometry as geom

from weavingspace import (
  ShapeMatcher,
  Symmetries,
  Tileable,
  Transform,
  tiling_utils,
)

if TYPE_CHECKING:
  from collections.abc import Callable, Iterable

"""Classes for working with topology of tilings.

Together the `Topology`, `Tile`, `Vertex`, `Edge`, and
`weavingspace.symmetry.Symmetries` classes enable extraction of the topological
structure of periodic `weavingspace.tileable.Tileable` objects so that
modification of equivalent tiles can be carried out while retaining
tileability.

It is important to note that these are not fully generalised classes and
methods, that is, the Topology object that is supported is not a permanent
'backing' data structure for our Tileable objects. While it might become that
in time, it is not yet such a data structure. Instead usage is

    tile = TileUnit(...)
    topology = Topology(tile)
    topology = topology.transform_*(...)
    new_tile = topology.tile_unit

A Topology plot function is provided for a user to be able to see what they are
doing, because how edges and vertices in a tiling are labelled under tile
equivalences is an essential step in the process.

Note that these classes do not precisely represent the distinctions made in the
mathematical literature between tiling vertices and tile corners, or between
tiling edges and tile sides.
"""

# all two letter pairs of the alphabet for labelling
# note that it is inconceivable that this many labels will ever be needed!
LABELS = \
  list(string.ascii_uppercase) + ["".join(x) for x in
   itertools.product(
    list(string.ascii_uppercase),
    list(string.ascii_uppercase))]
labels = \
  list(string.ascii_lowercase) + ["".join(x) for x in
   itertools.product(
    list(string.ascii_lowercase),
    list(string.ascii_lowercase))]


class Topology:
  """Class to represent topology of a Tileable object.

  NOTE: It is important that get_local_patch return the tileable elements and
  the translated copies in consistent sequence, i.e. if there are (say) four
  tiles in the unit, the local patch should be 1 2 3 4 1 2 3 4 1 2 3 4 ... and
  so on. This is because self.tiles[i % n_tiles] is frequently used to
  reference the base unit Tile which corresponds to self.tiles[i].
  """

  tileable: Tileable = None
  """the Tileable on which the topology will be based."""
  tiles: list[Tile] = []
  """list of the Tiles in the topology. We use polygons returned by the
  tileable.get_local_patch method for these. That is the base tiles and 8
  adjacent copies (for a rectangular tiling), or 6 adjacent copies (for a
  hexagonal tiling)."""
  points: dict[int, Vertex] = {}
  """dictionary of all points (vertices and corners) in the tiling, keyed by
  Vertex ID."""
  edges: dict[tuple[int,int], Edge] = {}
  """dictionary of the tiling edges, keyed by Edge ID."""
  dual_tiles: dict[int, geom.Polygon] = {}
  """list of geom.Polygons from which a dual tiling might be constructed."""
  n_tiles: int = 0
  """number of tiles in the base Tileable (retained for convenience)."""
  shape_groups: list[list[int]] = []
  """list of lists of tile IDs distinguished by shape and optionally tile_id"""
  tile_matching_transforms: list[tuple[float]] = []
  """shapely transform tuples that map tiles onto other tiles"""
  tile_transitivity_classes: list[tuple[int]] = []
  """list of lists of tile IDs in each transitivity class"""
  vertex_transitivity_classes: list[list[int]] = []
  """list of lists of vertex IDs in each transitivity class"""
  edge_transitivity_classes: list[list[tuple[int]]] = []
  """list of lists of edge IDs in each transitivity class"""

  def __init__(
      self,
      unit: None|Tileable,
      ignore_tile_ids:bool = True,
    ) -> None:
    """Class constructor.

    Args:
      unit (Tileable): the Tileable whose topology is required.
      ignore_tile_ids (bool): (EXPERIMENTAL) if True then only consider the tile
        shapes, not labels. If False consider any labels. Defaults to True.

    """
    # Note that the order of these setup steps is fragile sometimes
    # not obviously so.
    if unit is not None:
      self.tileable = unit # keep this for reference
      self.n_tiles = self.tileable.tiles.shape[0]
      self._initialise_points_into_tiles()
      self._setup_vertex_tile_relations()
      self._setup_edges()
      self._copy_base_tiles_to_patch()
      self._assign_vertex_and_edge_base_IDs()
      self._identify_distinct_tile_shapes(ignore_tile_ids)
      self._find_tile_transitivity_classes(ignore_tile_ids)
      self._find_vertex_transitivity_classes(ignore_tile_ids)
      self._find_edge_transitivity_classes(ignore_tile_ids)
      self.generate_dual()


  def __str__(self) -> str:
    """Return string representation of this Topology.

    Returns:
      str: a message that recommends examining the tiles, points and edges
        attributes.

    """
    return (f"""Topology of Tileable with {self.n_tiles} tiles.\n
            Examine .tiles, .points and .edges for more details.""")


  def __repr__(self) -> str:
    return str(self)


  def _initialise_points_into_tiles(self, debug:bool = False) -> None:
    """Set up dictionary of unique point locations and assign them to Tiles.

    Args:
      debug (bool): if True prints useful debugging information.

    """
    shapes = self.tileable.get_local_patch(r = 1, include_0 = True).geometry
    shapes = [tiling_utils.get_clean_polygon(s) for s in shapes]
    labels = list(self.tileable.tiles.tile_id) * (len(shapes) // self.n_tiles)
    self.tiles = []
    self.points = {}
    for (i, shape), label in zip(enumerate(shapes), labels, strict = True):
      tile = Tile(self, i)
      tile.label = label
      tile.base_ID = tile.ID % self.n_tiles
      self.tiles.append(tile)
      tile.corners = []
      corners = tiling_utils.get_corners(shape, repeat_first = False)
      for c in corners:
        prev_vertex = None
        for p in reversed(self.points.values()):
          if c.distance(p.point) <= 2 * tiling_utils.RESOLUTION:
            # found an already existing vertex, so add to the tile and break
            prev_vertex = p
            tile.corners.append(prev_vertex.ID)
            break
        if prev_vertex is None:
          # new vertex, add it to the topology dictionary and to the tile
          v = self.add_vertex(c)
          tile.corners.append(v.ID)
          if debug:
            print(f"Added new Vertex {v} to Tile {i}")


  def _setup_vertex_tile_relations(self, debug:bool = False) -> None:
    """Determine relations between vertices and tiles.

    In particular vertices along tile edges that are not yet included in their
    list of vertices are added. Meanwhile vertex lists of incident tiles and
    neighbours are set up.

    Args:
      debug (bool): if True prints debugging information.

    """
    # we do this for all tiles in the radius-1 local patch
    for tile in self.tiles:
      if debug:
        print(f"Checking for vertices incident on Tile {tile.ID}")
      corners = []
      # performance is much improved using the vertex IDs
      initial_corner_IDs = tile.get_corner_IDs()
      # we need the current shape (not yet set) to check for incident vertices
      shape = geom.Polygon([c.point for c in tile.get_corners()])
      # get points incident on tile boundary, not already in the tile corners
      new_points = [v for v in self.points.values()
                    if v.ID not in initial_corner_IDs and
                    v.point.distance(shape) <= 2 * tiling_utils.RESOLUTION]
      # iterate over sides of the tile to see which the vertex is incident on
      for c1, c2 in zip(tile.corners, tile.corners[1:] + tile.corners[:1],
                        strict = True):
        all_points = [c1, c2]
        if len(new_points) > 0:
          if debug:
            print(f"{[v.ID for v in new_points]} incident on tile")
          ls = geom.LineString([self.points[c1].point, self.points[c2].point])
          to_insert = [v for v in new_points
                       if v.point.distance(ls) <= 2 * tiling_utils.RESOLUTION]
          if len(to_insert) > 0:
            # sort by distance along the side
            d_along = sorted([(ls.line_locate_point(v.point), v)
                              for v in to_insert], key = lambda x: x[0])
            to_insert = [v for d, v in d_along]
            all_points = all_points[:1] + [v.ID for v in to_insert] + all_points[1:]
        corners.extend(all_points[:-1])
        for x1, x2 in zip(all_points[:-1], all_points[1:], strict = True):
          # x2 will add the tile and neigbour when we get to the next side
          # every vertex gets a turn!
          self.points[x1].add_tile(tile.ID)
          self.points[x1].add_neighbour(x2)
      tile.corners = corners
      tile.set_shape_from_corners()


  def _setup_edges(
      self,
      debug:bool = False,
    ) -> None:
    """Set up the tiling edges.

    First vertices in the base tiles are classified as tiling vertices or not -
    only these can be classified reliably (e.g vertices on the perimeter are
    tricky). Up to here all vertices have been considered tiling vertices.

    Second edges are created by traversing tile corner lists. Edges are stored
    once only by checking for edges in the reverse direction already in the
    edges dictionary. Edge right and left tiles are initialised.

    Third tile edge direction lists are initialised.

    Args:
      debug (bool): if True print debug messages. Defaults to False.

    """
    # classify vertices in the base tiles
    for tile in self.tiles[:self.n_tiles]:
      for v in tile.get_corners():
        v.is_tiling_vertex = len(v.neighbours) > 2
    if debug:
      print("Classified base tile vertices")
    self.edges = {}
    for tile in self.tiles:
      if debug:
        print(f"Adding edges from Tile {tile.ID}")
      tile.edges = []
      vertices = [v for v in tile.get_corners() if v.is_tiling_vertex]
      # note that through here finding ints in lists is much faster than
      # finding Vertex objects, hence we use lists of IDs not Vertex objects
      if len(vertices) > 1:
        for v1, v2 in zip(vertices, vertices[1:] + vertices[:1], strict = True):
          corner_IDs = tile.get_corner_IDs()
          idx1 = corner_IDs.index(v1.ID)
          idx2 = corner_IDs.index(v2.ID)
          if idx1 < idx2:
            corners = corner_IDs[idx1:(idx2 + 1)]
          else:
            corners = corner_IDs[idx1:] + corner_IDs[:(idx2 + 1)]
          ID = (corners[0], corners[-1])
          if ID not in self.edges:
            # check that reverse direction edge is not present first
            r_ID = ID[::-1]
            if r_ID in self.edges:
              # if it is, then add it and set left_tile
              if debug:
                print(f"reverse edge {r_ID} found")
              e = self.edges[r_ID]
              e.left_tile = tile.ID
              tile.edges.append(e.ID)
            else:
              # we've found a new edge so make and add it
              if debug:
                print(f"adding new edge {corners}")
              e = self.add_edge([self.points[c] for c in corners])
              e.right_tile = tile.ID
              tile.edges.append(e.ID)
      # initialise the edge direction information in the tile
      tile.set_edge_directions()


  def _assign_vertex_and_edge_base_IDs(self) -> None:
    """Assign the base_ID attributes of vertices and edges.

    These allow us to determine corrspondences between vertices and edges in
    the 'base' tiles in the Topology tileable, and those we have added at
    radius 1 for labelling and visualisation.
    """
    self._assign_vertex_base_IDs()
    self._assign_edge_base_IDs()


  def _assign_vertex_base_IDs(self) -> None:
    """Assign base_ID attribute of vertices."""
    # assign vertex base_ID frome the core tiles
    for tile0 in self.tiles[:self.n_tiles]:
      for v in tile0.get_corner_IDs():
        self.points[v].base_ID = v
    # assign others from their corresponding vertex in the core
    for t0 in self.tiles[:self.n_tiles]:
      for t1 in self.tiles[self.n_tiles:]:
        if t1.base_ID == t0.base_ID:
          for v0, v1 in zip(t0.get_corners(), t1.get_corners(), strict = True):
            v1.base_ID = v0.base_ID


  def _assign_edge_base_IDs(self) -> None:
    """Assign base_ID attribute of edges, based on base_IDs of vertices."""
    for tile0 in self.tiles[:self.n_tiles]:
      for e in tile0.get_edge_IDs():
        self.edges[e].base_ID = e
    for t0 in self.tiles[:self.n_tiles]:
      for t1 in self.tiles[self.n_tiles:]:
        if t1.base_ID == t0.base_ID:
          for e0, e1 in zip(t0.edges, t1.edges, strict = True):
            self.edges[e1].base_ID = self.edges[e0].base_ID


  def _copy_base_tiles_to_patch(self) -> None:
    """Copy attributes of base tiles to corresponding tiles in radius-1 patch.

    This requires:

    First, inserting any additional corners in the base tiles not found in the
    radius-1 tiles.

    Second, any vertices in the base tiles that are NOT tiling vertices are
    applied to radius-1 tiles leading to merging of some edges.
    """
    # the number of tiles in the base + radius-1
    n_r1 = len(self.tiles)
    # first add any missing vertices to the non-base tiles
    # we add all the missing vertices before doing any merges
    for base in self.tiles[:self.n_tiles]:
      for other in self.tiles[base.ID:n_r1:self.n_tiles]:
        self._match_reference_tile_vertices(base, other)
    # then merge any edges that meet at a corner
    for base in self.tiles[:self.n_tiles]:
      for other in self.tiles[base.ID:n_r1:self.n_tiles]:
        self._match_reference_tile_corners(base, other)


  def _match_reference_tile_vertices(
      self,
      tile1:Tile,
      tile2:Tile,
    ) -> None:
    """Add vertices to tile2 so it matches tile1 adjusting edges as required.

    This assumes the tiles are the same shape, but that tile2 may be missing
    some tiling vertices along some edges.

    Args:
      tile1 (Tile): reference tile.
      tile2 (Tile): tile to change.

    """
    to_add = len(tile1.corners) - len(tile2.corners)
    while to_add > 0:
      # find the reference x-y offset
      dxy = (tile2.centre.x - tile1.centre.x, tile2.centre.y - tile1.centre.y)
      for i, t1c in enumerate([c.point for c in tile1.get_corners()]):
        t2c = tile2.get_corners()[i % len(tile2.get_corners())].point
        if abs((t2c.x - t1c.x) - dxy[0]) > 10 * tiling_utils.RESOLUTION or \
           abs((t2c.y - t1c.y) - dxy[1]) > 10 * tiling_utils.RESOLUTION:
          # add vertex to t2 by copying the t1 vertex appropriately offset
          # note that this might alter the length of t2.corners
          v = self.add_vertex(geom.Point(t1c.x + dxy[0], t1c.y + dxy[1]))
          v.is_tiling_vertex = True
          old_edge, new_edges = tile2.insert_vertex_at(v, i)
          del self.edges[old_edge]
          for e in new_edges:
            self.edges[e.ID] = e
          to_add = to_add - 1


  def _match_reference_tile_corners(
      self,
      tile1:Tile,
      tile2:Tile,
    ) -> None:
    """Make vertices that are corners in tile1 corners in tile2.

    Edges are merged as required.

    Args:
        tile1 (Tile): reference tile.
        tile2 (Tile): tile to make match.

    """
    vs_to_change = [
      vj for vi, vj in zip(tile1.get_corners(), tile2.get_corners(), strict = True)
      if not vi.is_tiling_vertex and vj.is_tiling_vertex]
    if len(vs_to_change) > 0:
      for v in vs_to_change:
        v.is_tiling_vertex = False
        # it's a corner not an edge so will have no more than 2 v.tiles
        old_edges, new_edge = v.tiles[0].merge_edges_at_vertex(v)
        for e in old_edges:
          del self.edges[e]
        self.edges[new_edge.ID] = new_edge

  def _identify_distinct_tile_shapes(
      self,
      ignore_tile_id_labels:bool = True,
    ) -> None:
    """Identify unique tiles based on their symmetries and shapes.

    At the same time assembles a list of the affine transforms under which
    matches occurs since these are potential symmetries of the tiling.

    TODO: reimplement consideration of tile_id

    Args:
      ignore_tile_id_labels (bool): if True only the shape of tiles matters; if
        False the tile_id label is also considered. Defaults to True.

    """
    if ignore_tile_id_labels:
      matches = {}
      offsets = {}
      for tile in self.tiles[:self.n_tiles]:
        matches[tile.base_ID] = [tile.base_ID]
        matched = False
        s = Symmetries(tile.shape)
        for other in self.tiles[:self.n_tiles]:
          if other.ID > tile.ID:
            offset = s.get_corner_offset(other.shape)
            if offset is not None:
              offsets[tile.base_ID] = offset
              matches[tile.base_ID].append(other.base_ID)
              matched = True
        if not matched:
          offsets[tile.base_ID] = 0
      base_groups = list(
        nx.connected_components(nx.from_dict_of_lists(matches)))
      self.shape_groups = []
      for i, group in enumerate(base_groups):
        full_group = []
        for tile in self.tiles:
          if tile.base_ID in group:
            tile.shape_group = i
            tile.offset_corners(offsets[tile.base_ID])
            full_group.append(tile.ID)
        self.shape_groups.append(full_group)
    else:
      self.shape_groups = []
      for ti in self.tiles[:self.n_tiles]:
        self.shape_groups.append(
          [tj.ID for tj in self.tiles if tj.base_ID == ti.base_ID])
      for i, group in enumerate(self.shape_groups):
        for j in group:
          self.tiles[j].shape_group = i

  def _find_tile_transitivity_classes(
      self,
      ignore_tile_id_labels:bool = True,
    ) -> None:
    """Find tiles equivalent under symmetries.

    Also update the tile_matching_transforms attribute to contain only those
    transforms that pass this test.

    Args:
      ignore_tile_id_labels (bool): if True then consider only shapes; if False
        also consider labels of tiles. Defaults to True.

    """
    self.tile_matching_transforms = \
      self.get_potential_symmetries(ignore_tile_id_labels)
    if ignore_tile_id_labels:
      base_tiles = self.tiles[:self.n_tiles]
      # it is quicker (should be!) to only do within shape group tests
      # often there is only one when it will make no difference
      by_group_equivalent_tiles = []
      # maintain a set of transforms still potentially tiling symmetries
      poss_transforms = set(self.tile_matching_transforms.keys())
      # and a dictionary of booleans tracking which transforms are still valid
      eq_under_transform = dict.fromkeys(poss_transforms, True)
      for g, _ in enumerate(self.shape_groups):
        by_group_equivalent_tiles.append(set())
        source_tiles = [tile for tile in base_tiles if tile.shape_group == g]
        target_tiles = [tile for tile in self.tiles if tile.shape_group == g]
        for tr in poss_transforms:
          transform = self.tile_matching_transforms[tr].transform
          matched_tiles = {}
          eq_under_transform[tr] = True
          for source_tile in source_tiles:
            matched_tile_id = self._match_geoms_under_transform(
              source_tile, target_tiles, transform)
            if matched_tile_id == -1:
              eq_under_transform[tr] = False
              break
            matched_tiles[source_tile.ID] = matched_tile_id # actually a base_ID
          if eq_under_transform[tr]:
            for k, v in matched_tiles.items():
              # here we record the transform, in case it is later invalidated
              by_group_equivalent_tiles[g].add((tr, k, v))
        # remove valid transforms that didn't make it through this group
        poss_transforms = {t for t, x in eq_under_transform.items() if x}
      # compile equivalences from all groups made under still valid transforms
      # a dict of sets so singletons aren't lost in finding connected components
      equivalents = {i: set() for i in range(self.n_tiles)}
      for group_equivalents in by_group_equivalent_tiles:
        for (tr, tile_i, tile_j) in group_equivalents:
          if tr in poss_transforms:
            equivalents[tile_i].add(tile_j)
      self.tile_matching_transforms = {
        k: v for k, v in self.tile_matching_transforms.items()
        if k in poss_transforms}
      self.tile_transitivity_classes = []
      equivalents = nx.connected_components(nx.from_dict_of_lists(equivalents))
      for c, base_IDs in enumerate(equivalents):
        transitivity_class = []
        for tile in self.tiles:
          if tile.base_ID in base_IDs:
            transitivity_class.append(tile.ID)
            tile.transitivity_class = c
        self.tile_transitivity_classes.append(transitivity_class)
    else:
      # transitivity classes are just the individual tiles
      self.tile_transitivity_classes = []
      for i, tile in enumerate(self.tiles):
        tile.transitivity_class = tile.base_ID
        if i < self.n_tiles:
          self.tile_transitivity_classes.append([tile.ID])
        else:
          self.tile_transitivity_classes[tile.base_ID].append(tile.ID)


  def get_potential_symmetries(
      self,
      ignore_tile_id_labels:bool = True,
    ) -> dict[int, Transform]:
    """Assemble potential symmetries from symmetries of prototile and tiles.

    Also remove any duplicates that result. The result is assigned to the
    tile_matching_transforms attribute.

    TODO: consider retaining the Symmetry objects as these carry additional
    information that might facilitate labelling under a limited number of the
    symmetries not all of them.

    Returns:
      dict[int, tuple[float]]: dictionary of the symmetries (transforms
        actually) in shapely affine transform 6-tuple format.

    """
    self.tile_matching_transforms = {
      k: Transform("translation", 0, geom.Point(0, 0), v,
                   tiling_utils.get_translation_transform(v[0], v[1]))
      for k, v in enumerate(self.tileable.get_vectors())}
    if ignore_tile_id_labels:
      n_symmetries = len(self.tile_matching_transforms)
      ptile = self.tileable.prototile.loc[0, "geometry"]
      for tr in ShapeMatcher(ptile).get_polygon_matches(ptile):
        if tr.transform_type not in ["identity", "translation"]:
          self.tile_matching_transforms[n_symmetries] = tr
          n_symmetries = n_symmetries + 1
      for tile in self.tiles[:self.n_tiles]:
        for tr in ShapeMatcher(tile.shape).get_polygon_matches(tile.shape):
          if tr.transform_type not in ["identity", "translation"]:
            self.tile_matching_transforms[n_symmetries] = tr
            n_symmetries = n_symmetries + 1
      for tile in self.tiles[:self.n_tiles]:
        sm = ShapeMatcher(tile.shape)
        transforms = [sm.get_polygon_matches(self.tiles[i].shape)
          for i in self.shape_groups[tile.shape_group] if i < self.n_tiles]
        for tr in itertools.chain(*transforms):
          if tr.transform_type not in ["identity", "translation"]:
            self.tile_matching_transforms[n_symmetries] = tr
            n_symmetries = n_symmetries + 1
      self.tile_matching_transforms = self._remove_duplicate_symmetries(
        self.tile_matching_transforms)
    return self.tile_matching_transforms


  def _remove_duplicate_symmetries(
      self,
      transforms:dict[int,Transform],
    ) -> dict[int,Transform]:
    """Filter list of shapely affine transforms to remove duplicates.

    Args:
      transforms (dict[int,Transform]): dictionary of Transforms to filter.

    Returns:
      dict[int,Transform]: the filtered dictionary with duplicates removed.

    """
    uniques = {}
    for k, v in transforms.items():
      already_exists = False
      for u in uniques.values():
        already_exists = np.allclose(
          v.transform, u.transform, atol = 1e-4, rtol = 1e-4)
        if already_exists:
          break
      if not already_exists:
        uniques[k] = v
    return uniques


  def _find_vertex_transitivity_classes(
      self,
      ignore_tile_id_labels:bool = True,
    ) -> None:
    """Find vertex transitivity classes.

    This function checks which vertices align with which others under transforms
    in the tile_matching_transforms attribute. The process need only determine
    the classes for vertices in the core tileable.tiles, then assign those to
    all vertices by matched base_ID.
    """
    if ignore_tile_id_labels:
      equivalent_vertices = defaultdict(set)
      base_vertices = [v for v in
                      self.vertices_in_tiles(self.tiles[:self.n_tiles])
                      if v.is_tiling_vertex]
      for transform in self.tile_matching_transforms.values():
        for v in base_vertices:
          equivalent_vertices[v.ID].add(v.ID)
          match_ID = self._match_geoms_under_transform(
            v, base_vertices, transform.transform)
          if match_ID != -1:
            equivalent_vertices[v.ID].add(match_ID)
      equivalent_vertices = self._get_exclusive_supersets(
        [tuple(sorted(s)) for s in equivalent_vertices.values()])
      self.vertex_transitivity_classes = defaultdict(list)
      for c, vclass in enumerate(equivalent_vertices):
        for v in self.points.values():
          if v.base_ID in vclass:
            v.transitivity_class = c
            self.vertex_transitivity_classes[c].append(v.ID)
      self.vertex_transitivity_classes = list(
        self.vertex_transitivity_classes.values())
      # label vertices based on their transitivity class
      for v in self.points.values():
        if v.is_tiling_vertex:
          v.label = LABELS[v.transitivity_class]
    else:
      self.vertex_transitivity_classes = defaultdict(list)
      for v in self.points.values():
        if v.is_tiling_vertex:
          self.vertex_transitivity_classes[v.base_ID].append(v.ID)
          v.transitivity_class = v.base_ID
      self.vertex_transitivity_classes = list(
        self.vertex_transitivity_classes.values())
      for v in self.points.values():
        if v.is_tiling_vertex:
          v.label = LABELS[v.transitivity_class]


  def _find_edge_transitivity_classes(
      self,
      ignore_tile_id_labels:bool = True,
    ) -> None:
    """Find edge transitivity classes.

    This function works by checking which edges align with which others under
    transforms in the tile_matching_transforms attribute. The process need only
    determine the classes for edges in the core tileable.tiles, then assign
    those to all edges by matched base_ID.

    TODO: Note that this code is identical to the vertex transitivity code
    so it might make sense to merge.
    """
    if ignore_tile_id_labels:
      equivalent_edges = defaultdict(set)
      base_edges = self.edges_in_tiles(self.tiles[:self.n_tiles])
      for transform in self.tile_matching_transforms.values():
        for e in base_edges:
          equivalent_edges[e.ID].add(e.ID)
          match_id = self._match_geoms_under_transform(
            e, base_edges, transform.transform)
          if match_id != -1:
            equivalent_edges[e.ID].add(match_id)
      equivalent_edges = self._get_exclusive_supersets(
        [tuple(sorted(s)) for s in equivalent_edges.values()])
      self.edge_transitivity_classes = defaultdict(list)
      for c, eclass in enumerate(equivalent_edges):
        for e in self.edges.values():
          if e.base_ID in eclass:
            e.transitivity_class = c
            self.edge_transitivity_classes[c].append(e.ID)
      self.edge_transitivity_classes = list(
        self.edge_transitivity_classes.values())
      # label edges based on their transitivity class
      for e in self.edges.values():
        e.label = labels[e.transitivity_class]
    else:
      self.edge_transitivity_classes = defaultdict(list)
      for e in self.edges.values():
        self.edge_transitivity_classes[e.base_ID].append(e.ID)
      self.edge_transitivity_classes = list(
        self.edge_transitivity_classes.values())
      for i, eclass in enumerate(self.edge_transitivity_classes):
        for e in eclass:
          self.edges[e].transitivity_class = i
          self.edges[e].label = labels[i]


  def _match_geoms_under_transform(
      self,
      geom1:Tile|Vertex|Edge,
      geoms2:list[Tile|Vertex|Edge],
      transform:tuple[float,...],
    ) -> int|tuple[int]:
    """Determine if a geometry maps onto any in a patch under supplied symmetry.

    Args:
      geom1 (Tile|Vertex|Edge): element whose geometry we want to match.
      geoms2 (list[Tile|Vertex|Edge]): set of elements among which a
        match is sought.
      transform (tuple[float]): shapely affine transform 6-tuple to apply.

    Returns:
      int|tuple[int,int]: ID of the element in patch that matches the geom1
        element under the transform if one exists, otherwise returns -1. For
        edges note that the ID is a tuple.

    """
    match_id = -1
    for geom2 in geoms2:
      if isinstance(geom1, Tile):
        # an area of intersection based test
        match = self.polygon_matches(
          affine.affine_transform(geom1.shape, transform), geom2.shape)
      elif isinstance(geom1, Vertex):
        # distance test
        match = affine.affine_transform(geom1.point, transform).distance(
          geom2.point) <= 10 * tiling_utils.RESOLUTION
      else: # must be an Edge
        # since edges _should not_ intersect this test should work in
        # lieu of a more complete point by point comparison
        c1 = geom1.get_geometry().centroid
        c2 = geom2.get_geometry().centroid
        match = affine.affine_transform(c1, transform) \
          .distance(c2) <= 10 *tiling_utils.RESOLUTION
      if match:
        return geom2.base_ID
    return match_id


  def _get_exclusive_supersets(
      self, sets:list[Iterable]) -> list[Iterable]:
    """Return sets of elements not found in the same set among those supplied.

    The supplied sets share elements, i.e., they are non-exclusives sets. The
    returned sets are exclusive: each element will only appear in one of the
    sets in the returned list. This is accomplished using networkx's
    connected components applied to a graph where each intersection between two
    sets is an edge.

    Args:
        sets (list[Iterable]): list of lists of possibly overlapping sets.

    Returns:
      list[Iterable]: list of lists that include all the original
        elements without overlaps.

    """
    overlaps = []
    for i, si in enumerate(sets):
      s1 = set(si)
      for j, sj in enumerate(sets):
        s2 = set(sj)
        if len(s1 & s2) > 0:
          overlaps.append((i, j))
    G = nx.from_edgelist(overlaps)
    result = []
    for component in nx.connected_components(G):
      s = set()
      for i in component:
        s = s.union(sets[i])
      result.append(tuple(s))
    return result


  def vertices_in_tiles(self, tiles:list[Tile]) -> list[Vertex]:
    """Get vertices incident on tiles in supplied list.

    Args:
      tiles (list[Tile]): tiles whose vertices are required.

    Returns:
      list[Vertex]: the required vertices.

    """
    vs = set()
    for tile in tiles:
      vs = vs.union(tile.get_corner_IDs())
    return [self.points[v] for v in vs]


  def edges_in_tiles(self, tiles:list[Tile]) -> list[Edge]:
    """Get edges that are part of the boundary of tiles in supplied list.

    Args:
      tiles (list[Tile]): tiles whose edges are required.

    Returns:
      list[Edge]: the required edges.

    """
    es = set()
    for tile in tiles:
      es = es.union(tile.get_edge_IDs())
    return [self.edges[e] for e in es]


  def generate_dual(self) -> list[geom.Polygon]:
    """Create the dual tiiing for the tiling of this Topology.

    TODO: make this a viable replacement for the existing dual tiling
    generation.

    TODO: also need to ensure that this finds a set of dual tiles that exhaust
    the plane...

    Returns:
      list[geom.Polygon]: a list of polygon objects.

    """
    for v in self.points.values():
      v.clockwise_order_incident_tiles()
    self.dual_tiles = {}
    base_id_sets = defaultdict(list)
    for v in self.points.values():
      base_id_sets[v.base_ID].append(v.ID)
    minimal_set = [self.points[min(s)] for s in base_id_sets.values()]
    for v in minimal_set:
    # for v in self.points.values():
      if v.is_interior() and len(v.tiles) > 2:
        self.dual_tiles[v.ID] = \
          geom.Polygon([t.centre for t in v.get_tiles()])


  def get_dual_tiles(self) -> gpd.GeoDataFrame:
    """Return dual tiles as GeoDataFrame."""
    n = len(self.dual_tiles)
    return gpd.GeoDataFrame(
      data = {"tile_id": list(self.tileable.tiles.tile_id)[:n]},
      geometry = gpd.GeoSeries(self.dual_tiles.values()),
      crs = self.tileable.crs)


  def add_vertex(self, pt:geom.Point) -> Vertex:
    """Add and return Vertex at the specified point location.

    No attempt is made to ensure Vertex IDs are an unbroken sequence: a new ID
    is generated one greater than the existing highest ID. IDs will usually be
    an unbroken sequence up to removals when geometry transformations are
    applied.

    Args:
      pt (geom.Point): point location of the Vertex.

    Returns:
      Vertex: the added Vertex object.

    """
    n = 0 if len(self.points) == 0 else max(self.points.keys()) + 1
    v = Vertex(self, pt, n)
    self.points[n] = v
    return v


  def add_edge(self, vs:list[Vertex]) -> Edge:
    """Create an Edge from the suppled vertices and return it.

    The new Edge is added to the edges dictionary. Edges are self indexing by
    the IDs of their end Vertices.

    Args:
      vs (list[Vertex]): list of Vertices in the Edge to be created.

    Returns:
        Edge: the added Edge.

    """
    e = Edge(self, vs)
    self.edges[e.ID] = e
    return e


  def polygon_matches(
      self,
      geom1:geom.Polygon,
      geom2:geom.Polygon,
    ) -> bool:
    """Test if supplied polygons match geometrically.

    Tests for equality of area, and equality of their area of overlap to their
    shared area, i.e. Area1 == Area2 == (Area 1 intersection 2).

    Args:
      geom1 (geom.Polygon): first polygon.
      geom2 (geom.Polygon): second polygon.

    Returns:
      bool: True if the polygons are the same, False otherwise.

    """
    a, b = geom1.area, geom2.area
    return bool(
      np.isclose(a, b,
                 rtol = tiling_utils.RESOLUTION * 100,
                 atol = tiling_utils.RESOLUTION * 100) and
      np.isclose(a, geom1.intersection(geom2).area,
                 rtol = tiling_utils.RESOLUTION * 100,
                 atol = tiling_utils.RESOLUTION * 100))


  def _get_tile_geoms(self) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of the tiles.

    Returns:
      gpd.GeoDataFrame: the tiles as a GeoDataFrame.

    """
    return gpd.GeoDataFrame(
      data = {"transitivity_class": [t.transitivity_class for t in self.tiles]},
      geometry = gpd.GeoSeries([t.shape for t in self.tiles]))


  def _get_tile_centre_geoms(self) -> gpd.GeoSeries:
    """Return the centre of tiles as a GeoDataFrame.

    Returns:
      gpd.GeoDataFrame: the tiles centres.

    """
    return gpd.GeoSeries([t.centre for t in self.tiles])


  def _get_point_geoms(self) -> gpd.GeoSeries:
    """Return tiling vertices and tile corners as a GeoDataFrame of Points.

    Returns:
      gpd.GeoDataFrame: corners and vertices of the tiling.

    """
    return gpd.GeoSeries([v.point for v in self.points.values()])


  def _get_vertex_geoms(self) -> gpd.GeoSeries:
    """Return tiling vertices (not corners) as a GeoDataFrame of Points.

    Returns:
      gpd.GeoDataFrame: vertices of the tiling.

    """
    return gpd.GeoSeries([v.point for v in self.points.values()
                          if v.is_tiling_vertex])


  def _get_edge_geoms(
      self,
      offset:float = 0.0,
    ) -> gpd.GeoSeries:
    """Return tiling edges a GeoSeries of LineStrings optionally offset.

    Offsetting the edges allows their direction to be seen when visualised.

    Returns:
      gpd.GeoDataFrame: edges of the tiling.

    """
    return gpd.GeoSeries([e.get_topology().parallel_offset(offset)
                          for e in self.edges.values()])


  def plot(
      self,
      show_original_tiles: bool = True,
      show_tile_centres: bool = False,
      show_vertex_labels: bool = True,
      show_vertex_ids: bool = False,
      show_edges: bool = True,
      offset_edges: bool = True,
      show_edge_labels:bool = False,
      show_dual_tiles: bool = False,
    ) -> plt.Axes:
    """Delegate plotting of requested elements and return plt.Axes.

    Args:
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
    extent = gpd.GeoSeries([t.shape for t in self.tiles]).total_bounds
    dist = max([extent[2] - extent[0], extent[3] - extent[1]]) / 100
    if show_original_tiles:
      self._plot_tiles(ax)
    if show_tile_centres:
      self._plot_tile_centres(ax)
    if show_vertex_labels:
      self._plot_vertex_labels(ax, show_vertex_ids)
    if show_edge_labels or show_edges:
      self._plot_edges(ax, show_edges, show_edge_labels, dist, offset_edges)
    if show_dual_tiles:
      self._plot_dual_tiles(ax, dist)
    plt.axis("off")
    return ax


  def _plot_tiles(self, ax:plt.Axes) -> plt.Axes:
    """Plot Topology's tiles on supplied Axes.

    Tiles are coloured by equivalence class.

    Args:
      ax (plt.Axes): Axes on which to plot.

    Returns:
      plt.Axes: the Axes.

    """
    self._get_tile_geoms().plot(column = "transitivity_class",
      ax = ax, ec = "#444444", lw = 0.5, alpha = 0.25, cmap = "Greys")
    return ax


  def _plot_tile_centres(self, ax:plt.Axes) -> plt.Axes:
    """Print tile transitivity class at each tile centroid.

    Args:
      ax (plt.Axes): Axes on which to plot.

    Returns:
      plt.Axes: the Axes.

    """
    for _, tile in enumerate(self.tiles):
      ax.annotate(tile.transitivity_class, xy = (tile.centre.x, tile.centre.y),
                  ha = "center", va = "center")
    return ax


  def _plot_vertex_labels(
      self,
      ax:plt.Axes,
      show_vertex_ids:bool = False,
    ) -> plt.Axes:
    """Plot either the Vertex transitivity class label or its sequential ID.

    Args:
      ax (plt.Axes): Axes on which to plot.
      show_vertex_ids (bool, optional): If True plots the ID, else plots the
        transitivity class. Defaults to False.

    Returns:
      plt.Axes: the Axes.

    """
    for v in self.points.values():
      ax.annotate(v.ID if show_vertex_ids else v.label,
                  xy = (v.point.x, v.point.y), color = "k",
                  ha = "center", va = "center")
    return ax


  def _plot_edges(
      self,
      ax:plt.Axes,
      show_edges:bool = False,
      show_edge_labels:bool = False,
      dist:float = 0.0,
      offset_edges:bool = True,
    ) -> plt.Axes:
    """Plot edges, including an offset if specified and labels if requested.

    Can also be called to only plot the labels.

    Args:
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
      edges = self._get_edge_geoms(dist if offset_edges else 0)
      edges.plot(ax = ax, color = "dodgerblue", ls = ":")
    else:
      edges = [e.get_geometry() for e in self.edges.values()]
    if show_edge_labels:
      for ls, e in zip(edges, self.edges.values(), strict = True):
        c = ls.centroid
        ax.annotate(e.label, xy = (c.x, c.y), color = "k",
                    ha = "center", va = "center")
    return ax

  def _plot_dual_tiles(self, ax:plt.Axes,
                       dist:float = 0.0) -> plt.Axes:
    gpd.GeoSeries(self.dual_tiles).buffer(
      -dist / 4, join_style = "mitre", cap_style = "square").plot(
        ax = ax, fc = "g", alpha = 0.25)
    return ax


  def plot_tiling_symmetries(
      self,
      **kwargs:float,
    ) -> None:
    """Plot the symmetries of Topology's tiling.

    Most of the work here is delegated to `_plot_tiling_symmetry` which is run
    once per symmetry on a grid of plt.Axes built by this function.

    Args:
      kwargs: passed through to `_plot_tiling_symmetry`

    """
    n = len(self.tile_matching_transforms)
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))
    fig = plt.figure(figsize = (12, 12 * nr / nc))
    for i, tr in enumerate(self.tile_matching_transforms.values()):
      ax = fig.add_subplot(nr, nc, i + 1)
      self._plot_tiling_symmetry(tr, ax, **kwargs)


  def _plot_tiling_symmetry(
      self,
      transform:Transform,
      ax:plt.Axes,
      **kwargs:float,
    ) -> None:
    """Plot the supplied Transform on the supplied plt.Axes.

    Args:
        transform (Transform): the Transform to plot.
        ax (plt.Axes): the Axes on which to plot.
        kwargs: passed through to plt.plot functions.

    """
    tiles = gpd.GeoSeries([t.shape for t in self.tiles])
    base_tiles = tiles[:self.n_tiles]
    tiles.plot(ax = ax, fc = "k", alpha = .15, ec = "k", lw = .5)
    base_tiles.plot(ax = ax, fc = "#00000000", ec = "w", lw = 1, zorder = 2)
    transformed_base_tiles = gpd.GeoSeries(
      [transform.apply(g) for g in base_tiles])
    transformed_base_tiles.plot(ax = ax, fc = "k", alpha = .2, lw = 0, ec = "k")
    transform.draw(ax, **kwargs) # delegate to Transform.draw
    plt.axis("off")


  def transform_geometry(
      self,
      new_topology:bool,
      apply_to_tiles:bool,
      selector:str,
      transform_type:str,
      **kwargs:float) -> Topology:
    r"""Get a new Topology by applying specified transformation.

    A transformation specified by `transform_type` and keyword arguments is
    applied to elements in the Topology whose labels match the selector
    parameter. The transform is optionally applied to update tiles and
    optionally requests a new Topology object.

    Implemented in this way so that transformations can be applied one at a time
    without creating an intermediate set of new tiles, which may be invalid and
    fail. So, if you wish to apply (say) 3 transforms and generate a new
    Topology leaving the existing one intact:

        new_topo = old_topo.transform_geometry(True,  False, "a", ...) \
                           .transform_geometry(False, False, "B", ...) \
                           .transform_geometry(False, True,  "C", ...)

    The first transform requests a new Topology, subsequent steps do not, and it
    is only the last step which attempts to create the new tile polygons.

    **kwargs supply named parameters for the requested transformation.

    Args:
      new_topology (bool): if True returns a new Topology object, else returns
        the current Topology modified.
      apply_to_tiles (bool): if True attempts to create new Tiles after the
        transformation has been applied. Usually set to False, unless the last
        transformation in a pipeline, to avoid problems of topologically invalid
        tiles at intermediate steps.
      selector (str): label of elements to which to apply the transformation.
        Note that all letters in the supplied string are checked, so you can
        use e.g. "abc" to apply a transformation to edges labelled "a", "b" or
        "c", or "AB" for vertices labelled "A" or "B".
      transform_type (str): name of the type of transformation requested.
        Currently supported are `zigzag_edge`, `rotate_edge`, `scale_edge`,
        `push_vertex`, and `nudge_vertex`. Keyword arguments for each are
        documented in the corresponding methods.
      kwargs: contains any needed arguments of the requested transform_type.

    Returns:
      Topology: if new_topology is True a new Topology based on this one with
        after transformation, if False this Topology is returned after the
        transformation.

    """
    print("CAUTION: new Topology will probably not be correctly labelled. "
          "To build a correct Topology, extract the tileable attribute and "
          "rebuild Topology from that.")
    topo = (copy.deepcopy(self)) if new_topology else self
    transform_args = topo.get_kwargs(getattr(topo, transform_type), **kwargs)
    match transform_type:
      case "zigzag_edge":
        for e in topo.edges.values():
          if e.label in selector:
            topo.zigzag_edge(e, **transform_args)
      case "rotate_edge":
        for e in topo.edges.values():
          if e.label in selector:
            topo.rotate_edge(e, **transform_args)
      case "scale_edge":
        for e in topo.edges.values():
          if e.label in selector:
            topo.scale_edge(e, **transform_args)
      case "push_vertex":
        pushes = {}
        for v in topo.vertices_in_tiles(topo.tiles[:topo.n_tiles]):
          if v.label in selector:
            pushes[v.base_ID] = topo.push_vertex(v, **transform_args)
        for base_ID, (dx, dy) in pushes.items():
          for v in [v for v in topo.points.values() if v.base_ID == base_ID]:
            v.point = affine.translate(v.point, dx, dy)
      case "nudge_vertex":
         for v in topo.points.values():
          if v.label in selector:
            topo.nudge_vertex(v, **transform_args)
    if apply_to_tiles:
      for t in topo.tiles:
        t.set_corners_from_edges()
    topo.tileable.tiles.geometry = gpd.GeoSeries(
      [topo.tiles[i].shape for i in range(topo.n_tiles)])
    topo.tileable._setup_regularised_prototile()
    return topo


  def get_kwargs(
      self,
      fn:Callable,
      **kwargs:str|float,
    ) -> dict:
    """Filter the supplied kwargs to only contain arguments required by fn.

    Args:
      fn (Callable): the function that is to be inspected.
      **kwargs (str|float): kwargs to be filtered.

    Returns:
      str|float: filtered dictionary of kwargs.

    """
    args = inspect.signature(fn).parameters
    return {k: kwargs.pop(k) for k in dict(kwargs) if k in args}


  def zigzag_edge(
      self,
      edge:Edge,
      start:str = "A",
      n:int = 2,
      h:float = 0.5,
      smoothness:int = 0,
    ) -> None:
    """Apply zigzag transformation to supplied Edge.

    Currently this will only work correctly if n is even.

    TODO: make it possible for odd numbers of 'peaks' to work (this may require
    allowing bidirectional Edges, i.e. storing Edges in both directions so that
    all Tile edges are drawn CW). The `start` parameter is a temporary hack for
    this.

    Args:
      edge (Edge): Edge to transform
      start (str, optional): label at one end of edge which is used to determine
        the sense of h, enabling C-curves with an odd number n of zigs and zags
        to be applied. Defaults to 'A'.
      n (int, optional): number of zigs and zags in the edge. Defaults to 2.
      h (float, optional): width of the zig zags relative to edge length.
        Defaults to 0.5.
      smoothness (int, optional): spline smoothness. 0 gives a zig zag proper,
        higher values will produce a sinusoid. Defaults to 0.

    """
    v0, v1 = edge.get_vertices()[0], edge.get_vertices()[1]
    if n % 2 == 1 and v0.label != start:
      h = -h
    ls = self.zigzag_between_points(v0.point, v1.point, n, h, smoothness)
    # remove current corners
    self.points = {k: v for k, v in self.points.items()
                   if k not in edge.get_corner_IDs()[1:-1]}
    # add the new ones
    new_corners = [self.add_vertex(geom.Point(xy)).ID for xy in ls.coords]
    edge.corners = edge.vertices[:1] + new_corners + edge.vertices[-1:]
    if edge.right_tile is not None:
      self.tiles[edge.right_tile].set_corners_from_edges(False)
    if edge.left_tile is not None:
      self.tiles[edge.left_tile].set_corners_from_edges(False)


  def zigzag_between_points(
      self,
      p0:geom.Point,
      p1:geom.Point,
      n:int,
      h:float = 1.0,
      smoothness:int = 0,
    ) -> geom.LineString:
    """Return a zig zag line optionally smoothed as a spline between two points.

    Args:
      p0 (geom.Point): start point.
      p1 (geom.Point): end point.
      n (int): number of zig zags.
      h (float, optional): amplitude of zig zags relative to distance between
        points. Defaults to 1.0.
      smoothness (int, optional): number of spline smoothed points to add. If
        set to 0 a straight line zig zag is produced. Defaults to 0.

    Returns:
      geom.LineString: the resulting zig zag line.

    """
    r = p0.distance(p1)
    xs = np.linspace(0, n * np.pi, (n + smoothness) * 2 + 1, endpoint = True)
    ys = [np.sin(x) for x in xs]

    sfx = 1 / max(xs) * r
    sfy = h * r / 2
    theta = np.arctan2(p1.y - p0.y, p1.x - p0.x)

    ls = geom.LineString(
      [geom.Point(x, y) for x, y in zip(xs, ys, strict = True)])
    ls = affine.translate(ls, 0, -(ls.bounds[1] + ls.bounds[3]) / 2)
    ls = affine.scale(ls, xfact = sfx, yfact = sfy, origin = (0, 0))
    ls = affine.rotate(ls, theta, (0, 0), use_radians = True)
    x0, y0 = next(iter(ls.coords))
    return affine.translate(ls, p0.x - x0, p0.y - y0)


  def rotate_edge(
      self,
      edge:Edge,
      centre:str = "",
      angle:float = 0,
    ) -> None:
    """Rotate edge.

    Args:
      edge (Edge): the edge to rotate.
      centre (str): centre of rotation which should be label of one of its
        end point vertices or "" when rotation will be about the Edge centroid.
        Defaults to "".
      angle (float): angle of rotation.

    """
    v0, v1 = edge.get_vertices()
    ls = geom.LineString([v0.point, v1.point])
    if v0.label == centre:
      c = v0.point
    elif v1.label == centre:
      c = v1.point
    else:
      c = ls.centroid
    ls = affine.rotate(ls, angle, origin = c)
    v0.point, v1.point = [geom.Point(c) for c in ls.coords]


  def scale_edge(
      self,
      edge:Edge,
      sf:float = 1.0,
    ) -> None:
    """Scale edge.

    Args:
      edge (Edge): the edge to scale.
      sf (float): amount by which to scale the edge.

    """
    v0, v1 = edge.get_vertices()
    ls = geom.LineString([v0.point, v1.point])
    ls = affine.scale(ls, xfact = sf, yfact = sf, origin = ls.centroid)
    v0.point, v1.point = [geom.Point(c) for c in ls.coords]


  def push_vertex(
      self,
      vertex:Vertex,
      push_d:float,
    ) -> tuple[float]:
    """Return displacement vector to push a vertex based on incident edges.

    Args:
      vertex (Vertex): the vertex to push.
      push_d (float): the distance to push it.

    """
    neighbours = [self.points[v] for v in vertex.neighbours]
    dists = [vertex.point.distance(v.point) for v in neighbours]
    x, y = vertex.point.x, vertex.point.y
    unit_vectors = [((x - v.point.x) / d, (y - v.point.y) / d)
                    for v, d in zip(neighbours, dists, strict = True)]
    return  (push_d * sum([xy[0] for xy in unit_vectors]),
             push_d * sum([xy[1] for xy in unit_vectors]))


  def nudge_vertex(
      self,
      vertex:Vertex,
      dx:float,
      dy:float,
    ) -> None:
    """Nudge vertex by specified displacement.

    Args:
      vertex (Vertex): the vertext to nudge.
      dx (float): x displacement.
      dy (float): y displacement.

    """
    vertex.point = affine.translate(vertex.point, dx, dy)


class Tile:
  """Class to represent essential features of polygons in a tiling."""

  topology: Topology
  ID: int
  """integer ID number which indexes the Tile in the containing Topology tiles
  list."""
  base_ID: int
  """ID of corresponding Tile in the base tileable unit"""
  corners: list[int]
  """list of Vertex IDs. This includes all corners of the original polygon
  and any tiling vertices induced by (for example) a the corner of an adjacent
  tile lying halfway along an edge of the original polygon on which this tile
  is based. Vertex objects are stored in strictly clockwise sequence."""
  edges: list[tuple[int]]
  """list of Edge IDs objects that together compose the tile boundary."""
  edges_CW: list[bool]
  """list of Edge direction. Edges are stored only once in a Topology so some
  edges are in clockwise order and others  are in counter-clockwise order.
  These boolean flags are True if the corresponding Edge is clockwise, False if
  counter-clockwise."""
  label: str
  """tile_id label from the tileable source"""
  shape: geom.Polygon = None
  """the tile geometry (which may include some redundant points along sides
  where neighbouring tiles induce a tiling vertex). So for example a rectangle
  might have additional points along its sides:

        +---+-------+
        |   |   2   |
        | 1 A---B---E---+
        |   |   |   4   |
        +---C 3 D-------+
            |   |
            +---+

  In the above Tile 1 has additional point A, 2 has B and 3 has C and D induced
  by the corners of neighbouring tiles."""
  centre: geom.Point = None
  """a point centre for the Tile (determined by weavingspace.tiling_utils.
  incentre)."""
  shape_group: int = None
  """the tile shape group of this tile in its containing Topology."""
  transitivity_class: int = None
  """the tile transitivity class of this tile its containing Topology"""

  def __init__(
      self,
      topology:Topology,
      ID:int,
    ) -> None:
    """Class constructor.

    Args:
      ID (int): sequence number ID of this vertex in the Topology.

    """
    self.topology = topology
    self.ID = ID
    self.corners = []
    self.edges = []
    self.edges_CW = []


  def __str__(self) -> str:
    """Return string representation of the Tile.

    Returns:
      str: string including Tile ID, list of corner vertex IDs and list of
        edge IDs.

    """
    return (f"Tile {self.ID} Corners: {self.get_corner_IDs()} "
            f"Edges: {self.get_edge_IDs()}")


  def __repr__(self) -> str:
    return str(self)


  def get_corner_IDs(self) -> list[int]:
    """Return list of corner IDs (not Vertex objects).

    Returns:
      list[int]: list of integer IDs of tile corners.

    """
    return self.corners


  def get_corners(self) -> list[Vertex]:
    return [self.topology.points[v] for v in self.corners]


  def get_edge_IDs(self) -> list[tuple[int,int]]:
    """Return list of edge IDs (not Edge objects).

    Returns:
      list[tuple[int]]: list of 2-element tuples of the start and end Vertex IDs
        of each edge.

    """
    return self.edges


  def get_edges(self) -> list[Edge]:
    return [self.topology.edges[ij] for ij in self.edges]


  def set_shape_from_corners(self) -> None:
    """Set the shape attribute based on corners, and associated tile centre."""
    self.shape = geom.Polygon([c.point for c in self.get_corners()])
    self.centre = tiling_utils.get_incentre(
      tiling_utils.get_clean_polygon(self.shape))


  def set_corners_from_edges(
      self,
      update_shape:bool = True,
    ) -> None:
    """Set corners attribute from the edges attribute.

    Typically called after modification of topology edges. Optionally the shape
    attribute is NOT updated, which may save time when multiple changes to the
    edges of a tile are in process (i.e., only update the shape after all
    changes are complete).

    Args:
      update_shape (bool, optional): if True the shape attribute will be
        updated, otherwise not. Defaults to True.

    """
    self.corners = []
    for e, cw in zip(self.get_edges(), self.edges_CW, strict = True):
      if cw: # clockwise to extend by all but the first corner
        self.corners.extend(e.get_corner_IDs()[:-1])
      else: # counter-clockwise so extend in reverse
        self.corners.extend(e.get_corner_IDs()[1:][::-1])
    if update_shape:
      self.set_shape_from_corners()


  def set_edge_directions(self) -> None:
    """Set up edges_CW attribute by inspection of the edges list.

    It is (frankly!) hard to keep track of the correct sequence of CW/CCW order
    of edges as new ones are created or old ones merged. This method inspects
    the 'tail-head' relations between consecutive edges to set these flags
    correctly.

    The test is simply to check if the 'tail' Vertex ID in each edge appears
    in the ID tuple of the following edge, i.e. if successive edge
    IDs are (0, 1) (2, 1) or (0, 1) (1, 2), then edge (0, 1) is in clockwise
    direction, but if we have (0, 1) (2, 3) then it is not.
    """
    edge_IDs = self.edges
    self.edges_CW = [e1[-1] in e2 for e1, e2 in
                     zip(edge_IDs, edge_IDs[1:] + edge_IDs[:1], strict = True)]


  def insert_vertex_at(
      self,
      vx:Vertex,
      i:int,
      update_shape:bool = False,
    ) -> tuple[tuple[int,int],tuple[tuple[int,int],...]]:
    """Insert the Vertex into tile at index position i.

    Both corners and edges attributes are updated, and the old edge IDs for
    removal and the new edge itself are returned to the calling context (the
    containing Topology) for update of its edges collection. Optionally update
    the shape attribute.

    This is NOT a generic vertex insertion method: it is only for use during
    Topology initialisation, and does not guarantee correct maintenance of
    all tile, edge and vertex relations in the general case---at any rate it
    has not been tested for this!

    Args:
      v (Vertex): the Vertex to insert.
      i (int): index position in current corners after which to insert
        supplied Vertex.
      update_shape (bool, optional): if True shape attribute is updated.
        Defaults to False.

    Returns:
      tuple: the (tuple) ID of the old edge which should be deleted, and
        the new Edges arising from insertion of this Vertex.

    """
    v = vx.ID
    self.corners = [*self.corners[:i], v, *self.corners[i:]]
    old_edge = self.get_edges()[i - 1]
    # store current ID of the affected edge for return to calling context
    old_edge_ID = old_edge.ID
    new_edges = [e.ID for e in old_edge.insert_vertex(v, self.corners[i - 1])]
    self.edges = self.edges[:(i-1)] + new_edges + self.edges[i:]
    self.set_edge_directions()
    if update_shape:
      self.set_shape_from_corners()
    return old_edge_ID, new_edges


  def merge_edges_at_vertex(
      self,
      vx:Vertex,
    ) -> tuple:
    """Merge edges that meet at the supplied Vertex.

    It is assumed that only two tiles are impacted this one, and its neighbour
    across the Edge on which v lies. Both are updated. For this reason the work
    is delegated to `get_updated_edges_from_merge` which is run on both affected
    tiles, but only determines the edges to remove and the new edge to be added
    once. See that method for details.

    Args:
      v (Vertex): Vertex at which to merge Edges. This should currently be an
        end

    Returns:
      tuple: 2 item list of the edge IDs to be removed and a new Edge object to
        be added by the calling context (i.e. the containing Topology).

    """
    v = vx.ID
    to_remove, new_edge = self.get_updated_edges_from_merge(v)
    if len(v.tiles) > 1:
      v.tiles[1].get_updated_edges_from_merge(v, new_edge)
    return to_remove, new_edge


  def get_updated_edges_from_merge(
      self,
      vx:Vertex,
      new_edge:Edge = None,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], Edge]|None:
    """Update edges and edges_CW attributes based on insertion of Vertex.

    If new_edge is supplied then the neighbour tile at v has already created
    the needed new Edge and this Edge is the one that will be 'slotted in' at
    the appropriate spot in the edges list.

    The edges_CW is also updated to maintain correct directions of the edges.
    The corners attribute is unaffected by these changes.

    Args:
      v (Vertex): existing Vertex at which to carry out the merge.
      new_edge (Edge, optional): if another Tile has already carried out this
        merge this should be the resulting new Edge for insertion into this
        Tile. Defaults to None (when the new Edge will be constructed).

    Returns:
      None|tuple: either None (if a new edge was supplied) or a tuple
        of the two edge IDs to be removed and the new edge added for return to
        the calling context (i.e. the containing Topology).

    """
    # get the two edge list index positions in which Vertex v is found
    i, j = self.get_edge_IDs_including_vertex(vx)
    if new_edge is None: # then we must make a new one
      # also record existing edge IDs to be removed
      to_remove = [self.edges[i], self.edges[j]]
      new_edge = self.get_merged_edge(i, j)
      return_edge_updates = True
    else:
      return_edge_updates = False
    if abs(i - j) != 1:
      # edge indices 'wrap' around from end of edge list to start so drop
      # first and last current edges and stick new one on at the end
      self.edges = [*self.edges[1:-1], new_edge.ID]
    else:
      # insert new edge into list in place of the two old ones
      self.edges = [*self.edges[:i], new_edge.ID, *self.edges[j + 1:]]
    # update the edge directions
    self.set_edge_directions()
    if return_edge_updates:
      return to_remove, new_edge
    return None


  def get_edge_IDs_including_vertex(
      self,
      vx:Vertex,
    ) -> tuple[int]:
    """Get the (two) index positions of the edges that include supplied Vertex.

    Args:
        v (Vertex): Vertex of interest.

    Returns:
      tuple[int]: two index positions of Edges in edges list that contain v.

    """
    return (i for i, e in enumerate(self.edges) if vx.ID in e.ID)


  def get_merged_edge(self, i:int, j:int) -> Edge:
    """Return edge made by merging existing edges at i and j in the edges list.

    For example, if the current list of edge IDs was

        (0 1 2) (4 2) (4 5) (5 0)

    and the merge requested is 0 and 1, the resulting new edge is constructed
    from vertices (0 1 2 4).

    Returns:
      Edge: the requested new Edge.

    """
    # if i and j are not consecutive, then j is predecessor edge
    if abs(i - j) != 1:
      i, j = j, i
    # get edges and their directions
    ei, ej = self.edges[i], self.edges[j]
    CWi, CWj = self.edges_CW[i], self.edges_CW[j]
    # DON'T MESS WITH THIS!!!
    # for predecessors (the head) we want everything including the Vertex
    # where the merge is occurring; for successors (the tail) we want all but
    # the first Vertex (which is the one where the merge is occurring). In both
    # cases contingent on whether existing Edges are CW or CCW we may need to
    # flip the Vertex sequence to ensure that the merge Vertex is in the middle
    # of the new edge that will be created
    head = ei.corners if CWi else ei.corners[::-1]
    tail = ej.corners[1:] if CWj else ej.corners[::-1][1:]
    v_sequence = (head if CWi else head[::-1]) + (tail if CWj else tail[::-1])
    return Edge(v_sequence)


  def offset_corners(
      self,
      offset:int,
    ) -> None:
    """Shift shape, corners, edges, and edges_CW by an offset amount.

    This is used to align tiles that are similar, which is required for correct
    transfer of 'base' tile labelling on to 'radius 1' tiles during Topology
    construction.

    Args:
      offset (int): the number of positions to shift the lists.

    """
    if offset is not None or offset != 0:
      self.corners = self.corners[offset:] + self.corners[:offset]
      self.shape = geom.Polygon([c.point for c in self.get_corners()])
      self.edges = self.edges[offset:] + self.edges[:offset]
      self.edges_CW = self.edges_CW[offset:] + self.edges_CW[:offset]


  def angle_at(self, vx:Vertex) -> float:
    """Return interior angle at the specified corner (in degrees).

    Args:
        v (Vertex): corner where angle is requested.

    Returns:
        float: angle at corner v in degrees.

    """
    v = vx.ID
    i = self.corners.index(v)
    n = len(self.corners)
    return tiling_utils.get_inner_angle(
      self.topology.points[self.corners[i-1]].point,
      self.topology.points[self.corners[i]].point,
      self.topology.points[self.corners[(i + 1) % n]].point)


class Vertex:
  """Class to store attributes of a vertex in a tiling."""

  topology: Topology
  point: geom.Point
  """point (geom.Point): point location of the vertex."""
  ID: int
  """integer (mostly but not necessarily in sequence) of vertex keyed into the
  points dictionary of the containing Topology."""
  tiles: list[int]
  """list of Tile IDs incident on this vertex."""
  neighbours: list[int]
  """list of the immediately adjacent other corner IDs. Only required to
  determine if a point is a tiling vertex (when it will have) three or more
  neighbours, so only IDs are stored."""
  base_ID: int = 1_000_000
  """ID of corresponding Vertex in the tileable base_unit"""
  transitivity_class: int = None
  """transitivity class of the vertex under symmetries of the tiling"""
  label: str = ""
  """the (upper case letter) label of the vertex under the symmetries of the
  tiling."""
  is_tiling_vertex: bool = True
  """is_tiling_vertex (bool): True if this is a tiling vertex, rather than a
  tile corner. E.g., A below is a corner, not a tiling vertex. B is a tiling
  vertex:

      +-------+
      | 1     |
      |   A---B---+
      |   | 2     |
      +---C   +---+
          |   |
          +---+"""


  def __init__(
      self,
      topology:Topology,
      point:geom.Point,
      ID:int,
    ) -> None:
    """Class constructor.

    Args:
      point (geom.Point): point location of the vertex.
      ID (int): a unique integer ID (which will be its key in the containing
        Topology points dictionary).

    """
    self.topology = topology
    self.point = point
    self.ID = ID
    self.base_ID = self.ID
    self.tiles = []
    self.neighbours = []


  def __str__(self) -> str:
    """Return string representation of Vertex.

    Returns:
        str: string including ID, point and list of incident Tiles.

    """
    return f"Vertex {self.ID} at {self.point} Tiles: {self.get_tile_IDs()}"


  def __repr__(self) -> str:
    return str(self)


  def get_tile_IDs(self) -> list[int]:
    """Return list of Tile IDs (not the Tiles themselves).

    Returns:
        list[int]: list of Tile IDs

    """
    return self.tiles

  
  def get_tiles(self) -> list[Tiles]:
    return [self.topology.tiles[t] for t in self.tiles]


  def add_tile(self, tile:int) -> None:
    """Add supplied Tile to the tiles list if it is not already present.

    Args:
        tile (Tile): Tile to add.

    """
    if tile not in self.tiles:
      self.tiles.append(tile)


  def add_neighbour(
      self,
      vertex_id:int,
    ) -> None:
    """Add supplied ID to the neighbours list if it is not already present.

    Args:
      vertex_id (int): ID to add to the neighbours list.

    """
    if vertex_id not in self.neighbours:
      self.neighbours.append(vertex_id)


  def clockwise_order_incident_tiles(self) -> None:
    """Reorder tiles list clockwise (this is for dual tiling construction)."""
    cw_order = self._order_of_pts_cw_around_centre(
      [t.centre for t in self.get_tiles()], self.point)
    self.tiles = [self.tiles[i] for i in cw_order]


  def is_interior(self) -> bool:
    """Test if vertex is completely enclosed by its incident Tiles.

    Based on summing the interior angles of the incident tiles at this vertex.

    Returns:
        bool: True if vertex is completely enclosed by incident Tiles.

    """
    return abs(360 - sum([t.angle_at(self) for t in self.get_tiles()])) \
                     < tiling_utils.RESOLUTION


  def _order_of_pts_cw_around_centre(
      self,
      pts:list[geom.Point],
      centre:geom.Point,
    ) -> list[int]:
    """Return order of points clockwise relative to centre point.

    Args:
      pts (list[geom.Point]): list of points to order.
      centre (geom.Point): centre relative to which CW order is determined.

    Returns:
      list[int]: list of indices of reordered points.

    """
    dx = [p.x - centre.x for p in pts]
    dy = [p.y - centre.y for p in pts]
    angles = [np.arctan2(dy, dx) for dx, dy in zip(dx, dy, strict = True)]
    d = dict(zip(angles, range(len(pts)), strict = True))
    return [i for angle, i in sorted(d.items(), reverse=True)]


class Edge:
  """Class to represent edges in a tiling (not tile sides)."""

  topology: Topology
  ID: tuple[int]
  """IDs of the vertices at ends of the edge. Used as key in the containing
  Topology's edges dictionary."""
  vertices: list[int]
  """two item list of the end vertices."""
  corners: list[Vertex]
  """list of all the vertices in the edge (including its end vertices). In a
  'normal' edge to edge tiling corners and vertices will be identical."""
  right_tile: int = None
  """the tile to the right of the edge traversed from its first to its last
  vertex. Given clockwise winding default, all edges will have a right_tile."""
  left_tile: int = None
  """the tile to the left of the edge traversed from its first to its last
  vertex. Exterior edges of the tiles in a Topology will not have a left_tile.
  """
  base_ID: tuple[int] = (1_000_000, 1_000_000)
  """ID of corresponding edge in the base tileable"""
  transitivity_class: int = None
  """transitivity class of the edge under symmetries of the tiling"""
  label: str = ""
  """the (lower case letter) label of the edge under the symmetries of the
  tiling."""

  def __init__(
      self,
      topology:Topology,
      corners:list[Vertex],
    ) -> None:
    """Class constructor.

    Initialises the corners and vertices lists and sets ID to (vertices[0].ID,
    vertices[1].ID). The vertices list is all the corners with is_tiling_vertex
    property True -- Note that during initialisation the default of this
    property is True until after the relations between tiles and vertices have
    been determined.

    Args:
      corners (list[Vertex]): list of all corners along the edge.

    """
    self.topology = topology
    self.corners = [v.ID for v in corners]
    self.vertices = [v.ID for v in corners if v.is_tiling_vertex]
    self.ID = tuple(self.vertices)


  def __str__(self) -> str:
    """Return a string representation of the Edge.

    Returns:
      str: include ID and a list of corner vertex IDs.

    """
    return f"Edge {self.ID} Corners: {self.corners}"


  def __repr__(self) -> str:
    return str(self)


  def get_corner_IDs(self) -> list[int]:
    """Return the IDs of edge corners.

    Returns:
      list[int]: IDs of all corners.

    """
    return self.corners
  

  def get_corners(self) -> list[Vertex]:
    return [self.topology.points[v] for v in self.corners]


  def get_vertex_IDs(self) -> list[int]:
    """Return IDs of edge vertices.

    Returns:
      list[int]: list of IDs of the vertices.

    """
    return self.vertices

  
  def get_vertices(self) -> list[Vertex]:
    return [self.topology.points[v] for v in self.vertices]


  def insert_vertex(self, v:Vertex, predecessor:Vertex) -> list[Edge]:
    """Insert vertex after predecessor and return modified edge and new edge.

    If the initial edge was (say) (0 1 2 5) and the predecessor was set to 1
    the returned edges would be (0 1 v) and (v 2 5).
    """
    v, pred = vx.ID, predecessor.ID
    i = self.corners.index(pred)
    new_edge = Edge([v, *self.corners[i + 1:]])
    if self.right_tile is not None:
      new_edge.right_tile = self.right_tile
    if self.left_tile is not None:
      new_edge.left_tile = self.left_tile
    self.corners = [*self.corners[:i + 1], v]
    self.vertices = [self.vertices[0], v]
    self.ID = tuple(v for v in self.vertices)
    return [self, new_edge]


  def get_geometry(
      self,
      forward:bool = True,
    ) -> geom.LineString:
    """Return geom.LineString representing geometry (including corners).

    Args:
      forward (bool, optional): if True the returned LineString starts at
        corners[0], else at corners[-1]. Defaults to True.

    Returns:
      geom.LineString: the required LineString.

    """
    if forward:
      return geom.LineString([v.point for v in self.get_corners()])
    return geom.LineString([v.point for v in self.get_corners()[::-1]])


  def get_topology(
      self,
      forward:bool = True,
    ) -> geom.LineString:
    """Return LineString connecting vertices of this edge.

    Args:
      forward (bool, optional): if True LineString starts at vertices[0],
        else at vertices[1]. Defaults to True.

    Returns:
        geom.LineString: the required LineString.

    """
    if forward:
      return geom.LineString([v.point for v in self.get_vertices()])
    return geom.LineString([v.point for v in self.get_vertices()[::-1]])
