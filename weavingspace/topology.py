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
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely.affinity as affine
import shapely.geometry as geom

from weavingspace import (
  Edge,
  ShapeMatcher,
  Symmetries,
  Tile,
  Tileable,
  Transform,
  Vertex,
  tiling_utils,
)

if TYPE_CHECKING:
  from collections.abc import Callable, Iterable

"""Classes for working with topology of tilings.

Together the `Topology` and `weavingspace.symmetry.Symmetries` classes enable
extraction of the topological structure of periodic
`weavingspace.tileable.Tileable` objects so that modification of equivalent
tiles can be carried out while retaining tileability.

It is important to note that the Topology object that is supported is not a
permanent 'backing' data structure for our Tileable objects. While it might
become that in time, it is not yet such a data structure. Instead usage is

    tile = TileUnit(...)
    topology = Topology(tile)
    topology = topology.transform_*(...)
    new_tile = topology.tileable

A Topology plot function is provided for a user to be able to see what they are
doing, because how edges and vertices in a tiling are labelled under tile
equivalences is an essential step in the process.

Topology relies on the Vertex, Edge, and Tile classes implemented in
weavingspace.element.py. These classes do not precisely represent distinctions
in the mathematical literature between tiling vertices and tile corners, or
between tiling edges and tile sides.

IMPORTANT: only the `Topology` class directly references collections of the
other three classes `Vertex`, `Edge`, and `Tile`, apart from a single reference
each of these contains back to its containing Topology instance. This reduces
circular references, which made applying `copy.deepcopy()` to a Topology object
impossible, necessitating the use of `pickle`, which might be unwelcome in some
settings (e.g., a QGIS plugin). Thus all references in `Vertex`, `Edge`, and
`Tile` are by IDs, which are assigned as instances are created during Topology
construction.
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


class ElementType(Enum):
  VERTEX = 0
  EDGE = 1
  TILE = 2


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
  tiles: list[Tile]
  """list of the Tiles in the topology. We use polygons returned by the
  tileable.get_local_patch method for these. That is the base tiles and 8
  adjacent copies (for a rectangular tiling), or 6 adjacent copies (for a
  hexagonal tiling)."""
  points: dict[int, Vertex]
  """dictionary of all points (vertices and corners) in the tiling, keyed by
  Vertex ID."""
  edges: dict[tuple[int, int], Edge]
  """dictionary of the tiling edges, keyed by Edge ID."""
  dual_tiles: dict[int, geom.Polygon]
  """list of geom.Polygons from which a dual tiling might be constructed."""
  n_tiles: int
  """number of tiles in the base Tileable (retained for convenience)."""
  shape_groups: list[list[int]]
  """list of lists of tile IDs distinguished by shape and optionally tile_id"""
  tile_matching_transforms: list[tuple[float]]
  """shapely transform tuples that map tiles onto other tiles"""
  tile_transitivity_classes: list[tuple[int]]
  """list of lists of tile IDs in each transitivity class"""
  vertex_transitivity_classes: list[list[int]]
  """list of lists of vertex IDs in each transitivity class"""
  edge_transitivity_classes: list[list[tuple[int]]]
  """list of lists of edge IDs in each transitivity class"""
  check_points: dict
  """list of points to be used for checking isometries"""
  check_points_lookup: defaultdict(set)
  """lookup from check_points to corresponding element base_IDs."""
  reducer: Callable
  """function to reduce coordinates to fractional part of their components"""
  orbits: defaultdict[set[str]] = None
  """assignment of element ids to sets under symmetries."""

  def __init__(
      self,
      unit: Tileable | None,
      ignore_tile_ids: bool = True,
    ) -> None:
    """Class constructor.

    Args:
      unit (Tileable): the Tileable whose topology is required.
      ignore_tile_ids (bool): (EXPERIMENTAL) if True then only consider the tile
        shapes, not labels. If False consider any labels. Defaults to True.

    """
    # Note that the order of these setup steps is critical sometimes not
    # obviously so. NULL initialisation with unit = None accommodates cloning a
    # new Topology, and (potentially) simplifies notebook-based debugging.
    if unit is not None:
      self.tileable = unit # keep this for reference
      self.n_tiles = self.tileable.tiles.shape[0]
      self._initialise_points_into_tiles()
      self._setup_vertex_tile_relations()
      self._setup_edges()
      self._copy_base_tiles_to_patch()
      self._assign_vertex_and_edge_base_IDs()
      self._setup_reducer()
      self._setup_symmetry_check_points()
      self._check_rotations()
      self._check_reflections()
      self._label_elements()
      # self._identify_distinct_tile_shapes(ignore_tile_ids)
      # self._find_tile_transitivity_classes(ignore_tile_ids)
      # self._find_vertex_transitivity_classes(ignore_tile_ids)
      # self._find_edge_transitivity_classes(ignore_tile_ids)
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


  def _initialise_points_into_tiles(self, debug: bool = False) -> None:
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
        for p in self.points.values():
          if c.distance(p.point) <= 2 * tiling_utils.RESOLUTION:
            # an already existing vertex, so add to tile and break
            tile.corners.append(p.ID)
            # set flag so we know that we're done with this one
            prev_vertex = p
            break
        if prev_vertex is None:
          # new vertex, add it to topology dictionary and to tile
          v = self.add_vertex(c)
          tile.corners.append(v.ID)
          if debug:
            print(f"Added new Vertex {v} to Tile {i}")


  def _setup_vertex_tile_relations(self, debug: bool = False) -> None:
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
      # performance much improved using vertex IDs to match, not Vertex objects
      # we need current shape (not yet set) to check for incident vertices
      shape = geom.Polygon([c.point for c in tile.get_corners()])
      # get points incident on tile boundary, not already in tile corners
      new_points = [v for v in self.points.values()
                    if v.ID not in tile.corners and
                    v.point.distance(shape) <= 2 * tiling_utils.RESOLUTION]
      # iterate over sides of tile to see which side vertex is incident on
      for c1, c2 in tile.get_corner_pairs():
        to_insert = []
        if len(new_points) > 0:
          if debug:
            print(f"{[v.ID for v in new_points]} incident on tile")
          ls = geom.LineString([self.points[c1].point, self.points[c2].point])
          to_insert = [v for v in new_points
                       if v.point.distance(ls) <= 2 * tiling_utils.RESOLUTION]
          if len(to_insert) > 0:
            # sort by distance along side
            d_along = sorted([(ls.line_locate_point(v.point), v)
                              for v in to_insert], key = lambda x: x[0])
            to_insert = [v.ID for d, v in d_along]
        all_points = [c1, *to_insert, c2]
        corners.extend(all_points[:-1])
        for (x1, x2) in itertools.pairwise(all_points):
          # x2 will add tile and neigbour when we get to next side, so no need
          # to do it here: every vertex gets its turn!
          self.points[x1].add_tile(tile.ID)
          self.points[x1].add_neighbour(x2)
      tile.corners = corners
      tile.set_shape_from_corners()


  def _setup_edges(self, debug: bool = False) -> None:
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
      # finding ints in lists is much faster than finding Vertex objects
      # hence we use lists of IDs not Vertex objects
      if len(vertices) > 1:
        for v1, v2 in zip(vertices, vertices[1:] + vertices[:1], strict = True):
          corner_IDs = tile.corners
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
              # if it is, then set left_tile and add to tile edges
              if debug:
                print(f"reverse edge {r_ID} found")
              e = self.edges[r_ID]
              e.left_tile = tile.ID
              tile.edges.append(e.ID)
            else:
              # we've found a new edge so make and add it
              if debug:
                print(f"adding new edge {corners}")
              e = self.add_edge(corners)
              e.right_tile = tile.ID
              tile.edges.append(e.ID)
      # initialise the edge direction information in the tile
      tile.set_edge_directions()


  def _assign_vertex_and_edge_base_IDs(self) -> None:
    """Assign the base_ID attributes of vertices and edges.

    These allow us to determine correspondences between vertices and edges in
    the 'base' tiles in the Topology tileable, and those we have added at
    radius 1 for labelling and visualisation.
    """
    self._assign_vertex_base_IDs()
    self._assign_edge_base_IDs()


  def _assign_vertex_base_IDs(self) -> None:
    """Assign base_ID attribute of vertices."""
    # assign vertex base_ID from core tiles
    for tile0 in self.tiles[:self.n_tiles]:
      for v in tile0.get_corners():
        v.base_ID = v.ID
    # assign others from their corresponding vertex in the core
    for tile0 in self.tiles[:self.n_tiles]:
      for tile1 in self.tiles[self.n_tiles:]:
        if tile1.base_ID == tile0.base_ID:
          for v0, v1 in zip(tile0.get_corners(),
                            tile1.get_corners(), strict = True):
            v1.base_ID = v0.base_ID


  def _assign_edge_base_IDs(self) -> None:
    """Assign base_ID attribute of edges, based on base_IDs of edges."""
    for tile0 in self.tiles[:self.n_tiles]:
      for e in tile0.get_edges():
        e.base_ID = e.ID
    for tile0 in self.tiles[:self.n_tiles]:
      for tile1 in self.tiles[self.n_tiles:]:
        if tile1.base_ID == tile0.base_ID:
          for e0, e1 in zip(tile0.get_edges(), 
                            tile1.get_edges(), strict = True):
            e1.base_ID = e0.base_ID


  def _copy_base_tiles_to_patch(self) -> None:
    """Copy attributes of base tiles to corresponding tiles in radius-1 patch.

    This requires:

    1. Inserting any additional corners in the base tiles not found in the
       radius-1 tiles.

    2. Any vertices in the base tiles that are NOT tiling vertices are added
       to radius-1 tiles leading to merging of some edges.
    """
    # the number of tiles in the base + radius-1
    n_r1 = len(self.tiles)
    # first add any missing vertices to the non-base tiles
    # add all missing vertices before doing any merges
    for tile0 in self.tiles[:self.n_tiles]:
      for tile1 in self.tiles[tile0.ID:n_r1:self.n_tiles]:
        self._match_reference_tile_vertices(tile0, tile1)
    # then merge any edges that meet at a corner
    for tile0 in self.tiles[:self.n_tiles]:
      for tile1 in self.tiles[tile0.ID:n_r1:self.n_tiles]:
        self._match_reference_tile_corners(tile0, tile1)


  def _match_reference_tile_vertices(self, tile1: Tile, tile2: Tile) -> None:
    """Add vertices to tile2 so it matches tile1 adjusting edges as required.

    This assumes the tiles are the same shape, but that tile2 may be missing
    some tiling vertices along some edges.

    Args:
      tile1 (Tile): reference tile.
      tile2 (Tile): tile to change.

    """
    while len(tile1.corners) > len(tile2.corners):
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
          for new_edge in new_edges:
            e = self.add_edge(new_edge)
            self.edges[e.ID] = e


  def _match_reference_tile_corners(self, tile1: Tile, tile2: Tile) -> None:
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
        old_edges, new_edge = self.tiles[v.tiles[0]].merge_edges_at_vertex(v.ID)
        for e in old_edges:
          del self.edges[e]
        self.edges[new_edge.ID] = new_edge


  def _setup_reducer(self) -> None:
    """Return function to get coordinates in the unit basis vector space."""
    basis = np.transpose(np.array(
      self.tileable.get_vectors()[:2], dtype = float))
    inverse = np.linalg.inv(basis)
    def reducer(pt) -> tuple[float, float]:
      result = inverse @ np.array([pt[0], pt[1]], dtype = float)
      fraction = result - np.floor(result)
      fraction = np.where(fraction > 1 - 1e-9, 0, fraction)
      return (float(round(fraction[0], tiling_utils.PRECISION)),
              float(round(fraction[1], tiling_utils.PRECISION)))
    self.reducer = reducer


  def _setup_symmetry_check_points(self) -> None:
    """Set the topology's sentinel points and their mapped points."""

    class CheckPoint(NamedTuple):
      xy: tuple[float, float]
      id: str
      type: ElementType

    self.check_points = {}
    self.check_points_lookup = {}
    # vertices
    for i, v in enumerate(self.vertices_in_tiles(self.tiles[:self.n_tiles])):
      pt = (v.point.x, v.point.y)
      self.check_points[self.reducer(pt)] = CheckPoint(
        pt, f"v{i}", ElementType.VERTEX)
      self.check_points_lookup[f"v{i}"] = v.base_ID
    for i, e in enumerate(self.edges_in_tiles(self.tiles[:self.n_tiles])):
      v0 = self.points[e.vertices[0]].point
      v1 = self.points[e.vertices[1]].point
      pt = ((v0.x + v1.x) / 2, (v0.y + v1.y) / 2)
      self.check_points[self.reducer(pt)] = CheckPoint(
        pt, f"e{i}", ElementType.EDGE)
      self.check_points_lookup[f"e{i}"] = e.base_ID
    for i, t in enumerate(self.tiles[:self.n_tiles]):
      pt = (t.centre.x, t.centre.y)
      self.check_points[self.reducer(pt)] = CheckPoint(
        pt, f"t{i}", ElementType.TILE)
      self.check_points_lookup[f"t{i}"] = t.base_ID


  def _check_rotations(self) -> None:
    if self.orbits is None:
      self.orbits = defaultdict(set)
    order = int(Symmetries(
      self.tileable.prototile.geometry[0]).get_symmetry_group_code()[1])
    while order > 1:
      if self._check_order_x_rotation(order):
        break
      if order in (6, 4):
        order = 3
      elif order == 3:
        order = 2
      else:
        break


  def _check_order_x_rotation(self, order:int) -> bool:
    angle = 2 * np.pi / order
    cos, sin = np.cos(angle), np.sin(angle)
    m = ((cos, -sin), (sin, cos))
    for centre in self.check_points.values():
      # Edges can only be centre of 180º rotations
      if centre.type == ElementType.EDGE and order != 2:
        continue
      found = True
      targets = defaultdict(set)
      for pt in self.check_points.values():
        px, py = pt.xy[0] - centre.xy[0], pt.xy[1] - centre.xy[1]
        pt_dash = self.reducer(
          (m[0][0] * px + m[0][1] * py + centre.xy[0],
            m[1][0] * px + m[1][1] * py + centre.xy[1]))
        if ((pt_dash in self.check_points) and
            (pt.type == self.check_points[pt_dash].type)):
          targets[pt_dash].add((pt.id, self.check_points[pt_dash].id))
        else:
          found = False
          break
      if found:
        for k, v in targets.items():
          self.orbits[k] = self.orbits[k].union(*v)


  def _check_reflections(self) -> None:
    if self.orbits is None:
      self.orbits = defaultdict(set)
    # reflections can only be tiling symmetries if the mirror is parallel to the
    # the translation vectors, or along or perpendicular to their angle bisector
    vs = self.tileable.get_vectors()[:2]
    vs = [
      *vs, (vs[0][0] + vs[1][0], vs[0][1] + vs[1][1]),
      (vs[0][0] - vs[1][0], vs[0][1] - vs[1][1])]
    directions = [np.atan2(*(v[::-1])) for v in vs]

    for direction in directions:
      cos, sin = np.cos(2 * direction), np.sin(2 * direction)
      m = ((cos, sin), (sin, -cos))
      for centre in self.check_points.values():
        found = True
        targets = defaultdict(set)
        for pt in self.check_points.values():
          px, py = pt.xy[0] - centre.xy[0], pt.xy[1] - centre.xy[1]
          pt_dash = self.reducer(
            (m[0][0] * px + m[0][1] * py + centre.xy[0],
             m[1][0] * px + m[1][1] * py + centre.xy[1]))
          if ((pt_dash in self.check_points) and
              (pt.type == self.check_points[pt_dash].type)):
            targets[pt_dash].add((pt.id, self.check_points[pt_dash].id))
          else:
            found = False
            break
        if found:
          for k, v in targets.items():
            self.orbits[k] = self.orbits[k].union(*v)


  def _label_elements(self) -> None:
    self._setup_transitivity_classes()
    self._label_vertices()
    self._label_edges()
    self._label_tiles()


  def _setup_transitivity_classes(self) -> None:
    self.vertex_transitivity_classes = []
    self.edge_transitivity_classes = []
    self.tile_transitivity_classes = []
    if len(self.orbits) == 0:
      self.orbits = {xyt: {pt.id} for xyt, pt in self.check_points.items()}
    element_sets = nx.connected_components(
      nx.from_edgelist(
        itertools.chain.from_iterable(
          [itertools.combinations_with_replacement(x, 2)
          for x in self.orbits.values()])))
    for elements in element_sets:
      element_type = next(iter(elements))[0]
      if element_type == "v":
        self.vertex_transitivity_classes.append(list(elements))
      elif element_type == "e":
        self.edge_transitivity_classes.append(list(elements))
      else:
        self.tile_transitivity_classes.append(list(elements))


  def _label_vertices(self) -> None:
    done = set()
    for i, orbit in enumerate(self.vertex_transitivity_classes):
      for element_id in orbit:
        ID = self.check_points_lookup[element_id]
        if ID in done:
          continue
        for v in [v for v in self.points.values() if v.base_ID == ID]:
          v.label = LABELS[i]
        done.add(id)


  def _label_edges(self) -> None:
    done = set()
    for i, orbit in enumerate(self.edge_transitivity_classes):
      for element_id in orbit:
        ID = self.check_points_lookup[element_id]
        if ID in done:
          continue
        for e in [e for e in self.edges.values() if e.base_ID == ID]:
          e.label = labels[i]


  def _label_tiles(self) -> None:
    done = set()
    for i, orbit in enumerate(self.tile_transitivity_classes):
      for element_id in orbit:
        ID = self.check_points_lookup[element_id]
        if ID in done:
          continue
        for t in [t for t in self.tiles if t.base_ID == ID]:
          t.transitivity_class = i


  def vertices_in_tiles(self, tiles: list[Tile]) -> list[Vertex]:
    """Get vertices incident on tiles in supplied list.

    Args:
      tiles (list[Tile]): tiles whose vertices are required.

    Returns:
      list[Vertex]: the required vertices.

    """
    vs = set()
    for tile in tiles:
      vs = vs.union(tile.corners)
    return [self.points[v] for v in vs]


  def edges_in_tiles(self, tiles: list[Tile]) -> list[Edge]:
    """Get edges that are part of the boundary of tiles in supplied list.

    Args:
      tiles (list[Tile]): tiles whose edges are required.

    Returns:
      list[Edge]: the required edges.

    """
    es = set()
    for tile in tiles:
      es = es.union(tile.edges)
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


  def add_vertex(self, pt: geom.Point) -> Vertex:
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


  def add_edge(self, vs: list[int]) -> Edge:
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


  def transform_geometry(
      self,
      new_topology: bool,
      apply_to_tiles: bool,
      selector: str,
      transform_type: str,
      **kwargs: dict[str:float]) -> Topology:
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


  def get_kwargs(self, fn: Callable, **kwargs: dict[str:str | float]) -> dict:
    """Filter the supplied kwargs to only contain arguments required by fn.

    Args:
      fn (Callable): the function that is to be inspected.
      **kwargs (str | float): kwargs to be filtered.

    Returns:
      str | float: filtered dictionary of kwargs.

    """
    args = inspect.signature(fn).parameters
    return {k: kwargs.pop(k) for k in dict(kwargs) if k in args}


  def zigzag_edge(
      self,
      edge: Edge,
      start: str = "A",
      n: int = 2,
      h: float = 0.5,
      smoothness: int = 0,
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
                   if k not in edge.corners[1:-1]}
    # add the new ones
    new_corners = [self.add_vertex(geom.Point(xy)).ID for xy in ls.coords[1:-1]]
    edge.corners = [edge.vertices[0], *new_corners, edge.vertices[-1]]
    if edge.right_tile is not None:
      self.tiles[edge.right_tile].set_corners_from_edges(False)
    if edge.left_tile is not None:
      self.tiles[edge.left_tile].set_corners_from_edges(False)


  def zigzag_between_points(
      self,
      p0: geom.Point,
      p1: geom.Point,
      n: int,
      h: float = 1.0,
      smoothness: int = 0,
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
      edge: Edge,
      centre: str = "",
      angle: float = 0,
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


  def scale_edge(self, edge: Edge, sf: float = 1.0) -> None:
    """Scale edge.

    Args:
      edge (Edge): the edge to scale.
      sf (float): amount by which to scale the edge.

    """
    v0, v1 = edge.get_vertices()
    ls = geom.LineString([v0.point, v1.point])
    ls = affine.scale(ls, xfact = sf, yfact = sf, origin = ls.centroid)
    v0.point, v1.point = [geom.Point(c) for c in ls.coords]


  def push_vertex(self, vertex: Vertex, push_d:float) -> tuple[float]:
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


  def nudge_vertex(self, vertex: Vertex, dx: float, dy: float) -> None:
    """Nudge vertex by specified displacement.

    Args:
      vertex (Vertex): the vertext to nudge.
      dx (float): x displacement.
      dy (float): y displacement.

    """
    vertex.point = affine.translate(vertex.point, dx, dy)


  def _get_tile_geoms(self) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of the tiles.

    Returns:
      gpd.GeoDataFrame: tiles as a GeoDataFrame, with their transitivity class
        and label as attributes.

    """
    return gpd.GeoDataFrame(
      data = {
        "transitivity_class": [t.transitivity_class for t in self.tiles],
        "label": [t.label for t in self.tiles]},
      geometry = gpd.GeoSeries([t.shape for t in self.tiles]))


  def _get_tile_centre_geoms(self) -> gpd.GeoDataFrame:
    """Return the centre of tiles as a GeoDataFrame.

    Returns:
      gpd.GeoDataFrame: the tile centres with tile label as an attribute.

    """
    return gpd.GeoDataFrame(
      data = {"label": [t.label for t in self.tiles]},
      geometry = gpd.GeoSeries([t.centre for t in self.tiles]))


  def _get_point_geoms(self) -> gpd.GeoDataFrame:
    """Return tiling vertices and tile corners as a GeoDataFrame of Points.

    Returns:
      gpd.GeoDataFrame: corners and vertices of the tiling, including
        transitivity_class, label, and is_tiling_vertex as attributes.

    """
    return gpd.GeoDataFrame(
      data = {"transitivity_class": [v.transitivity_class
                                    for v in self.points.values()],
              "label": [v.label for v in self.points.values()
                        if v.is_tiling_vertex],
              "is_tiling_vertex": [v.is_tiling_vertex
                                  for v in self.points.values()]},
      geometry = gpd.GeoSeries([v.point for v in self.points.values()]))


  def _get_vertex_geoms(self) -> gpd.GeoDataFrame:
    """Return tiling vertices (not corners) as a GeoDataFrame of Points.

    Returns:
      gpd.GeoDataFrame: vertices of the tiling, including transitivity_class and
        label as attributes.

    """
    return gpd.GeoDataFrame(
      data = {"transitivity_class": [v.transitivity_class
                                    for v in self.points.values()
                                    if v.is_tiling_vertex],
              "label": [v.label for v in self.points.values()
                        if v.is_tiling_vertex]},
      geometry = gpd.GeoSeries([v.point for v in self.points.values()
                                if v.is_tiling_vertex]))


  def _get_edge_geoms(self, offset: float = 0.0) -> gpd.GeoDataFrame:
    """Return tiling edges as GeoDataFrame of LineStrings optionally offset.

    Offsetting the edges allows their direction to be seen when visualised.

    Returns:
      gpd.GeoDataFrame: edges of the tiling including transitivity class and label
        as attributes.

    """
    return gpd.GeoDataFrame(
      data = {"transitivity_class": [e.transitivity_class
                                    for e in self.edges.values()],
              "label": [e.label for e in self.edges.values()]                       },
      geometry = gpd.GeoSeries([e.get_topological_edge().parallel_offset(offset)
                                for e in self.edges.values()]))
