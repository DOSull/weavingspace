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

import numpy as np
import shapely.geometry as geom

from weavingspace import tiling_utils

if TYPE_CHECKING:
  from weavingspace import Topology

"""The topological elements of a tiling are vertices, edges, and tiles. The
`Vertex`, `Edge`, and `Tile` classes in this file implement versions of these to
be used by the `weavingspace.Topology` class in labelling the transitivity
classes of edges, tiles, and vertices in a tiling to enable topologically
consistent transformations thereof. 

These classes do not precisely represent distinctions in the mathematical
literature between tiling vertices and tile corners, or between tiling edges and
tile sides, although they're close enough for our purposes!
"""


class Tile:
  """Class to represent essential features of polygons in a tiling."""

  topology: Topology
  """the containing Topology object."""
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

  def __init__(self, topology: Topology, ID: int) -> None:
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
    return (f"Tile {self.ID} Corners: {self.corners} "
            f"Edges: {self.edges}")


  def __repr__(self) -> str:
    return str(self)


  def get_corners(self) -> list[Vertex]:
    """Return corners of Tile as Vertex objects.

    Returns:
      list[Vertex]: list of corners of this tile retrieved via self.topology.

    """
    return [self.topology.points[v] for v in self.corners]


  def get_corner_pairs(self) -> list[tuple[int,int]]:
    """Return sequence of consecutive pairs of corner IDs.

    Returns:
      list[tuple[int,int]]: list of pairs of edge IDs that constitute edges of
        this Tile.

    """
    return zip(self.corners, [*self.corners[1:], *self.corners[:1]],
               strict = True)


  def get_edges(self) -> list[Edge]:
    """Return edges as list of Edge objects.

    Returns:
      list[Edge]: list of edges of this tile retrieved via self.topology.

    """
    return [self.topology.edges[ij] for ij in self.edges]


  def set_shape_from_corners(self) -> None:
    """Set the shape attribute based on corners, and associated tile centre."""
    self.shape = geom.Polygon([c.point for c in self.get_corners()])
    # self.centre = tiling_utils.get_clean_polygon(self.shape).centroid
    self.centre = tiling_utils.get_incentre(
      tiling_utils.get_clean_polygon(self.shape))


  def set_corners_from_edges(self, update_shape: bool = True) -> None:
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
        self.corners.extend(e.corners[:-1])
      else: # counter-clockwise so extend in reverse
        self.corners.extend(e.corners[1:][::-1])
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
      v: Vertex,
      i: int,
      update_shape: bool = False,
    ) -> tuple[tuple[int, int], tuple[tuple[int, int],...]]:
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
        tuple IDs of new edges arising from insertion of this Vertex.

    """
    self.corners = [*self.corners[:i], v.ID, *self.corners[i:]]
    old_edge = self.get_edges()[i - 1]
    # store current ID of the affected edge for return to calling context
    old_edge_ID = old_edge.ID
    new_edges = [e.ID for e in old_edge.insert_vertex(v.ID, self.corners[i - 1])]
    self.edges = [*self.edges[:(i-1)], *new_edges, *self.edges[i:]]
    self.set_edge_directions()
    if update_shape:
      self.set_shape_from_corners()
    return old_edge_ID, new_edges


  def merge_edges_at_vertex(self, v: int) -> tuple:
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
    to_remove, new_edge = self.get_updated_edges_from_merge(v)
    if len(self.topology.points[v].tiles) > 1:
      self.topology.tiles[self.topology.points[v].tiles[1]] \
        .get_updated_edges_from_merge(v, new_edge)
    return to_remove, new_edge


  def get_updated_edges_from_merge(
      self,
      v: Vertex,
      new_edge: Edge = None,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], Edge] | None:
    """Update edges and edges_CW attributes based on insertion of Vertex.

    If new_edge is supplied then the neighbour tile at v has already created
    the needed new Edge and this Edge is the one that will be 'slotted in' at
    the appropriate spot in the edges list.

    The edges_CW is also updated to maintain correct directions of the edges.
    The corners attribute is unaffected by these changes.

    Args:
      v (int): ID of Vertex at which to carry out the merge.
      new_edge (Edge, optional): if another Tile has already carried out this
        merge this should be the resulting new Edge for insertion into this
        Tile. Defaults to None (when the new Edge will be constructed).

    Returns:
      tuple | None: either None (if a new edge was supplied) or a tuple
        of the two edge IDs to be removed and the new edge added for return to
        the calling context (i.e. the containing Topology).

    """
    # get the two edge list index positions in which vertex v is found
    i, j = self.get_edge_IDs_including_vertex(v)
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
      v: int,
    ) -> tuple[int]:
    """Get two index positions of edges that include supplied Vertex v.

    Args:
        v (int): ID of Vertex of interest.

    Returns:
      tuple[int]: index positions of edges in edges list that contain v.

    """
    return (i for i, e in enumerate(self.edges) if v in e)


  def get_merged_edge(self, i: int, j: int) -> Edge:
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
    ei, ej = (self.topology.edges[self.edges[i]],
              self.topology.edges[self.edges[j]])
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
    v_sequence = [*(head if CWi else head[::-1]), *(tail if CWj else tail[::-1])]
    return Edge(self.topology, v_sequence)


  def offset_corners(self, offset: int) -> None:
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


  def angle_at(self, v: int) -> float:
    """Return interior angle at the specified corner (in degrees).

    Args:
        v (int): ID of corner where angle is requested.

    Returns:
        float: angle at corner in degrees.

    """
    i = self.corners.index(v)
    n = len(self.corners)
    return tiling_utils.get_inner_angle(
      self.topology.points[self.corners[i-1]].point,
      self.topology.points[self.corners[i]].point,
      self.topology.points[self.corners[(i + 1) % n]].point)


class Vertex:
  """Class to store attributes of a vertex in a tiling."""

  topology: Topology
  """the containing Topology object."""
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


  def __init__(self, topology: Topology, point: geom.Point, ID: int) -> None:
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
    return f"Vertex {self.ID} at {self.point} Tiles: {self.tiles}"


  def __repr__(self) -> str:
    return str(self)


  def get_tiles(self) -> list[Tile]:
    """Return list of tiles as Tile objects.

    Returns:
      list[Tile]: list of Tiles incident at this Vertex retrieved via
        topology.

    """
    return [self.topology.tiles[t] for t in self.tiles]


  def add_tile(self, tile: int) -> None:
    """Add supplied Tile to the tiles list if it is not already present.

    Args:
        tile (Tile): Tile to add.

    """
    if tile not in self.tiles:
      self.tiles.append(tile)


  def add_neighbour(self, vertex_id: int) -> None:
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
    return abs(360 - sum([t.angle_at(self.ID) for t in self.get_tiles()])) \
                     < tiling_utils.RESOLUTION


  def _order_of_pts_cw_around_centre(
      self,
      pts: list[geom.Point],
      centre: geom.Point,
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
  """the containing Topology object."""
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

  def __init__(self, topology:Topology, corners:list[int]) -> None:
    """Class constructor.

    Initialises the corners and vertices lists and sets ID to tuple(vertices).
    The vertices list is all the corners with is_tiling_vertex = True. Note that
    during initialisation this property default True until after relations
    between tiles and vertices have been determined.

    Args:
      corners (list[int]): list of all corners along the edge.

    """
    self.topology = topology
    self.corners = corners
    self.vertices = [
      v for v in corners if self.topology.points[v].is_tiling_vertex]
    self.ID = tuple(self.vertices)


  def __str__(self) -> str:
    """Return a string representation of the Edge.

    Returns:
      str: include ID and a list of corner vertex IDs.

    """
    return f"Edge {self.ID} Corners: {self.corners}"


  def __repr__(self) -> str:
    return str(self)


  def get_corners(self) -> list[Vertex]:
    """Return corners as list of Vertex objects.

    Returns:
      list[Vertex]: list of Vertex objects retrieved via topology.

    """
    return [self.topology.points[v] for v in self.corners]


  def get_vertices(self) -> list[Vertex]:
    """Return vertices as list of Vertex objects.

    Returns:
      list[Vertex]: list of Vertex objects retrieved via topology.

    """
    return [self.topology.points[v] for v in self.vertices]


  def insert_vertex(self, v: int, predecessor: int) -> list[Edge]:
    """Insert vertex after predecessor and return modified edge and new edge.

    If the initial edge was (say) (0 3 2 5) and the predecessor was set to 1
    the returned edges would be (0 3 v) and (v 2 5).

    Args:
      v (int): ID of the Vertex to insert.
      predecessor (int): ID of Vertex after which to insert v.

    Returns:
      list[Edge]: two Edges, this one, and a new one arising from the
        insertion.

    """
    i = self.corners.index(predecessor)
    new_edge = Edge(self.topology, [v, *self.corners[i + 1:]])
    if self.right_tile is not None:
      new_edge.right_tile = self.right_tile
    if self.left_tile is not None:
      new_edge.left_tile = self.left_tile
    self.corners = [*self.corners[:i + 1], v]
    self.vertices = [self.vertices[0], v]
    self.ID = tuple(v for v in self.vertices)
    return [self, new_edge]


  def get_geometry(self, forward: bool = True) -> geom.LineString:
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


  def get_topological_edge(self, forward: bool = True) -> geom.LineString:
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
