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

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import shapely
import shapely.affinity as affine
import shapely.geometry as geom
from matplotlib import pyplot as plt

from weavingspace import tiling_utils

if TYPE_CHECKING:
  from collections.abc import Sequence

"""Classes for working with symmetries."""

@dataclass(slots=True)
class KMP_Matcher:  # noqa: N801
  """Class to find matching subsequences in a sequence."""

  sequence:Sequence
  """Iterable in which subsequences are to be found."""

  def find_matches(
      self,
      pattern:Sequence[tuple[float,...]]) -> list[int]:
    """Find matches in supplied pattern using Knuth-Morris-Pratt algorithm.

    See:
    https://en.wikipedia.org/wiki/Knuth-Morris-Pratt_algorithm which provides
    detailed pseudo-code on which this code is directly based. See also:

    Knuth DE, JH Morris Jr, and VR Pratt. 1977. Fast pattern  matching in
    strings. SIAM Journal on Computing 6(2): 323-350. doi: 10.1137/0206024.

    This implementation expects sequences of tuples of floats, although in
    principle any objects could be contained in the Sequence.

    Args:
      pattern (Sequence[tuple[float,...]]): The sequence to match.

    Returns:
      Sequence[int]: the index positions at which matches are found.

    """
    j, k, = 0, 0
    finds = []
    table = self._get_table(pattern)
    while j < len(self.sequence):
      if self._equal_tuples(pattern[k], self.sequence[j]):
        j = j + 1
        k = k + 1
        if k == len(pattern):
          finds.append(j - k)
          k = table[k]
      else:
        k = table[k]
        if k < 0:
          j = j + 1
          k = k + 1
    return finds

  def _get_table(
      self,
      pattern:Sequence) -> Sequence[int]:
    """Return 'offsets' table as required by self.find_matches.

    Note that the terse variable names are those used in the KMP 1977 paper.

    Args:
      pattern (Iterable): The pattern to set up the table for.
      equals (function, optional): A function to use to test for equality of
        elements in patterns. Defaults to _equal_tuples().

    Returns:
      Iterable[int]: _description_

    """
    pos = 1
    cnd = 0
    T = {0: -1}
    while pos < len(pattern):
      if np.allclose(pattern[pos], pattern[cnd]):
        T[pos] = T[cnd]
      else:
        T[pos] = cnd
        while cnd >= 0 and not self._equal_tuples(pattern[pos], pattern[cnd]):
          cnd = T[cnd]
      pos = pos + 1
      cnd = cnd + 1
    T[pos] = cnd
    return tuple(T.values())

  def _equal_tuples(
      self,
      t1:Sequence[float],
      t2:Sequence[float]) -> bool:
    """Test for near equality of two iterables of floats.

    We use numpy.allclose to avoid issues with floating point exact equality
    tests. Wrapped to allow find_matches function to take alternative equality
    tests as input.

    Args:
      t1 (Sequence[float]): First set of floats.
      t2 (Sequence[float]): Second set of floats.

    Returns:
      bool: True if t1 == t2 for all corresponding items in each sequence.

    """
    return np.allclose(t1, t2, atol = tiling_utils.RESOLUTION * 10)

  def _round_tuple(
      self,
      t:Sequence[float],
      digits:int = 3) -> Sequence[float]:
    """Round all floats in a sequence.

    Args:
      t (Sequence[float]): the floats to round.
      digits (int): number of digits to round to.

    Returns:
      Sequence[float]: the rounded Sequence.

    """
    return tuple([np.round(x, digits) for x in t])


@dataclass(slots=True, repr=False)
class Transform:
  """Class to store details of a transform and draw it."""

  transform_type:str
  """Type of transform 'rotation', 'reflection', 'translation' or 'identity'."""
  angle:float
  """Angle of rotation (degrees)."""
  centre:geom.Point
  """Centre of the transformation."""
  translation:tuple[float,...]
  """X and Y coordinates shifts of a translation transform. A glide reflection
  may also include this."""
  transform:tuple[float,...]
  """Six element tuple for the transform in shapely.transform format. See
  https://shapely.readthedocs.io/en/stable/manual.html#affine-transformations
  and methods in `weavingspace.tiling_utils`."""

  def __str__(self) -> str:
    if self.transform_type == "rotation":
      return (
        f"{self.transform_type:<11}{self.angle:>6.1f}° "
        f"POINT ({self.centre.x:6.1f} {self.centre.y:6.1f}) "
        f"{tuple([float(np.round(x, 3)) for x in self.transform])}"
      )
    if self.transform_type == "translation":
      return (
        f"{self.transform_type:<11}{'':>14}"
        f"({self.translation[0]:>6.1f},{self.translation[0]:>6.1f}) "
        f"{tuple([float(np.round(x, 3)) for x in self.transform])}"
      )
    if self.transform_type == "reflection":
      return (
        f"{self.transform_type:<11}{self.angle:>6.1f}° "
        f"POINT ({self.centre.x:6.1f} {self.centre.y:6.1f}) "
        f"{tuple([float(np.round(x, 3)) for x in self.transform])}"
      )
    return (
      f"{self.transform_type:<11}{'':>30}"
      f"{tuple([float(np.round(x, 3)) for x in self.transform])}"
    )


  def __repr__(self) -> str:
    return str(self)


  def apply(
      self,
      geometry:shapely.Geometry) -> shapely.Geometry:
    """Apply this transform to supplied shapely geometry and return result.

    Args:
      geometry (shapely.Geometry): a shapely geometry to transform.

    Returns:
      shapely.Geometry: the resulting transformed shapely geometry.

    """
    return affine.affine_transform(geometry, self.transform)


  def draw(
      self,
      ax:plt.Axes,
      **kwargs:str,
    ) -> plt.Axes:
    """Draw this transform on the supplied axes.

    Arguments specific to each transform type are supplied as **kwargs and are
    documented in `draw_rotation`, `draw_reflection`, and `draw_translation`.

    Args:
      ax (pyplot.axes): the axes on which to draw the transform.
      kwargs: passed on to specific draw method.

    Returns:
      pyplot.Axes: the axes with the rendering of this transform added.

    """
    match self.transform_type:
      case "rotation":
        args = inspect.signature(self.draw_rotation).parameters
        rotn_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in args}
        return self.draw_rotation(ax = ax, **rotn_args)
      case "reflection":
        args = inspect.signature(self.draw_reflection).parameters
        refn_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in args}
        return self.draw_reflection(ax = ax, **refn_args)
      case "translation":
        args = inspect.signature(self.draw_translation).parameters
        trans_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in args}
        return self.draw_translation(ax = ax, **trans_args)
      case "identity":
        w = ax.get_window_extent().width
        return ax.set_title(f"{self.transform_type}", fontsize = w / 20)


  def draw_rotation(
      self,
      ax:plt.Axes,
      radius:float = 200,
      add_title:bool = True,
    ) -> plt.Axes:
    """Draw a rotation transform on the supplied Axes.

    Args:
      ax (pyplot.Axes): axes on which to draw the representation.
      radius (float): radius in object units to draw arcs representing the
        angle. Defaults to 200.
      add_title (bool): whether or not to add a title to the drawing.
        Defaults to True.

    Returns:
      the Axes with the rendering of this transform added.

    """
    x, y = self.centre.x, self.centre.y
    axis = geom.LineString([(x, y), (x + radius * 1.25, y)])
    arc = geom.LineString([
      geom.Point(x + radius * np.cos(np.radians(a)),
                 y + radius * np.sin(np.radians(a)))
      for a in np.linspace(0, self.angle, 50)])
    gpd.GeoSeries(self.centre).plot(
      ax = ax, color = "k", markersize = 4, zorder = 5)
    gpd.GeoSeries([axis, arc]).plot(ax = ax, color = "k", lw = .5)
    if add_title:
      w = ax.get_window_extent().width
      ax.set_title(
        f"{self.transform_type} {round(self.angle, 2)}", fontsize = w / 20)
    return ax


  def draw_reflection(
      self,
      ax:plt.Axes,
      w:float = 5,
      mirror_length:float = 100,
      add_title:bool = True,
    ) -> plt.Axes:
    """Draw a reflection transform on the supplied Axes.

    Args:
      ax (pyplot.Axes): axes on which to draw the representation.
      w (float): linewidth for the mirror line. Defaults to 5.
      mirror_length (float): length of the mirror line in units of the
        objects being reflected. Defaults to 100.
      add_title (bool): whether or not to add a title to the drawing.
        Defaults to True.

    Returns:
      the Axes with the rendering of this transform added.

    """
    x, y = self.centre.x, self.centre.y
    dx, dy = self.translation
    r = np.sqrt(self.translation[0] ** 2 + self.translation[1] ** 2)
    mirror = geom.LineString([
      (x - mirror_length / 2 * np.cos(np.radians(self.angle)),
       y - mirror_length / 2 * np.sin(np.radians(self.angle))),
      (x + mirror_length / 2 * np.cos(np.radians(self.angle)),
       y + mirror_length / 2 * np.sin(np.radians(self.angle)))])
    gpd.GeoSeries([mirror]).plot(
      ax = ax, color = "k", lw = 1, ls = "dashdot", zorder = 5)
    no_slide = np.isclose(r, 0, rtol = 1e-6, atol = 1e-6)
    if not no_slide:
      ax.arrow(
        x - dx / 2, y - dy / 2, dx, dy, length_includes_head = True,
        width = w, fc = "k", ec = None, head_width = w * 6, zorder = 5)
    if add_title:
      w = ax.get_window_extent().width
      ax.set_title(
        f"{self.transform_type} {round(self.angle, 2)}", fontsize = w / 20)
    return ax


  def draw_translation(
      self,
      ax:plt.Axes,
      c:geom.Point,
      w:float = 5,
      add_title:bool = True,
    ) -> plt.Axes:
    """Draw a translation transform on the supplied Axes.

    Args:
      ax (pyplot.Axes): axes on which to draw the representation.
      c (geom.Point): an origin point in the coordinate space of the objects
        from which to start the arrow representing the translation.
      w (float): linewidth for the arrow. Defaults to 5.
      add_title (bool): whether or not to add a title to the drawing.
        Defaults to True.

    Returns:
      the Axes with the rendering of this transform added.

    """
    gpd.GeoSeries([c]).plot(ax = ax, color = "k")
    ax.arrow(c.x, c.y, self.translation[0], self.translation[1],
             length_includes_head = True, lw = 0.5, width = w,
             fc = "k", ec = None, head_width = w * 6, zorder = 5)
    if add_title:
      w = ax.get_window_extent().width
      ax.set_title(
        (f"{self.transform_type} "
         f"({self.translation[0]:.1f}, {self.translation[1]:.1f})"),
        fontsize = w / 20)
    return ax


@dataclass(slots=True, init=False)
class Symmetries:
  """Class to identify and store symmetries of a shapely.Polygon."""

  polygon:geom.Polygon
  """Polygon from which these symmetries are derived."""
  matcher:KMP_Matcher
  """the subsequence matcher used in symmetry detection."""
  n:int
  """number of vertices of the polygon."""
  poly_code:list[tuple[float,...]]
  """the encoding of the polygon as sequence of length, angle pairs."""
  poly_code_r:list[tuple[float,...]]
  """the reversed encoding used to detect reflection symmetries."""
  rotation_shifts:list[int]
  """list of number of 2pi/n rotation symmetries."""
  reflection_shifts:list[int]
  """list of pi/n relection angle symmetries."""
  symmetries:list[Transform]
  """list of Transform objects with more complete information."""
  symmetry_group:str
  """the code denoting the symmetry group"""

  def __init__(self, polygon:geom.Polygon) -> None:
    """Construct a Symmetries instance."""
    self.polygon = polygon
    self.n = shapely.count_coordinates(self.polygon) - 1
    self.poly_code = self._get_polygon_code(self.polygon)
    self.poly_code_r = self._get_polygon_code(self.polygon, mirrored = True)
    self.matcher = KMP_Matcher(self.poly_code + self.poly_code[:-1])
    self.symmetries = self.get_symmetries()
    self.symmetry_group = self.get_symmetry_group_code()


  def _get_polygon_code(
      self,
      polygon:geom.Polygon,
      mirrored:bool = False,
    ) -> list[tuple[float,...]]:
    r"""Return list of length-angle pairs to encode a polygon shape.

    This encoding is unique up to scale. The alignment of lengths and angles is
    important and symmetry detection is sensitively dependent on it...

    The forward (i.e. unmirrored) polygon pairs are of edge lengths and the
    angle between each edge and its successor going CW around the polygon.

           A[i] ----L[i]---- A (i+1)
          /                   \
         /                     \

    i.e. edge i is between angles i and i + 1, and angle i is between edge
    i-1 and i. The mirrored encoding pairs length[i] with angle[i+1].

    The mirrored encoding pairs matched indexes of lengths and angles in the
    original polygon. This means the angle is still the one between and edge
    and its successor but proceeding CCW around the polygon.

    See in particular for clarification:

      Eades P. 1988. Symmetry Finding Algorithms. In Machine Intelligence and
      Pattern Recognition, ed. GT Toussaint, 6:41-51. Computational Morphology.
      North-Holland. doi: 10.1016/B978-0-444-70467-2.50009-6.

    Args:
      polygon (geom.Polygon): the polygon to encode
      mirrored (bool, optional): if true encoding will be in CCW order.
        Defaults to False.

    Returns:
      list[tuple[float]]: a list of side-length angle pairs.

    """
    # get edge lengths and angles
    lengths = tiling_utils.get_side_lengths(polygon)
    raw_angles = tiling_utils.get_interior_angles(polygon)
    if mirrored:
      lengths_r = lengths[::-1]
      angles_r = raw_angles[::-1]
      return list(zip(lengths_r, angles_r, strict = True))
    angles = raw_angles[1:] + raw_angles[:1]
    return list(zip(lengths, angles, strict = True))


  def get_symmetries(self) -> list[Transform]:
    """Find rotation and reflection symmetries of a polygon.

    The work is delegated to `get_rotations()` and `get_reflections()`.

    Based on

      Eades P. 1988. Symmetry Finding Algorithms. In Machine Intelligence and
      Pattern Recognition, ed. GT Toussaint, 6:41-51. Computational Morphology.
      North-Holland. doi: 10.1016/B978-0-444-70467-2.50009-6.

    and also

      Wolter JD, TC Woo, and RA Volz. 1985. Optimal algorithms for symmetry
      detection in two and three dimensions. The Visual Computer 1(1): 37-48.
      doi: 10.1007/BF01901268.

    Details in these papers are not unambiguous. This implementation was
    developed based on them, but with some trial and error to get the index
    offsets and (especially) retrieval of the reflection axes angles to work.

    Returns:
      list[Transform]: a list of Transform objects representing the polygon
        symmetries.

    """
    return self.get_rotations() + self.get_reflections()


  def get_rotations(
      self,
      offsets:list[int]|None = None,
    ) -> list[Transform]:
    """Get the rotations associated with this collection of symmetries.

    Returns:
      list[Transform]: A list of the rotation symmetries associated with this
        Symmetries instance's polygon.

    """
    if offsets is None:
      self.rotation_shifts = self.matcher.find_matches(self.poly_code)
      offsets = self.rotation_shifts
    c = self.polygon.centroid
    rot_angles = [np.round(i * 360 / self.n, 6) for i in offsets]
    rotation_transforms = [tuple(
      np.round(tiling_utils.get_rotation_transform(a, (c.x, c.y)), 6))
      for a in rot_angles]
    return [
      Transform("rotation", angle, c, (0, 0), transform) for
      transform, angle in zip(rotation_transforms, rot_angles, strict = True)]


  def get_reflections(
      self,
      offsets:list[int]|None = None,
    ) -> list[Transform]:
    """Get the reflections associated with this collection of symmetries.

    Args:
      offsets (list[int]): list of integer offsets i where i / n * 360 where n
        is the number of sides of this Symmetries instance's polygons is the
        angle of the mirror in which a reflection symmetry has been identified.

    Returns:
      list[Transform]: A list of the reflection symmetries associated with this
        Symmetries instance's polygon.

    """
    # collate all possible reflection axes - ordering of these is important
    # for correct picking - so don't mess with this code!
    bearings = tiling_utils.get_side_bearings(self.polygon)
    reflection_axes = []
    interior_angles = tiling_utils.get_interior_angles(self.polygon)
    if offsets is None:
      self.reflection_shifts = self.matcher.find_matches(self.poly_code_r)
      offsets = self.reflection_shifts
    for i in range(self.n):
      # the angle bisectors
      reflection_axes.append(bearings[i] - interior_angles[i] / 2)
      # the edge_perpendicular bisectors
      reflection_axes.append(bearings[i] - 90)  # normal to the edge
    # pick out those where matches occurred using the offsets
    ref_angles = [np.round(reflection_axes[i], 6) for i in offsets]
    c = self.polygon.centroid
    reflection_transforms = [tuple(
      np.round(tiling_utils.get_reflection_transform(angle, (c.x, c.y)), 6))
      for angle in ref_angles]
    return [
      Transform("reflection", angle, c, (0, 0), transform) for
      transform, angle in zip(reflection_transforms, ref_angles, strict = True)]

  def get_symmetry_group_code(self) -> str:
    """Return symmetry group code for this instance's polygon.

    See https://en.wikipedia.org/wiki/Symmetry_group#Two_dimensions.

    Returns:
        str: code such as C2 or D4 for cyclic or dihedral symmetry groups.

    """
    if len(self.reflection_shifts) == 0:
      # no reflection symmetries, so it is a cyclic group Cn
      return f"C{len(self.rotation_shifts)}"
    # must be the dihedral group Dn
    return f"D{len(self.rotation_shifts)}"


  def get_corner_offset(
      self,
      poly2:geom.Polygon) -> int|None:
    """Find offset between corners of this polygon and another one.

    The offset is determined under whatever transformation will match one on
    to the other. Used by the `Topology` class. The offset is expressed as the
    number of corners clockwise by which the supplied polygon is offset from
    this one.

    Args:
      poly2 (geom.Polygon): polygon for which the offset is desired.

    Returns:
      int|None: integer number of corners clockwise by which the polygon
        is offset from this one, or None if they don't match.

    """
    s2 = Symmetries(poly2)
    matches = self.matcher.find_matches(s2.poly_code)
    if len(matches) > 0:
      return matches[0]
    matches = self.matcher.find_matches(s2.poly_code_r)
    if len(matches) > 0:
      return -matches[0]
    return None


@dataclass(slots=True, init=False)
class ShapeMatcher:
  """Class to manage finding transforms that map one polygon on to another.

  This was previously handled 'internally' by an earlier version of the
  Symmetries code, but that quickly became ungainly and difficult to maintain.
  This approach makes for cleaner code, and does the additional work of finding
  e.g. centres of rotations (which may lie outside either or both polygons).
  """

  shape:geom.Polygon
  """The target polygon on to which matching transforms are required."""
  s1:Symmetries
  """The Symmetries of the target polygon."""

  def __init__(  # noqa: D107
      self,
      shape: geom.Polygon) -> None:
    self.shape = shape
    self.s1 = Symmetries(shape)


  def get_polygon_matches(
      self,
      shape2:geom.Polygon) -> list[Transform]|None:
    """Return all transforms that match shape2 onto this instance's polygon.

    Args:
      shape2 (geom.Polygon): polygon for which matching transforms are
        requested.

    Returns:
      list[Transform]|None: a list of Transforms that will match shape2
        onto this instance's shape. None is returned if no matching transforms
        are found.

    """
    s2 = Symmetries(shape2)
    if self.s1.symmetry_group != s2.symmetry_group:
      return None
    match_rot, rots = self._get_rotation_matches(s2)
    match_ref, refs = self._get_reflection_matches(s2)
    if match_rot and match_ref:
      return rots + refs
    if match_rot:
      return rots
    if match_ref:
      return refs
    return None


  def _get_rotation_matches(
      self,
      s2:Symmetries) -> tuple[bool,list[Transform]]:
    """Get rotations to match s2's polygon onto this instance's polygon.

    Args:
      s2 (Symmetries): Symmetries associated with the polygon of interest.

    Returns:
      tuple[bool,list[Transform]]: boolean is True if there are any matches
        when matching transforms will be included in the associated list. If
        there are no matches (False, []) will be returned.

    """
    # need this locally to check for equality of all members of a list
    def all_equal(lst:list[float]) -> bool:
      return all(np.isclose(lst[0], x, 1e-3, 1e-3) for x in lst)

    matches = self.s1.matcher.find_matches(s2.poly_code)
    if len(matches) == 0:
      return False, None
    transforms = []
    # get lists of polygon corners aligned correctly to measure the angle
    p1_corners = tiling_utils.get_corners(self.shape, repeat_first = False)
    p2_corners = tiling_utils.get_corners(s2.polygon, repeat_first = False)
    corner_distances = self._get_corner_distances_between_polys(
      p1_corners, p2_corners, matches)
    equalities = [all_equal(d) for d in corner_distances]
    if all(equalities):
      # the two shapes are identical and in the same spot
      # so rotations symmetries are those of the target shape
      transforms.extend(self.s1.get_rotations(matches))
      return True, transforms
    if any(equalities):
      # the one where all are equal is a translation so we'll skip it below
      # and add it here as a translation transform
      m = equalities.index(True)
      del matches[m]
      dx, dy = (p1_corners[0].x - p2_corners[m].x,
                p1_corners[0].y - p2_corners[m].y)
      transforms.append(
        Transform("translation", None, None, (dx, dy), (1, 0, 0, 1, dx, dy)))
    for m in matches:
      a, c = self._get_angle_between_polygons(p1_corners, p2_corners, m)
      if a is None:
        continue
      transforms.append(Transform("rotation", a, c, (0, 0),
        tiling_utils.get_rotation_transform(a, (c.x, c.y))))
    return True, transforms


  def _get_corner_distances_between_polys(
      self,
      corners1:list[geom.Point],
      corners2:list[geom.Point],
      offsets:list[int],
    ) -> list[list[float]]:
    return [
      [p1.distance(p2) for p1, p2 in
       zip(corners1, corners2[i:] + corners2[:i], strict = True)]
      for i in offsets]


  def _get_angle_between_polygons(
      self,
      corners1:list[geom.Point],
      corners2:list[geom.Point],
      offset:int,
    ) -> tuple[None,...]|tuple[str,tuple[float,...]]|tuple[float,geom.Point]:
    """Return angle by which lists of corners are rotationally offset.

    If lists are at the same orientation then the translation required to match
    them (if any) will be returned.

    Args:
      corners1 (list[geom.Point]): list of corners of the first polygon.
      corners2 (list[geom.Point]): list of corners of the second polygon.
      offset (int): number of clockwise steps through the corners by which
        corners2 are offset from corners1 as previously determined.

    Returns:
      tuple[float,geom.Point]|tuple[str,tuple[float]]: one of
      ('identity' (0, 0)), ('translation', (x, y)), or (angle, geom.Point),
        where the last of these is an angle and a centre of rotation.

    """
    corners2 = corners2[offset:] + corners2[:offset]
    dists = [
      p1.distance(p2) for p1, p2 in zip(corners1, corners2, strict = True)]
    if all(np.isclose(dists[0], d, atol = 1e-3, rtol = 1e-3) for d in dists):
      # polygons already aligned similarly so we just need to find translation
      if np.isclose(dists[0], 0, atol = 1e-3, rtol  = 1e-3):
        return "identity", (0, 0)
      if offset == 0:
        return "translation", (corners1[0].x - corners2[0].x,
                               corners1[0].y - corners2[0].y)
    # Polygons not lined up. Order them by distances between them
    ordered_dists = sorted([(i, d) for i, d in enumerate(dists)],
                           key = lambda x: x[1], reverse = True)
    # the two furthest will give the most reliable numbers
    a, b = [i for i, d in ordered_dists[:2]]
    p_1a, p_1b = corners1[a], corners1[b]
    p_2a, p_2b = corners2[a], corners2[b]
    # get the perpendiculars of the lines joining A and B in each polygon
    perp_1a_2a = tiling_utils.get_straight_line(
      p_1a, p_2a, perpendicular = True)
    perp_1b_2b = tiling_utils.get_straight_line(
      p_1b, p_2b, perpendicular = True)
    # and intersect them
    centre = tiling_utils.get_intersection(perp_1a_2a, perp_1b_2b)
    if centre is None:
      return None, None
    angle = -tiling_utils.get_inner_angle(p_1a, centre, p_2a)
    if angle < -179:
      angle = angle + 360
    elif angle > 180:
      angle = angle - 360
    return angle, centre


  def _get_reflection_matches(
      self,
      s2:Symmetries,
    ) -> tuple[bool,list[Transform]|None]:
    """Get reflections that match s2's polygon to this instance's polygon.

    Args:
      s2 (Symmetries): Symmetries associated with the polygon of interest.

    Returns:
      tuple[bool,list[Transform]]: boolean is True if there are any matches
        when matching transforms will be included in the associated list. If
        there are no matches (False, []) will be returned.

    """
    matches = self.s1.matcher.find_matches(s2.poly_code_r)
    if len(matches) == 0:
      return False, None
    # get mean centroid - all reflection matches will pass through this
    c1 = self.s1.polygon.centroid
    c2 = s2.polygon.centroid
    c = ((c1.x + c2.x) / 2, (c1.y + c2.y) / 2)
    # get each polygon's reflection symmetries
    reflections1 = self.s1.get_reflections(matches)
    reflections2 = s2.get_reflections(matches)
    # their mean angle will be a line of reflection through c
    angles = [(ref1.angle + ref2.angle) / 2
              for ref1, ref2 in zip(reflections1, reflections2, strict = True)]
    transforms = [tiling_utils.get_reflection_transform(angle, c)
                  for angle in angles]
    # there may also be a 'glide' component additional to the reflection
    # to find it reflect the centre of polygon 2 by each transform and
    # then find translation necessary to map it onto the centre of polygon 1
    ctrs2r = [affine.affine_transform(c2, transform)
              for transform in transforms]
    dxdys = [(c1.x - ctr2r.x, c1.y - ctr2r.y) for ctr2r in ctrs2r]
    return (True,
      [Transform(
        "reflection", angle, geom.Point(c), dxdy,
        tiling_utils.combine_transforms([
          tiling_utils.get_reflection_transform(angle, c),
          tiling_utils.get_translation_transform(dxdy[0], dxdy[1])]))
          for angle, dxdy in zip(angles, dxdys, strict = True)])
