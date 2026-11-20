//-----------------------------------------------------------------------------
/*

3D Mesh, 3d triangles connected to create closed manifold objects.

A MeshSDF3 converts a closed, consistently oriented, non-intersecting
triangle mesh into an SDF3. The unsigned distance is the minimum distance to
any triangle; the sign is determined with ray parity (even crossings =
outside, odd = inside), so the inside/outside sign and the triangle normals
follow the topological orientation of the mesh. An r-tree of triangle
bounding boxes accelerates both the nearest-triangle and parity queries.

Open, non-manifold or self-intersecting meshes cannot define a solid and are
rejected by Mesh3D with a *MeshTopologyError rather than silently producing a
meaningless distance field.

*/
//-----------------------------------------------------------------------------

package sdf

import (
	"math"

	v2 "github.com/deadsy/sdfx/vec/v2"
	v3 "github.com/deadsy/sdfx/vec/v3"
	"github.com/dhconnelly/rtreego"
)

//-----------------------------------------------------------------------------

// triangleInfo stores pre-calculated triangle information.
type triangleInfo struct {
	id  int        // triangle index, unique within the mesh
	tri *Triangle3 // original triangle
	m   M44        // rotate/translate to XY matrix
	t   [3]v2.Vec  // transformed triangle vertices
	e   [3]v2.Vec  // anti-clockwise (from +z) unit edge vectors
	n   [3]v2.Vec  // outward pointing unit normals to edge vectors
}

// newTriangleInfo pre-calculates the triangle information.
func newTriangleInfo(t *Triangle3) *triangleInfo {

	m := t.rotateToXY()

	x1 := m.MulPosition(t[1]) // maps to x axis
	x2 := m.MulPosition(t[2]) // maps to xy plane

	// triangle vertices on xy plane
	t0 := v2.Vec{0, 0}
	t1 := v2.Vec{x1.X, 0}
	t2 := v2.Vec{x2.X, x2.Y}

	// triangle edge vectors
	e0 := t1.Sub(t0).Normalize()
	e1 := t2.Sub(t1).Normalize()
	e2 := t0.Sub(t2).Normalize()

	// normals to triangle edges
	n0 := v2.Vec{e0.Y, -e0.X}
	n1 := v2.Vec{e1.Y, -e1.X}
	n2 := v2.Vec{e2.Y, -e2.X}

	return &triangleInfo{
		id:  -1,
		tri: t,
		m:   m,
		t:   [3]v2.Vec{t0, t1, t2},
		e:   [3]v2.Vec{e0, e1, e2},
		n:   [3]v2.Vec{n0, n1, n2},
	}
}

func convertTriangles(tSet []*Triangle3) []*triangleInfo {
	ti := make([]*triangleInfo, len(tSet))
	for i := range tSet {
		info := newTriangleInfo(tSet[i])
		info.id = i
		ti[i] = info
	}
	return ti
}

// minDistance2 returns the minium distance squared between a point and the triangle.
func (a *triangleInfo) minDistance2(p v3.Vec) float64 {

	// See: https://www.researchgate.net/publication/243787422_3D_Distance_from_a_Point_to_a_Triangle
	// We use the 2d method. Rotate/translate the point and triangle so the triangle is in the XY
	// plane and then project p onto the xy plane for consideration of the plane/edge/vertex cases.

	// rotate/translate the point so the triangle is in the xy plane.
	p = a.m.MulPosition(p)

	// pXY is the closest point on the XY plane
	pXY := v2.Vec{p.X, p.Y}

	// pXY wrt the triangle vertices
	pXY0 := pXY.Sub(a.t[0])
	pXY1 := pXY.Sub(a.t[1])
	pXY2 := pXY.Sub(a.t[2])

	d2 := p.Z * p.Z

	// edge 0
	if pXY0.Cross(a.e[0]) > 0 {
		// right of edge 0
		if pXY0.Cross(a.n[0]) > 0 {
			// closest to vertex 0
			return d2 + pXY0.Length2()
		}
		if pXY1.Cross(a.n[0]) < 0 {
			// closest to vertex 1
			return d2 + pXY1.Length2()
		}
		// closest to edge 0
		dn := pXY0.Dot(a.n[0])
		return d2 + (dn * dn)
	}

	// edge 1
	if pXY1.Cross(a.e[1]) > 0 {
		// right of edge 1
		if pXY1.Cross(a.n[1]) > 0 {
			// closest to vertex 1
			return d2 + pXY1.Length2()
		}
		if pXY2.Cross(a.n[1]) < 0 {
			// closest to vertex 2
			return d2 + pXY2.Length2()
		}
		// closest to edge 1
		dn := pXY1.Dot(a.n[1])
		return d2 + (dn * dn)
	}

	// edge 2
	if pXY2.Cross(a.e[2]) > 0 {
		// right of edge 2
		if pXY2.Cross(a.n[2]) > 0 {
			// closest to vertex 2
			return d2 + pXY2.Length2()
		}
		if pXY0.Cross(a.n[2]) < 0 {
			// closest to vertex 0
			return d2 + pXY0.Length2()
		}
		// closest to edge 2
		dn := pXY2.Dot(a.n[2])
		return d2 + (dn * dn)
	}

	// left of all edges, pXY is in the triangle
	return d2
}

//-----------------------------------------------------------------------------
// Triangle / AABB overlap (broad phase).
//
// See: C. Ericson, Real-Time Collision Detection, section 5.2.3 (test of a
// triangle against an AABB using the separating axis theorem).

// triangleOverlapsBox returns true if triangle t overlaps (or is inside) box b.
func triangleOverlapsBox(t *Triangle3, b Box3) bool {
	center := b.Center()
	half := b.Size().MulScalar(0.5)

	// move the box center to the origin
	v := [3]v3.Vec{t[0].Sub(center), t[1].Sub(center), t[2].Sub(center)}
	e := [3]v3.Vec{
		v[1].Sub(v[0]),
		v[2].Sub(v[1]),
		v[0].Sub(v[2]),
	}

	// test the 3 AABB face normals
	min := v[0].Min(v[1]).Min(v[2])
	max := v[0].Max(v[1]).Max(v[2])
	if min.X > half.X || max.X < -half.X {
		return false
	}
	if min.Y > half.Y || max.Y < -half.Y {
		return false
	}
	if min.Z > half.Z || max.Z < -half.Z {
		return false
	}

	// test the 9 axes a_i x e_j (one per box axis / edge pair)
	fe := [3]v3.Vec{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			axis := fe[i].Cross(e[j])
			// project the triangle onto the axis
			p0 := v[0].Dot(axis)
			p1 := v[1].Dot(axis)
			p2 := v[2].Dot(axis)
			r := half.X*math.Abs(fe[0].Dot(axis)) +
				half.Y*math.Abs(fe[1].Dot(axis)) +
				half.Z*math.Abs(fe[2].Dot(axis))
			if math.Max(-math.Max(p0, math.Max(p1, p2)), math.Min(p0, math.Min(p1, p2))) > r {
				return false
			}
		}
	}

	// test the triangle normal
	normal := e[0].Cross(e[1])
	planeDist := normal.Dot(v[0])
	boxRadius := half.X*math.Abs(normal.X) + half.Y*math.Abs(normal.Y) + half.Z*math.Abs(normal.Z)
	if math.Abs(planeDist) > boxRadius {
		return false
	}

	return true
}

//-----------------------------------------------------------------------------
// Ray/AABB intersection, used to prune r-tree searches for parity tests.

// rayHitsBox is a slab test for an axis-aligned ray against a box. A ray
// starting exactly on a box face counts as entering the box (tmin == 0), so
// query points on a triangle plane keep a well defined crossing count.
func rayHitsBox(b Box3, rp, rv v3.Vec) bool {
	tmin := math.Inf(-1)
	tmax := math.Inf(1)
	lo := [3]float64{b.Min.X, b.Min.Y, b.Min.Z}
	hi := [3]float64{b.Max.X, b.Max.Y, b.Max.Z}
	o := [3]float64{rp.X, rp.Y, rp.Z}
	d := [3]float64{rv.X, rv.Y, rv.Z}

	for axis := 0; axis < 3; axis++ {
		if d[axis] == 0 {
			// parallel ray: the origin must lie within the slab
			if o[axis] < lo[axis] || o[axis] > hi[axis] {
				return false
			}
			continue
		}
		inv := 1 / d[axis]
		t1 := (lo[axis] - o[axis]) * inv
		t2 := (hi[axis] - o[axis]) * inv
		if t1 > t2 {
			t1, t2 = t2, t1
		}
		if t1 > tmin {
			tmin = t1
		}
		if t2 < tmax {
			tmax = t2
		}
		if tmin > tmax {
			return false
		}
	}
	return tmax >= 0
}

//-----------------------------------------------------------------------------
// Mesh3D. 3D mesh evaluation with r-tree acceleration.

// triEntry pairs a triangle with its r-tree rectangle. It is shared by the
// distance/parity r-tree of MeshSDF3 and the broad-phase r-tree used for
// self-intersection validation.
type triEntry struct {
	tri  *Triangle3
	id   int
	rect rtreego.Rect
}

// Bounds implements rtreego.Spatial.
func (e *triEntry) Bounds() rtreego.Rect {
	return e.rect
}

// MeshSDF3 is an SDF3 made from a closed manifold set of 3d triangles.
type MeshSDF3 struct {
	mesh  []*Triangle3    // validated, outward-wound triangles (mesh attributes)
	info  []*triangleInfo // pre-calculated triangle data
	rtree *rtreego.Rtree  // r-tree over triangle bounding boxes
	bb    Box3            // bounding box
}

// Mesh3D returns an SDF3 made from a set of triangles. The mesh must be a
// closed, consistently oriented, non-self-intersecting 2-manifold; otherwise
// a *MeshTopologyError is returned. On success the returned triangles are
// wound consistently outwards (the input slice is not modified).
func Mesh3D(mesh []*Triangle3) (SDF3, error) {
	clean, err := ValidateManifoldMesh(mesh)
	if err != nil {
		return nil, err
	}

	// work out the bounding box
	bb := clean[0].BoundingBox()
	for _, t := range clean {
		bb = bb.Extend(t.BoundingBox())
	}

	info := convertTriangles(clean)

	// r-tree over triangle AABBs; the same structure serves distance and
	// parity queries, with per-query de-duplication of overlapping boxes
	entries := make([]rtreego.Spatial, len(clean))
	for i, t := range clean {
		b := t.BoundingBox()
		rect, _ := rtreego.NewRectFromPoints(v3ToPoint(b.Min), v3ToPoint(b.Max))
		entries[i] = &triEntry{tri: t, id: i, rect: rect}
	}
	rtree := rtreego.NewTree(3, 3, 5, entries...)

	return &MeshSDF3{
		mesh:  clean,
		info:  info,
		rtree: rtree,
		bb:    bb,
	}, nil
}

// nearestDistance2 returns the squared distance to the nearest triangle.
// A box centered on the query point is grown until its half-side reaches the
// current best distance: every triangle whose AABB intersects it is a valid
// candidate and no closer triangle can be missed. A per-query set
// de-duplicates triangles returned through overlapping search boxes.
func (s *MeshSDF3) nearestDistance2(p v3.Vec) float64 {
	best := math.MaxFloat64
	visited := make(map[int]struct{}, 32)

	extent := s.bb.Size().MaxComponent()
	radius := extent * 1e-9
	for radius <= extent*16 {
		searchBox := NewBox3(p, v3.Vec{2 * radius, 2 * radius, 2 * radius})
		rect, err := rtreego.NewRectFromPoints(
			v3ToPoint(searchBox.Min), v3ToPoint(searchBox.Max))
		if err != nil {
			radius *= 2
			continue
		}
		for _, c := range s.rtree.SearchIntersect(rect) {
			e := c.(*triEntry)
			if _, seen := visited[e.id]; seen {
				continue
			}
			visited[e.id] = struct{}{}
			if d2 := s.info[e.id].minDistance2(p); d2 < best {
				best = d2
			}
		}
		// half-side >= best distance means every closer triangle has been seen
		if best <= radius*radius {
			return best
		}
		radius *= 2
	}

	// safety net: the growing search did not converge (e.g. a point far from
	// the mesh). Scan every triangle so the result can never be undefined.
	for _, ti := range s.info {
		if d2 := ti.minDistance2(p); d2 < best {
			best = d2
		}
	}
	return best
}

// rayCandidates returns every triangle whose AABB can intersect the ray from
// p in direction rv. The search rectangle covers the segment from p to a
// point a diagonal-length past the far side of the mesh bounding box, so no
// crossing triangle can be missed regardless of where p lies; an exact
// ray/AABB slab test then filters out boxes that merely touch the rectangle.
func (s *MeshSDF3) rayCandidates(p, rv v3.Vec) []*triangleInfo {
	// length guaranteed to exceed every ray/box crossing distance
	reach := s.bb.ScaleAboutCenter(1.01).Size().Length() * 2
	end := p.Add(rv.MulScalar(reach))
	lo := p.Min(end)
	hi := p.Max(end)
	rect, err := rtreego.NewRectFromPoints(v3ToPoint(lo), v3ToPoint(hi))
	if err != nil {
		// degenerate rectangle (should not happen for a unit ray): fall back
		// to scanning every triangle
		return s.info
	}
	var out []*triangleInfo
	for _, c := range s.rtree.SearchIntersect(rect) {
		e := c.(*triEntry)
		if rayHitsBox(e.tri.BoundingBox(), p, rv) {
			out = append(out, s.info[e.id])
		}
	}
	return out
}

// parityRayDir is the direction used for inside/outside tests. All three
// components are non-zero (an octant direction), so the ray is never parallel
// to an axis-aligned triangle edge and cannot pass exactly through an edge
// shared by two coplanar triangles, which would double-count a crossing.
var parityRayDir = v3.Vec{1, 1.3796273112749635, 1.7342957453123987}

// Evaluate returns the signed minimum distance to the mesh. Negative values
// are inside the closed solid, positive values outside; the sign follows the
// topological orientation of the triangles.
func (s *MeshSDF3) Evaluate(p v3.Vec) float64 {
	// unsigned distance to the nearest triangle
	d := math.Sqrt(s.nearestDistance2(p))
	if s.inside(p) {
		return -d
	}
	return d
}

// inside returns true if p is inside the closed mesh (odd ray parity).
func (s *MeshSDF3) inside(p v3.Vec) bool {
	crossings := 0
	for _, ti := range s.rayCandidates(p, parityRayDir) {
		crossings += ti.tri.rayCrossCount(p, parityRayDir)
	}
	return crossings&1 == 1
}

// Mesh returns the validated, consistently oriented triangles of the mesh.
// This preserves the mesh attributes for downstream tooling.
func (s *MeshSDF3) Mesh() []*Triangle3 {
	return s.mesh
}

// BoundingBox returns the bounding box of a 3d mesh.
func (s *MeshSDF3) BoundingBox() Box3 {
	return s.bb
}

//-----------------------------------------------------------------------------
// Mesh3D Slow. Provided for testing and benchmarking purposes. It evaluates
// every triangle directly and must return the same signed distances as
// Mesh3D (up to floating point order differences).

// MeshSDF3Slow is an SDF3 made from a closed manifold set of 3d triangles.
type MeshSDF3Slow struct {
	mesh []*Triangle3
	info []*triangleInfo
	bb   Box3 // bounding box
}

// Mesh3DSlow returns an SDF3 made from a set of triangles. As with Mesh3D the
// mesh must be a closed, consistently oriented, non-self-intersecting
// 2-manifold.
func Mesh3DSlow(mesh []*Triangle3) (SDF3, error) {
	clean, err := ValidateManifoldMesh(mesh)
	if err != nil {
		return nil, err
	}

	// work out the bounding box
	bb := clean[0].BoundingBox()
	for _, t := range clean {
		bb = bb.Extend(t.BoundingBox())
	}

	return &MeshSDF3Slow{
		mesh: clean,
		info: convertTriangles(clean),
		bb:   bb,
	}, nil
}

// Evaluate returns the signed minimum distance to the mesh.
func (s *MeshSDF3Slow) Evaluate(p v3.Vec) float64 {
	d2 := math.MaxFloat64
	crossings := 0
	for _, ti := range s.info {
		d2 = math.Min(d2, ti.minDistance2(p))
		crossings += ti.tri.rayCrossCount(p, parityRayDir)
	}
	d := math.Sqrt(d2)
	if crossings&1 == 1 {
		// p is inside the closed solid
		return -d
	}
	return d
}

// Mesh returns the validated, consistently oriented triangles of the mesh.
func (s *MeshSDF3Slow) Mesh() []*Triangle3 {
	return s.mesh
}

// BoundingBox returns the bounding box of a 3d mesh.
func (s *MeshSDF3Slow) BoundingBox() Box3 {
	return s.bb
}

//-----------------------------------------------------------------------------
