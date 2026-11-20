//-----------------------------------------------------------------------------
/*

3D Triangles

*/
//-----------------------------------------------------------------------------

package sdf

import (
	"math"
	"sync"

	v3 "github.com/deadsy/sdfx/vec/v3"
	"github.com/dhconnelly/rtreego"
)

//-----------------------------------------------------------------------------

// Triangle3 is a 3D triangle
type Triangle3 [3]v3.Vec

// Normal returns the normal vector to the plane defined by the 3D triangle.
func (t *Triangle3) Normal() v3.Vec {
	e1 := t[1].Sub(t[0])
	e2 := t[2].Sub(t[0])
	return e1.Cross(e2).Normalize()
}

// Degenerate returns true if the triangle is degenerate.
func (t *Triangle3) Degenerate(tolerance float64) bool {
	// check for identical vertices
	if t[0].Equals(t[1], tolerance) {
		return true
	}
	if t[1].Equals(t[2], tolerance) {
		return true
	}
	if t[2].Equals(t[0], tolerance) {
		return true
	}
	// TODO more tests needed
	return false
}

// BoundingBox returns a bounding box for the triangle.
func (t *Triangle3) BoundingBox() Box3 {
	return Box3{Min: t[0], Max: t[0]}.Include(t[1]).Include(t[2])
}

// Equals tests if two triangles are equal within tolerance.
func (t *Triangle3) Equals(a *Triangle3, tolerance float64) bool {
	return t[0].Equals(a[0], tolerance) &&
		t[1].Equals(a[1], tolerance) &&
		t[2].Equals(a[2], tolerance)
}

// rotateVertex rotates the vertices of a triangle
func (t *Triangle3) rotateVertex() Triangle3 {
	return Triangle3{t[2], t[0], t[1]}
}

//-----------------------------------------------------------------------------

func v3ToPoint(v v3.Vec) rtreego.Point {
	return rtreego.Point{v.X, v.Y, v.Z}
}

// Bounds returns a r-tree bounding rectangle for the triangle.
func (t *Triangle3) Bounds() rtreego.Rect {
	b := t.BoundingBox()
	r, _ := rtreego.NewRectFromPoints(v3ToPoint(b.Min), v3ToPoint(b.Max))
	return r
}

//-----------------------------------------------------------------------------

// rotateToXY returns the transformation matrix that maps
// t[0] to the origin, t[1] to the x axis (x > 0), t[2] to the xy plane (y > 0)
func (t *Triangle3) rotateToXY() M44 {

	a := t[0]        // maps to the origin
	b := t[1].Sub(a) // maps to the x-axis
	c := t[2].Sub(a) // maps to the xy-plane

	// u maps to the x-axis
	u := b.Normalize()
	// w maps to the z-axis (normal to the triangle plane)
	w := c.Cross(u).Normalize()
	// v maps to the xy-plane (in the plane of the triangle)
	v := u.Cross(w)

	// translate to the origin
	m := Translate3d(a.Neg())

	return M44{
		u.X, u.Y, u.Z, 0,
		v.X, v.Y, v.Z, 0,
		w.X, w.Y, w.Z, 0,
		0, 0, 0, 1,
	}.Mul(m)
}

//-----------------------------------------------------------------------------
// https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm

func (t *Triangle3) intersectRay(rp, rv v3.Vec) []v3.Vec {

	e1 := t[1].Sub(t[0])
	e2 := t[2].Sub(t[0])
	h := rv.Cross(e2)
	a := e1.Dot(h)

	if EqualFloat64(a, 0, epsilon) {
		// This ray is parallel to this triangle.
		return nil
	}

	f := 1.0 / a
	s := rp.Sub(t[0])
	u := f * s.Dot(h)

	if u < 0 || u > 1 {
		return nil
	}

	q := s.Cross(e1)
	v := f * rv.Dot(q)

	if v < 0 || u+v > 1 {
		return nil
	}

	// At this stage we can compute t to find out where the intersection point is on the line.
	rt := f * e2.Dot(q)

	if rt > epsilon {
		return []v3.Vec{rp.Add(rv.MulScalar(rt))}
	}

	// This means that there is a line intersection but not a ray intersection.
	return nil
}

// rayCrossCount counts how many times the ray from p (in direction rv)
// crosses the triangle. Unlike intersectRay, a hit at rt == 0 (the ray
// starting on the triangle plane) counts, because inside/outside parity must
// be well defined for query points on the mesh surface or on octree
// boundaries. Edge-on (parallel) rays miss, so axis-aligned rays only
// degenerate against triangles coplanar with that axis, which are skipped.
func (t *Triangle3) rayCrossCount(rp, rv v3.Vec) int {
	e1 := t[1].Sub(t[0])
	e2 := t[2].Sub(t[0])
	h := rv.Cross(e2)
	a := e1.Dot(h)
	if EqualFloat64(a, 0, epsilon) {
		// ray parallel to the triangle plane
		return 0
	}
	f := 1.0 / a
	s := rp.Sub(t[0])
	u := f * s.Dot(h)
	if u < 0 || u > 1 {
		return 0
	}
	q := s.Cross(e1)
	v := f * rv.Dot(q)
	if v < 0 || u+v > 1 {
		return 0
	}
	rt := f * e2.Dot(q)
	// rt >= 0: the crossing lies on the ray, including at its origin
	if rt >= 0 {
		return 1
	}
	return 0
}

//-----------------------------------------------------------------------------
// Triangle/triangle intersection (Möller's separating axis method).
//
// See: T. Möller, "A Fast Triangle-Triangle Intersection Test", 1997.
// The two triangles intersect if the segment where their planes overlap
// pierces both triangles. Coplanar triangles are handled with a 2D projection
// overlap test.

// trianglePlaneInterval projects the triangle vertices onto the line of
// intersection of the two planes (axis) and returns the covered interval.
func trianglePlaneInterval(t *Triangle3, axis v3.Vec) (lo, hi float64) {
	lo = math.MaxFloat64
	hi = -math.MaxFloat64
	for k := 0; k < 3; k++ {
		p := t[k].Dot(axis)
		if p < lo {
			lo = p
		}
		if p > hi {
			hi = p
		}
	}
	return lo, hi
}

// triIntersectEps is the absolute tolerance for the triangle/triangle
// separating-axis tests. Meshes in this package work in millimetre scale
// coordinates, where 1e-12 is well below numerical noise.
const triIntersectEps = 1e-12

// intersectTriangle returns true if the two triangles intersect. Triangles
// sharing a vertex or edge should be filtered out by the caller.
func (t *Triangle3) intersectTriangle(o *Triangle3) bool {
	n1 := t[1].Sub(t[0]).Cross(t[2].Sub(t[0]))
	n2 := o[1].Sub(o[0]).Cross(o[2].Sub(o[0]))

	d1 := o[0].Sub(t[0]).Dot(n1)
	d2 := o[1].Sub(t[0]).Dot(n1)
	d3 := o[2].Sub(t[0]).Dot(n1)

	if math.Abs(d1) < triIntersectEps && math.Abs(d2) < triIntersectEps && math.Abs(d3) < triIntersectEps {
		// coplanar: fall back to a projected 2d overlap test
		return t.coplanarOverlap(o, n1)
	}

	if (d1 > triIntersectEps && d2 > triIntersectEps && d3 > triIntersectEps) ||
		(d1 < -triIntersectEps && d2 < -triIntersectEps && d3 < -triIntersectEps) {
		// triangle o lies entirely on one side of t's plane
		return false
	}

	// direction of the line of intersection of the two planes
	axis := n1.Cross(n2)
	if axis.Length2() < triIntersectEps {
		// parallel planes that are not coplanar never intersect
		return false
	}

	lo1, hi1 := trianglePlaneInterval(o, axis)
	lo2, hi2 := trianglePlaneInterval(t, axis)

	return lo1 <= hi2+triIntersectEps && lo2 <= hi1+triIntersectEps
}

// coplanarOverlap tests two triangles known to lie in (nearly) the same plane
// for overlap, by projecting onto the dominant axis of the plane normal.
func (t *Triangle3) coplanarOverlap(o *Triangle3, n v3.Vec) bool {
	// scale the tolerance to the triangle coordinates so the absolute
	// 1e-12 constant does not misclassify points at normal model scales
	eps := triIntersectEps * (1 + t.BoundingBox().Size().MaxComponent())
	// project onto the two axes most perpendicular to the plane normal:
	// the normal's dominant component names the axis to discard
	ax := math.Abs(n.X)
	ay := math.Abs(n.Y)
	az := math.Abs(n.Z)
	proj := func(p v3.Vec) (float64, float64) {
		switch {
		case ax >= ay && ax >= az:
			// normal points mainly along X: project onto YZ
			return p.Y, p.Z
		case ay >= ax && ay >= az:
			// normal points mainly along Y: project onto XZ
			return p.X, p.Z
		default:
			// normal points mainly along Z: project onto XY
			return p.X, p.Y
		}
	}
	var pa [3][2]float64
	var pb [3][2]float64
	for k := 0; k < 3; k++ {
		pa[k][0], pa[k][1] = proj(t[k])
		pb[k][0], pb[k][1] = proj(o[k])
	}
	// edge/edge intersection test in the projection plane
	segIntersect := func(a0, a1, b0, b1 [2]float64) bool {
		d := (a1[0]-a0[0])*(b1[1]-b0[1]) - (a1[1]-a0[1])*(b1[0]-b0[0])
		if math.Abs(d) < eps {
			return false
		}
		q := ((b0[0]-a0[0])*(b1[1]-b0[1]) - (b0[1]-a0[1])*(b1[0]-b0[0])) / d
		r := ((b0[0]-a0[0])*(a1[1]-a0[1]) - (b0[1]-a0[1])*(a1[0]-a0[0])) / d
		return q >= -eps && q <= 1+eps && r >= -eps && r <= 1+eps
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if segIntersect(pa[i], pa[(i+1)%3], pb[j], pb[(j+1)%3]) {
				return true
			}
		}
	}
	// no edge crossings: one triangle may be entirely inside the other
	pointInTri := func(p [2]float64, tri [3][2]float64) bool {
		sign := func(a, b, p [2]float64) float64 {
			return (a[0]-p[0])*(b[1]-p[1]) - (b[0]-p[0])*(a[1]-p[1])
		}
		d1 := sign(tri[0], tri[1], p)
		d2 := sign(tri[1], tri[2], p)
		d3 := sign(tri[2], tri[0], p)
		hasNeg := d1 < -eps || d2 < -eps || d3 < -eps
		hasPos := d1 > eps || d2 > eps || d3 > eps
		return !(hasNeg && hasPos)
	}
	if pointInTri(pa[0], pb) || pointInTri(pb[0], pa) {
		return true
	}
	return false
}

//-----------------------------------------------------------------------------

// WriteTriangles writes a stream of triangles to a slice.
func WriteTriangles(wg *sync.WaitGroup, triangles *[]*Triangle3) chan<- []*Triangle3 {
	// External code writes triangles to this channel.
	// This goroutine reads the channel and appends the triangles to a slice.
	c := make(chan []*Triangle3)

	wg.Add(1)
	go func() {
		defer wg.Done()
		// read triangles from the channel and append them to the slice
		for ts := range c {
			for _, t := range ts {
				*triangles = append(*triangles, t)
			}
		}
	}()

	return c
}

//-----------------------------------------------------------------------------
// Triangle3 Buffering

// We write triangles to a channel to decouple the rendering routines from the
// routine that writes file output. We have a lot of triangles and channels
// are not very fast, so it's best to bundle many triangles into a single channel
// write. The renderer doesn't naturally do that, so we buffer triangles before
// writing them to the channel.

// Triangle3Writer is the interface of a triangle writer/closer object.
type Triangle3Writer interface {
	Write(in []*Triangle3) error
	Close() error
}

// size the buffer to avoid re-allocations when appending.
const tBufferSize = 256
const tBufferMargin = 8 // marching cubes produces 0 to 5 triangles

// Triangle3Buffer buffers triangles before writing them to a channel.
type Triangle3Buffer struct {
	buf  []*Triangle3        // triangle buffer
	out  chan<- []*Triangle3 // output channel
	lock sync.Mutex          // lock the the buffer during access
}

// NewTriangle3Buffer returns a Triangle3Buffer.
func NewTriangle3Buffer(out chan<- []*Triangle3) Triangle3Writer {
	return &Triangle3Buffer{
		buf: make([]*Triangle3, 0, tBufferSize+tBufferMargin),
		out: out,
	}
}

func (a *Triangle3Buffer) Write(in []*Triangle3) error {
	a.lock.Lock()
	a.buf = append(a.buf, in...)
	if len(a.buf) >= tBufferSize {
		a.out <- a.buf
		a.buf = make([]*Triangle3, 0, tBufferSize+tBufferMargin)
	}
	a.lock.Unlock()
	return nil
}

// Close flushes out any remaining triangles in the buffer.
func (a *Triangle3Buffer) Close() error {
	a.lock.Lock()
	if len(a.buf) != 0 {
		a.out <- a.buf
		a.buf = nil
	}
	a.lock.Unlock()
	return nil
}

//-----------------------------------------------------------------------------
