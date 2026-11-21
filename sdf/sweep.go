//-----------------------------------------------------------------------------
/*

Sweep a 2D cross-section along a 3D path.

The cross-section (an SDF2, origin on the path) is swept along a 3D path
defined by a set of control points. The path may be a polyline (straight
segments) or a smooth Catmull-Rom spline through the control points. The
cross-section frame is parallel transported along the path tangent so the
sweep does not twist unexpectedly, and an additional uniform twist may be
applied. Closed paths are made seamless by distributing the frame closure
error evenly around the loop.

The 2D x-axis maps to the frame normal and the 2D y-axis to the frame
binormal. For open paths the ends are capped with flat discs.

*/
//-----------------------------------------------------------------------------

package sdf

import (
	"math"

	v2 "github.com/deadsy/sdfx/vec/v2"
	v3 "github.com/deadsy/sdfx/vec/v3"
)

//-----------------------------------------------------------------------------

// SweepMode selects how the sweep path interpolates its control points.
type SweepMode int

const (
	// SweepPolyline joins the control points with straight segments.
	SweepPolyline SweepMode = iota
	// SweepSmooth interpolates the control points with a Catmull-Rom spline.
	SweepSmooth
)

// samples per path segment for the frame table
const sweepPolylineSamples = 8
const sweepSmoothSamples = 16

// substeps for segment arc length estimation (smooth paths)
const sweepArcLenSteps = 16

//-----------------------------------------------------------------------------
// 3D path with arc-length parameterization and parallel transported frames

// sweepPath is a 3D path with an associated moving frame.
type sweepPath struct {
	mode   SweepMode // path interpolation mode
	closed bool      // true if the path is a closed loop
	pts    []v3.Vec  // control points
	nseg   int       // number of segments
	segLen []float64 // arc length of each segment
	cum    []float64 // cumulative arc length at segment starts (len nseg+1)
	total  float64   // total path length
	segC   []v3.Vec  // segment bounding sphere centers
	segR   []float64 // segment bounding sphere radii
	// frame table (uniformly spaced in segment/parameter space)
	su []float64 // arc-length fraction at each sample
	sn []v3.Vec  // frame normal at each sample
	// end frames for the end cap half-spaces (open paths only)
	pos0, tn0 v3.Vec // position and tangent at the path start
	pos1, tn1 v3.Vec // position and tangent at the path end
}

// crPoints returns the 4 Catmull-Rom control points for segment i.
func (p *sweepPath) crPoints(i int) (p0, p1, p2, p3 v3.Vec) {
	n := len(p.pts)
	if p.closed {
		return p.pts[(i-1+n)%n], p.pts[i], p.pts[(i+1)%n], p.pts[(i+2)%n]
	}
	p1 = p.pts[i]
	p2 = p.pts[i+1]
	p0 = p1
	if i > 0 {
		p0 = p.pts[i-1]
	}
	p3 = p2
	if i+2 < n {
		p3 = p.pts[i+2]
	}
	return
}

// pos returns the position on segment i at local parameter t in [0,1].
func (p *sweepPath) pos(i int, t float64) v3.Vec {
	if p.mode == SweepPolyline {
		p0 := p.pts[i]
		p1 := p.pts[(i+1)%len(p.pts)]
		return p0.Add(p1.Sub(p0).MulScalar(t))
	}
	p0, p1, p2, p3 := p.crPoints(i)
	// Catmull-Rom basis (uniform parameterization)
	a := p1.MulScalar(2)
	b := p2.Sub(p0).MulScalar(t)
	c := p0.MulScalar(2).Sub(p1.MulScalar(5)).Add(p2.MulScalar(4)).Sub(p3).MulScalar(t * t)
	d := p1.MulScalar(3).Sub(p0).Sub(p2.MulScalar(3)).Add(p3).MulScalar(t * t * t)
	return a.Add(b).Add(c).Add(d).MulScalar(0.5)
}

// tangent returns the unit tangent on segment i at local parameter t.
func (p *sweepPath) tangent(i int, t float64) v3.Vec {
	var d v3.Vec
	if p.mode == SweepPolyline {
		d = p.pts[(i+1)%len(p.pts)].Sub(p.pts[i])
	} else {
		p0, p1, p2, p3 := p.crPoints(i)
		// Catmull-Rom derivative
		a := p2.Sub(p0)
		b := p0.MulScalar(2).Sub(p1.MulScalar(5)).Add(p2.MulScalar(4)).Sub(p3).MulScalar(2 * t)
		c := p1.MulScalar(3).Sub(p0).Sub(p2.MulScalar(3)).Add(p3).MulScalar(3 * t * t)
		d = a.Add(b).Add(c).MulScalar(0.5)
	}
	if d.Length2() < epsilon {
		// degenerate derivative - use the segment direction
		d = p.pts[(i+1)%len(p.pts)].Sub(p.pts[i])
	}
	return d.Normalize()
}

// u returns the arc-length fraction for a segment/local-t pair.
func (p *sweepPath) u(seg int, t float64) float64 {
	return (p.cum[seg] + t*p.segLen[seg]) / p.total
}

// perpendicular returns a unit vector perpendicular to t.
func perpendicular(t v3.Vec) v3.Vec {
	// use the cardinal axis least aligned with t
	a := t.Abs()
	var axis v3.Vec
	switch {
	case a.X <= a.Y && a.X <= a.Z:
		axis = v3.Vec{1, 0, 0}
	case a.Y <= a.Z:
		axis = v3.Vec{0, 1, 0}
	default:
		axis = v3.Vec{0, 0, 1}
	}
	return axis.Sub(t.MulScalar(axis.Dot(t))).Normalize()
}

// rotateAbout rotates v about unit axis k by angle (right-hand rule).
func rotateAbout(v, k v3.Vec, angle float64) v3.Vec {
	c := math.Cos(angle)
	s := math.Sin(angle)
	return v.MulScalar(c).
		Add(k.Cross(v).MulScalar(s)).
		Add(k.MulScalar(k.Dot(v) * (1 - c)))
}

// buildFrames constructs the parallel transported frame table.
func (p *sweepPath) buildFrames() {
	k := sweepPolylineSamples
	if p.mode == SweepSmooth {
		k = sweepSmoothSamples
	}
	// number of frame samples
	n := p.nseg * k
	if !p.closed {
		n++ // add the end point of the last segment
	}
	p.su = make([]float64, n)
	p.sn = make([]v3.Vec, n)
	tn := make([]v3.Vec, n)
	// sample positions/tangents
	for i := 0; i < n; i++ {
		seg := i / k
		t := float64(i%k) / float64(k)
		if seg == p.nseg {
			// end point of an open path
			seg--
			t = 1
		}
		p.su[i] = p.u(seg, t)
		tn[i] = p.tangent(seg, t)
	}
	// parallel transport the normal along the path
	p.sn[0] = perpendicular(tn[0])
	for i := 1; i < n; i++ {
		axis := tn[i-1].Cross(tn[i])
		l := axis.Length()
		if l < epsilon {
			// tangents are (anti-)parallel - carry the normal forward
			p.sn[i] = p.sn[i-1]
			continue
		}
		angle := math.Atan2(l, tn[i-1].Dot(tn[i]))
		p.sn[i] = rotateAbout(p.sn[i-1], axis.DivScalar(l), angle)
	}
	if p.closed {
		// Transport the last sample normal back to the start tangent and
		// measure the closure error. Distribute the correction evenly
		// around the loop so the frame (and hence the solid) is seamless.
		axis := tn[n-1].Cross(tn[0])
		nEnd := p.sn[n-1]
		if l := axis.Length(); l >= epsilon {
			angle := math.Atan2(l, tn[n-1].Dot(tn[0]))
			nEnd = rotateAbout(nEnd, axis.DivScalar(l), angle)
		}
		theta := math.Atan2(tn[0].Dot(nEnd.Cross(p.sn[0])), nEnd.Dot(p.sn[0]))
		for i := 1; i < n; i++ {
			p.sn[i] = rotateAbout(p.sn[i], tn[i], theta*p.su[i])
		}
	}
}

// frame returns the position, tangent, normal and binormal at (seg, t).
func (p *sweepPath) frame(seg int, t float64) (pos, tn, n, b v3.Vec) {
	pos = p.pos(seg, t)
	tn = p.tangent(seg, t)
	if p.mode == SweepPolyline && t == 0 && (p.closed || seg > 0) {
		// miter frame at a joint: use the bisector of the segment tangents
		prev := (seg - 1 + p.nseg) % p.nseg
		if tb := p.tangent(prev, 1).Add(tn); tb.Length2() >= epsilon {
			tn = tb.Normalize()
		}
	}
	// interpolate the normal from the frame table
	k := float64(len(p.sn)) / float64(p.nseg)
	if !p.closed {
		k = float64(len(p.sn)-1) / float64(p.nseg)
	}
	g := (float64(seg) + t) * k
	i0 := int(g)
	f := g - float64(i0)
	i1 := i0 + 1
	if p.closed {
		i0 %= len(p.sn)
		i1 %= len(p.sn)
	} else if i1 >= len(p.sn) {
		i0 = len(p.sn) - 1
		i1 = i0
		f = 0
	}
	n = p.sn[i0].MulScalar(1 - f).Add(p.sn[i1].MulScalar(f))
	// re-orthogonalize against the tangent
	n = n.Sub(tn.MulScalar(n.Dot(tn)))
	if n.Length2() < epsilon {
		n = perpendicular(tn)
	} else {
		n = n.Normalize()
	}
	b = tn.Cross(n)
	return
}

// closest returns the segment and local t of the closest point on the path.
func (p *sweepPath) closest(pt v3.Vec) (int, float64) {
	best := math.MaxFloat64
	bestSeg, bestT := 0, 0.0
	for i := 0; i < p.nseg; i++ {
		// cull the segment with its bounding sphere
		if d := pt.Sub(p.segC[i]).Length() - p.segR[i]; d > 0 && d*d >= best {
			continue
		}
		var t float64
		if p.mode == SweepPolyline {
			// exact projection onto the segment
			a := p.pts[i]
			ab := p.pts[(i+1)%len(p.pts)].Sub(a)
			t = Clamp(pt.Sub(a).Dot(ab)/ab.Length2(), 0, 1)
		} else {
			t = p.segClosestSmooth(i, pt)
		}
		if d2 := pt.Sub(p.pos(i, t)).Length2(); d2 < best {
			best, bestSeg, bestT = d2, i, t
		}
	}
	// canonicalize: the end of a segment is the start of the next one
	if bestT == 1 && (p.closed || bestSeg < p.nseg-1) {
		bestSeg = (bestSeg + 1) % p.nseg
		bestT = 0
	}
	return bestSeg, bestT
}

// segClosestSmooth returns the local t of the closest point on a smooth segment.
func (p *sweepPath) segClosestSmooth(i int, pt v3.Vec) float64 {
	// ternary search (the distance along a short spline segment is unimodal)
	lo, hi := 0.0, 1.0
	for j := 0; j < 40; j++ {
		m1 := lo + (hi-lo)/3
		m2 := hi - (hi-lo)/3
		if pt.Sub(p.pos(i, m1)).Length2() < pt.Sub(p.pos(i, m2)).Length2() {
			hi = m2
		} else {
			lo = m1
		}
	}
	t := 0.5 * (lo + hi)
	// Snap to the segment ends. A minimum constrained to the end of the
	// segment converges towards (but never reaches) the end - snap to it
	// so that end-of-path handling in Evaluate is well defined.
	const snap = 1e-6
	if t < snap {
		t = 0
	} else if t > 1-snap {
		t = 1
	}
	// the segment ends may be the true minimum
	best := pt.Sub(p.pos(i, t)).Length2()
	if d := pt.Sub(p.pos(i, 0)).Length2(); d < best {
		best, t = d, 0
	}
	if d := pt.Sub(p.pos(i, 1)).Length2(); d < best {
		best, t = d, 1
	}
	return t
}

// boundingBox returns the bounding box of the path centerline.
func (p *sweepPath) boundingBox() Box3 {
	bb := Box3{p.pts[0], p.pts[0]}
	for i := 0; i < p.nseg; i++ {
		bb = bb.Include(p.pos(i, 0))
		bb = bb.Include(p.pos(i, 1))
		if p.mode == SweepPolyline {
			continue
		}
		// include the extrema of the cubic segment
		p0, p1, p2, p3 := p.crPoints(i)
		a := p2.Sub(p0)
		b := p0.MulScalar(2).Sub(p1.MulScalar(5)).Add(p2.MulScalar(4)).Sub(p3)
		c := p1.MulScalar(3).Sub(p0).Sub(p2.MulScalar(3)).Add(p3)
		for axis := 0; axis < 3; axis++ {
			roots, _ := quadratic(3*c.Get(axis), 2*b.Get(axis), a.Get(axis))
			for _, t := range roots {
				if t > 0 && t < 1 {
					bb = bb.Include(p.pos(i, t))
				}
			}
		}
	}
	return bb
}

//-----------------------------------------------------------------------------

// SweepSDF3 sweeps a 2D cross-section along a 3D path.
type SweepSDF3 struct {
	sdf   SDF2       // the cross-section being swept
	path  *sweepPath // the sweep path
	twist float64    // total twist (radians) along the path
	bb    Box3       // bounding box
}

// Sweep3D returns an SDF3 made by sweeping an SDF2 cross-section along a 3D
// path. The path is defined by its control points: mode selects straight
// segments (SweepPolyline) or a smooth Catmull-Rom spline (SweepSmooth).
// If closed is true the path is treated as a closed loop (a repeated first
// point at the end of the path is ignored) and the swept solid is seamless.
// twist is the total cross-section rotation in radians applied uniformly
// along the length of the path. The cross-section's origin follows the path,
// its x-axis maps to the frame normal and its y-axis to the frame binormal.
func Sweep3D(sdf SDF2, path []v3.Vec, mode SweepMode, closed bool, twist float64) (SDF3, error) {
	if sdf == nil {
		return nil, ErrMsg("sdf == nil")
	}
	if mode != SweepPolyline && mode != SweepSmooth {
		return nil, ErrMsg("invalid sweep mode")
	}
	if len(path) < 2 {
		return nil, ErrMsg("path needs at least 2 points")
	}
	// copy and validate the control points
	pts := make([]v3.Vec, len(path))
	copy(pts, path)
	for _, p := range pts {
		if math.IsNaN(p.X) || math.IsNaN(p.Y) || math.IsNaN(p.Z) ||
			math.IsInf(p.X, 0) || math.IsInf(p.Y, 0) || math.IsInf(p.Z, 0) {
			return nil, ErrMsg("path has non-finite points")
		}
	}
	// work out a length scale for degeneracy tests
	pbb := Box3{pts[0], pts[0]}
	for _, p := range pts {
		pbb = pbb.Include(p)
	}
	thresh := 1e-9 * (1 + pbb.Size().Length())
	if closed {
		// drop trailing points duplicating the first point
		for len(pts) > 1 && pts[0].Sub(pts[len(pts)-1]).Length() < thresh {
			pts = pts[:len(pts)-1]
		}
		if len(pts) < 3 {
			return nil, ErrMsg("closed path needs at least 3 distinct points")
		}
	}
	p := sweepPath{
		mode:   mode,
		closed: closed,
		pts:    pts,
		nseg:   len(pts) - 1,
	}
	if closed {
		p.nseg = len(pts)
	}
	// consecutive control points must be distinct
	for i := 0; i < p.nseg; i++ {
		if pts[(i+1)%len(pts)].Sub(pts[i]).Length() < thresh {
			return nil, ErrMsg("degenerate path segment")
		}
	}
	// segment lengths
	p.segLen = make([]float64, p.nseg)
	for i := 0; i < p.nseg; i++ {
		if mode == SweepPolyline {
			p.segLen[i] = pts[(i+1)%len(pts)].Sub(pts[i]).Length()
		} else {
			// approximate the arc length by sub-sampling
			l := 0.0
			prev := p.pos(i, 0)
			for j := 1; j <= sweepArcLenSteps; j++ {
				q := p.pos(i, float64(j)/sweepArcLenSteps)
				l += q.Sub(prev).Length()
				prev = q
			}
			p.segLen[i] = l
		}
	}
	// cumulative lengths
	p.cum = make([]float64, p.nseg+1)
	for i := 0; i < p.nseg; i++ {
		p.cum[i+1] = p.cum[i] + p.segLen[i]
	}
	p.total = p.cum[p.nseg]
	// segment bounding spheres
	p.segC = make([]v3.Vec, p.nseg)
	p.segR = make([]float64, p.nseg)
	for i := 0; i < p.nseg; i++ {
		if mode == SweepPolyline {
			p0 := pts[i]
			p1 := pts[(i+1)%len(pts)]
			p.segC[i] = p0.Add(p1).MulScalar(0.5)
			p.segR[i] = 0.5 * p.segLen[i]
		} else {
			p0, p1, p2, p3 := p.crPoints(i)
			c := p0.Min(p1.Min(p2.Min(p3))).Add(p0.Max(p1.Max(p2.Max(p3)))).MulScalar(0.5)
			r := math.Max(math.Max(p0.Sub(c).Length(), p1.Sub(c).Length()),
				math.Max(p2.Sub(c).Length(), p3.Sub(c).Length()))
			p.segC[i] = c
			p.segR[i] = r
		}
	}
	// frame table
	p.buildFrames()
	// end frames for the end caps
	if !closed {
		p.pos0 = p.pos(0, 0)
		p.tn0 = p.tangent(0, 0)
		p.pos1 = p.pos(p.nseg-1, 1)
		p.tn1 = p.tangent(p.nseg-1, 1)
	}
	// bounding box: path centerline enlarged by the cross-section radius
	r := 0.0
	for _, v := range sdf.BoundingBox().Vertices() {
		r = math.Max(r, v.Length())
	}
	s := SweepSDF3{
		sdf:   sdf,
		path:  &p,
		twist: twist,
		// note: Enlarge takes a size vector (half is applied to each side)
		bb: p.boundingBox().Enlarge(v3.Vec{2 * r, 2 * r, 2 * r}),
	}
	return &s, nil
}

// Evaluate returns the minimum distance to the swept solid.
func (s *SweepSDF3) Evaluate(p v3.Vec) float64 {
	seg, t := s.path.closest(p)
	pos, tn, n, b := s.path.frame(seg, t)
	if s.twist != 0 {
		a := s.twist * s.path.u(seg, t)
		n, b = rotateAbout(n, tn, a), rotateAbout(b, tn, a)
	}
	q := p.Sub(pos)
	// distance to the cross-section in the frame plane
	a := s.sdf.Evaluate(v2.Vec{q.Dot(n), q.Dot(b)})
	if s.path.closed {
		return a
	}
	// Intersect with the end cap half-spaces (the planes through the end
	// points perpendicular to the end tangents). The max() combination is
	// the standard CSG intersection: exact inside the solid and a lower
	// bound on the true distance outside, which keeps the adaptive
	// renderers' culling valid.
	return math.Max(a, math.Max(
		s.path.pos0.Sub(p).Dot(s.path.tn0),
		p.Sub(s.path.pos1).Dot(s.path.tn1)))
}

// BoundingBox returns the bounding box of the swept solid.
func (s *SweepSDF3) BoundingBox() Box3 {
	return s.bb
}

//-----------------------------------------------------------------------------
