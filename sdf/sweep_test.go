//-----------------------------------------------------------------------------
/*

Sweep Testing

*/
//-----------------------------------------------------------------------------

package sdf

import (
	"math"
	"testing"

	v2 "github.com/deadsy/sdfx/vec/v2"
	v3 "github.com/deadsy/sdfx/vec/v3"
)

//-----------------------------------------------------------------------------

func Test_Sweep3D_errors(t *testing.T) {
	circle, _ := Circle2D(1)
	good := []v3.Vec{{0, 0, 0}, {0, 0, 10}}
	testSet := []struct {
		sdf    SDF2
		path   []v3.Vec
		mode   SweepMode
		closed bool
	}{
		{nil, good, SweepPolyline, false},                                                   // nil sdf
		{circle, nil, SweepPolyline, false},                                                 // nil path
		{circle, []v3.Vec{{0, 0, 0}}, SweepPolyline, false},                                 // too few points
		{circle, good, SweepMode(42), false},                                                // bad mode
		{circle, []v3.Vec{{0, 0, 0}, {0, 0, 0}}, SweepPolyline, false},                      // degenerate segment
		{circle, []v3.Vec{{0, 0, 0}, {1, 0, 0}, {1, 0, 0}}, SweepSmooth, false},             // degenerate segment
		{circle, []v3.Vec{{0, 0, 0}, {1, 0, 0}}, SweepPolyline, true},                       // closed needs 3 points
		{circle, []v3.Vec{{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {0, 0, 0}}, SweepPolyline, true}, // all duplicates
		{circle, []v3.Vec{{0, 0, 0}, {math.NaN(), 0, 0}}, SweepPolyline, false},             // NaN
		{circle, []v3.Vec{{0, 0, 0}, {0, math.Inf(1), 0}}, SweepSmooth, false},              // Inf
	}
	for i, test := range testSet {
		s, err := Sweep3D(test.sdf, test.path, test.mode, test.closed, 0)
		if err == nil {
			t.Errorf("test %d: expected an error", i)
		}
		if s != nil {
			t.Errorf("test %d: expected a nil sdf", i)
		}
	}
	// a repeated first/last point on a closed path is ok (dropped)
	if _, err := Sweep3D(circle, []v3.Vec{{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}}, SweepSmooth, true, 0); err != nil {
		t.Errorf("closed path with repeated end point: %s", err)
	}
}

//-----------------------------------------------------------------------------

// Sweeping a circle along a straight line should give a cylinder.
func Test_Sweep3D_straight(t *testing.T) {
	circle, _ := Circle2D(2)
	for _, mode := range []SweepMode{SweepPolyline, SweepSmooth} {
		s, err := Sweep3D(circle, []v3.Vec{{0, 0, -10}, {0, 0, 10}}, mode, false, 0)
		if err != nil {
			t.Fatalf("mode %d: %s", mode, err)
		}
		testSet := []struct {
			p v3.Vec
			d float64
		}{
			{v3.Vec{0, 0, 0}, -2},       // on the path
			{v3.Vec{1, 0, 5}, -1},       // inside
			{v3.Vec{0, -1.5, -5}, -0.5}, // inside
			{v3.Vec{3, 0, 0}, 1},        // outside, radial
			{v3.Vec{0, 0, 12}, 2},       // outside, beyond the end cap
			{v3.Vec{0, 0, -11}, 1},      // outside, beyond the start cap
			{v3.Vec{0, 4, 12}, 2},       // outside, radial + beyond (max of the two)
			{v3.Vec{1, 0, 10.5}, 0.5},   // inside, beyond the end
		}
		for _, test := range testSet {
			d := s.Evaluate(test.p)
			if !EqualFloat64(d, test.d, 1e-6) {
				t.Errorf("mode %d p %v: expected %f, got %f", mode, test.p, test.d, d)
			}
		}
		// bounding box: path box enlarged by the cross-section bounding radius
		r := 2 * math.Sqrt2 // bounding radius of the circle's 2d bounding box
		bb := s.BoundingBox()
		expected := Box3{v3.Vec{-r, -r, -10 - r}, v3.Vec{r, r, 10 + r}}
		if !bb.Equals(expected, 1e-6) {
			t.Errorf("mode %d: expected bb %v, got %v", mode, expected, bb)
		}
	}
}

//-----------------------------------------------------------------------------

// Sweeping a circle around a closed circular path should give a seamless torus.
func Test_Sweep3D_closed_seamless(t *testing.T) {
	circle, _ := Circle2D(1)
	// circular path of radius 10 in the xy-plane
	var path []v3.Vec
	for i := 0; i < 12; i++ {
		theta := Tau * float64(i) / 12
		path = append(path, v3.Vec{10 * math.Cos(theta), 10 * math.Sin(theta), 0})
	}
	// vertex angles (exact centerline positions for the polyline path)
	vertex := make([]float64, 12)
	for i := range vertex {
		vertex[i] = Tau * float64(i) / 12
	}
	testSet := []struct {
		mode  SweepMode
		theta []float64
		hole  float64 // expected distance at the origin
	}{
		// for the polyline path the circle centerline only passes through
		// the control points, so test at the vertex angles. The hole is
		// bounded by the chord midpoints (10*cos(15°) from the origin).
		{SweepPolyline, vertex, 10*math.Cos(Pi/12) - 1},
		// the smooth path follows the circle everywhere, straddle the
		// u = 0/1 seam (theta = 0) to check for seamlessness
		{SweepSmooth, []float64{0, 0.01, 0.3, 1.1, 2.4, 3.9, 5.1, Tau - 0.01}, 9},
	}
	for _, test := range testSet {
		s, err := Sweep3D(circle, path, test.mode, true, 0)
		if err != nil {
			t.Fatalf("mode %d: %s", test.mode, err)
		}
		// check points on the centerline all the way around the loop
		for _, theta := range test.theta {
			c, sn := math.Cos(theta), math.Sin(theta)
			// on the centerline
			if d := s.Evaluate(v3.Vec{10 * c, 10 * sn, 0}); d > -0.9 {
				t.Errorf("mode %d theta %f: centerline d %f, expected ~-1", test.mode, theta, d)
			}
			// on the surface
			if d := s.Evaluate(v3.Vec{11 * c, 11 * sn, 0}); math.Abs(d) > 0.1 {
				t.Errorf("mode %d theta %f: surface d %f, expected ~0", test.mode, theta, d)
			}
			// well outside
			if d := s.Evaluate(v3.Vec{13 * c, 13 * sn, 0}); math.Abs(d-2) > 0.1 {
				t.Errorf("mode %d theta %f: outside d %f, expected ~2", test.mode, theta, d)
			}
			// inside the hole
			if d := s.Evaluate(v3.Vec{0, 0, 0}); math.Abs(d-test.hole) > 0.1 {
				t.Errorf("mode %d: hole d %f, expected ~%f", test.mode, d, test.hole)
			}
		}
	}
}

//-----------------------------------------------------------------------------

// A uniform twist should rotate the cross-section along the path.
func Test_Sweep3D_twist(t *testing.T) {
	// 2 x 1 rectangle (x half-size 1, y half-size 0.5)
	rect := Box2D(v2.Vec{2, 1}, 0)
	// sweep along z in [-5, 5] with a 90 degree twist
	s, err := Sweep3D(rect, []v3.Vec{{0, 0, -5}, {0, 0, 5}}, SweepPolyline, false, Pi/2)
	if err != nil {
		t.Fatalf("%s", err)
	}
	// at z = 4 (u = 0.9) the twist angle is 0.9 * pi/2 = 81 degrees
	z := 4.0
	a := 0.9 * Pi / 2
	c, sn := math.Cos(a), math.Sin(a)
	// a point at 2D (0.8, 0) in the rotated frame should be inside
	pIn := v3.Vec{0.8 * c, 0.8 * sn, z}
	if d := s.Evaluate(pIn); d >= 0 {
		t.Errorf("rotated point %v: d %f, expected < 0", pIn, d)
	}
	// the same point in the unrotated frame should be outside (y extent is only 0.5)
	pOut := v3.Vec{0.8, 0, z}
	if d := s.Evaluate(pOut); d <= 0 {
		t.Errorf("unrotated point %v: d %f, expected > 0", pOut, d)
	}
	// at z = -4 (u = 0.1) the twist is only 9 degrees
	p := v3.Vec{0.8, 0, -4}
	if d := s.Evaluate(p); d >= 0 {
		t.Errorf("start point %v: d %f, expected < 0", p, d)
	}
}

//-----------------------------------------------------------------------------

// Sweeps should compose with the standard CSG operations.
func Test_Sweep3D_csg(t *testing.T) {
	outer, _ := Circle2D(3)
	inner, _ := Circle2D(2)
	path := []v3.Vec{{-10, 0, 0}, {-10, 10, 0}, {10, 10, 0}}
	o, err := Sweep3D(outer, path, SweepSmooth, false, 0)
	if err != nil {
		t.Fatalf("%s", err)
	}
	i, err := Sweep3D(inner, path, SweepSmooth, false, 0)
	if err != nil {
		t.Fatalf("%s", err)
	}
	// a hollow bent pipe
	pipe := Difference3D(o, i)
	// in the wall (radius 2.5 from the centerline)
	if d := pipe.Evaluate(v3.Vec{-10, 5, 2.5}); d >= 0 {
		t.Errorf("pipe wall: d %f, expected < 0", d)
	}
	// in the hollow bore
	if d := pipe.Evaluate(v3.Vec{-10, 5, 0}); d <= 0 {
		t.Errorf("pipe bore: d %f, expected > 0", d)
	}
	// outside the pipe
	if d := pipe.Evaluate(v3.Vec{-10, 5, 4}); d <= 0 {
		t.Errorf("outside pipe: d %f, expected > 0", d)
	}
	// union with another object
	boss, _ := Box3D(v3.Vec{4, 4, 4}, 0)
	s := Union3D(pipe, Transform3D(boss, Translate3d(v3.Vec{-10, 0, 0})))
	if d := s.Evaluate(v3.Vec{-10, 0, 1}); d >= 0 {
		t.Errorf("union: d %f, expected < 0", d)
	}
}

//-----------------------------------------------------------------------------

func Test_Sweep3D_boundingbox(t *testing.T) {
	circle, _ := Circle2D(1)
	path := []v3.Vec{{0, 0, 0}, {10, 0, 0}, {10, 10, 0}}
	s, err := Sweep3D(circle, path, SweepPolyline, false, 0)
	if err != nil {
		t.Fatalf("%s", err)
	}
	bb := s.BoundingBox()
	// the surface should be inside the bounding box
	for _, p := range []v3.Vec{{5, 1, 0}, {10, 5, 1}, {0, -1, 0}} {
		if !bb.Contains(p) {
			t.Errorf("bb %v does not contain surface point %v", bb, p)
		}
	}
	// the bounding box corners should be outside the solid
	for _, p := range bb.Vertices() {
		if d := s.Evaluate(p); d < 0 {
			t.Errorf("bb corner %v is inside the solid (d %f)", p, d)
		}
	}
	// no NaN evaluations over the bounding box
	for i := 0; i < 1000; i++ {
		if d := s.Evaluate(bb.Random()); math.IsNaN(d) || math.IsInf(d, 0) {
			t.Fatalf("non-finite evaluation %f", d)
		}
	}
}

//-----------------------------------------------------------------------------
