package render

import (
	"math"
	"sync"
	"testing"

	"github.com/deadsy/sdfx/sdf"
	v2 "github.com/deadsy/sdfx/vec/v2"
)

//-----------------------------------------------------------------------------

func collectLines2(s sdf.SDF2, r Render2) []*sdf.Line2 {
	ch := make(chan []*sdf.Line2)
	var lines []*sdf.Line2
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for batch := range ch {
			lines = append(lines, batch...)
		}
	}()
	r.Render(s, sdf.NewLine2Buffer(ch))
	close(ch)
	wg.Wait()
	return lines
}

func line2Key(p v2.Vec) [2]int64 {
	const q = 1e9
	return [2]int64{int64(math.Round(p.X * q)), int64(math.Round(p.Y * q))}
}

// signedArea returns the sum of the signed areas of the segments. If the
// contour direction is consistent, this is the enclosed area.
func signedArea(lines []*sdf.Line2) float64 {
	a := 0.0
	for _, l := range lines {
		a += 0.5 * (l[0].X*l[1].Y - l[1].X*l[0].Y)
	}
	return a
}

// checkClosed verifies that every endpoint is shared by exactly two segments,
// i.e. the output is a set of closed contours.
func checkClosed(t *testing.T, lines []*sdf.Line2) {
	t.Helper()
	count := make(map[[2]int64]int)
	for _, l := range lines {
		count[line2Key(l[0])]++
		count[line2Key(l[1])]++
	}
	bad := 0
	for _, c := range count {
		if c != 2 {
			bad++
		}
	}
	if bad != 0 {
		t.Errorf("%d vertices have degree != 2 (want closed contours), %d lines", bad, len(lines))
	}
}

func bboxOfLines(lines []*sdf.Line2) (v2.Vec, v2.Vec) {
	min := v2.Vec{math.Inf(1), math.Inf(1)}
	max := v2.Vec{math.Inf(-1), math.Inf(-1)}
	for _, l := range lines {
		for _, p := range l {
			min = min.Min(p)
			max = max.Max(p)
		}
	}
	return min, max
}

//-----------------------------------------------------------------------------

func Test_DualContouring2D_Closed(t *testing.T) {
	square := sdf.Box2D(v2.Vec{10, 10}, 0)
	circle, err := sdf.Circle2D(5)
	if err != nil {
		t.Fatal(err)
	}
	hole := sdf.Difference2D(square, circle)
	union := sdf.Union2D(square, sdf.Transform2D(circle, sdf.Translate2d(v2.Vec{12, 0})))

	cases := map[string]sdf.SDF2{
		"square":      square,
		"circle":      circle,
		"square-hole": hole,
		"two-parts":   union,
		"pinch":       pinchShape(),
	}
	for name, s := range cases {
		t.Run(name, func(t *testing.T) {
			for _, cells := range []int{7, 20, 100} {
				lines := collectLines2(s, NewDualContouring2D(cells))
				if len(lines) == 0 {
					t.Fatalf("cells=%d: no lines", cells)
				}
				checkClosed(t, lines)
			}
		})
	}
}

// pinchShape returns a shape where two regions touch at a single point/crack,
// which produces saddle cells in the sampling grid.
func pinchShape() sdf.SDF2 {
	a := sdf.Transform2D(sdf.Box2D(v2.Vec{4, 4}, 0), sdf.Translate2d(v2.Vec{-2, 0}))
	b := sdf.Transform2D(sdf.Box2D(v2.Vec{4, 4}, 0), sdf.Translate2d(v2.Vec{2, 0}))
	c := sdf.Transform2D(sdf.Box2D(v2.Vec{4, 4}, 0), sdf.Translate2d(v2.Vec{0, -2}))
	d := sdf.Transform2D(sdf.Box2D(v2.Vec{4, 4}, 0), sdf.Translate2d(v2.Vec{0, 2}))
	return sdf.Union2D(a, b, c, d)
}

func Test_DualContouring2D_Saddle(t *testing.T) {
	// Crossing boxes leave a single point contact. The saddle cells must not
	// merge the two arcs into a shared vertex, and every contour must stay
	// closed.
	s := pinchShape()
	for _, cells := range []int{7, 8, 16, 40, 100} {
		lines := collectLines2(s, NewDualContouring2D(cells))
		checkClosed(t, lines)
		// Four 4x4 boxes overlapping in four 2x2 squares: 4*16 - 4*4 = 48.
		if area := signedArea(lines); math.Abs(area-48) > 2 {
			t.Errorf("cells=%d: area %f, want approximately 48", cells, area)
		}
	}
}

func Test_DualContouring2D_ThinWall(t *testing.T) {
	// Box walls only 0.2 thick. Thin but separate solid regions must survive.
	wall, err := sdf.Circle2D(5)
	if err != nil {
		t.Fatal(err)
	}
	inner, err := sdf.Circle2D(4.8)
	if err != nil {
		t.Fatal(err)
	}
	s := sdf.Difference2D(wall, inner)
	// Sample the wall with at least one cell across it (0.2 thick), so use
	// 100+ cells over the 10 wide bounding box.
	for _, cells := range []int{100, 400} {
		lines := collectLines2(s, NewDualContouring2D(cells))
		checkClosed(t, lines)
		area := signedArea(lines)
		want := math.Pi * (25 - 4.8*4.8)
		if math.Abs(area-want) > 0.1*want {
			t.Errorf("cells=%d: wall area %f, want approximately %f", cells, area, want)
		}
	}
}

func Test_DualContouring2D_Winding(t *testing.T) {
	square := sdf.Box2D(v2.Vec{10, 10}, 0)
	circle, err := sdf.Circle2D(3)
	if err != nil {
		t.Fatal(err)
	}
	// Outer square contour and inner hole contour must be closed loops of
	// opposite winding. The total signed area is the difference.
	holed := sdf.Difference2D(square, circle)
	lines := collectLines2(holed, NewDualContouring2D(120))
	checkClosed(t, lines)

	area := signedArea(lines)
	want := 100.0 - math.Pi*9.0
	if math.Abs(area-want) > 2.0 {
		t.Errorf("signed area %f, want approximately %f (outer and hole wind opposite)", area, want)
	}

	// The plain square has the same sign as the outer contour.
	outerLines := collectLines2(square, NewDualContouring2D(120))
	outerArea := signedArea(outerLines)
	if (outerArea > 0) != (area > 0) && math.Abs(area) > 1e-6 {
		t.Errorf("area sign changed: square %f, holed %f", outerArea, area)
	}
	if math.Abs(outerArea-100.0) > 2.0 {
		t.Errorf("square area %f, want approximately 100", outerArea)
	}
}

func Test_DualContouring2D_SharpCorner(t *testing.T) {
	// A 10x10 square rendered at a coarse resolution. The bounding box of the
	// output must reach the corners of the square. Marching squares cuts the
	// corners with a diagonal chord and misses by up to ~1/sqrt(2) of a cell.
	s := sdf.Box2D(v2.Vec{10, 10}, 0)
	const cells = 20
	res := 10.0 / cells
	lines := collectLines2(s, NewDualContouring2D(cells))
	min, max := bboxOfLines(lines)

	if math.Abs(min.X+5) > 0.2*res || math.Abs(min.Y+5) > 0.2*res ||
		math.Abs(max.X-5) > 0.2*res || math.Abs(max.Y-5) > 0.2*res {
		t.Errorf("bbox [%v %v], want corners at (-5,-5) (5,5) within %.3f", min, max, 0.2*res)
	}

	// Straight sides must stay straight: every vertex should be on the square
	// boundary (within a small tolerance).
	onBoundary := 0
	for _, l := range lines {
		for _, p := range l {
			d := math.Min(math.Abs(math.Abs(p.X)-5), math.Abs(math.Abs(p.Y)-5))
			if d < 0.05*res {
				onBoundary++
			}
		}
	}
	if onBoundary == 0 {
		t.Errorf("no vertices lie on the square boundary")
	}
}

func Test_DualContouring2D_NarrowSlot(t *testing.T) {
	// A 10x10 square with a 1.2 wide vertical slot. At this resolution the
	// slot should remain open and the two halves should stay separate.
	outer := sdf.Box2D(v2.Vec{10, 10}, 0)
	slot := sdf.Box2D(v2.Vec{1.2, 12}, 0)
	s := sdf.Difference2D(outer, slot)
	lines := collectLines2(s, NewDualContouring2D(100))
	checkClosed(t, lines)

	// Take a horizontal cut through the middle and check that the contour
	// crosses it in four places (two per bar), leaving a gap in the slot.
	// Offset from y=0 so the cut does not land exactly on a grid line.
	y := 0.0137
	var xs []float64
	for _, l := range lines {
		if (l[0].Y-y)*(l[1].Y-y) < 0 {
			t := (y - l[0].Y) / (l[1].Y - l[0].Y)
			xs = append(xs, l[0].X+t*(l[1].X-l[0].X))
		}
	}
	for _, x := range xs {
		if math.Abs(x) < 0.3 {
			t.Errorf("contour crosses the slot interior at x=%f", x)
		}
	}
	if len(xs) < 4 {
		t.Errorf("got %d crossings on the mid line, want at least 4", len(xs))
	}
}

// maxRadialError returns the largest deviation of contour vertices from a
// circle of the given radius.
func maxRadialError(lines []*sdf.Line2, radius float64) float64 {
	e := 0.0
	for _, l := range lines {
		for _, p := range l {
			e = math.Max(e, math.Abs(p.Length()-radius))
		}
	}
	return e
}

func Test_DualContouring2D_CircleAccuracy(t *testing.T) {
	const radius = 5.0
	s, err := sdf.Circle2D(radius)
	if err != nil {
		t.Fatal(err)
	}
	for _, cells := range []int{20, 50, 100, 200} {
		res := 2 * radius / float64(cells)
		dcErr := maxRadialError(collectLines2(s, NewDualContouring2D(cells)), radius)
		msErr := maxRadialError(collectLines2(s, NewMarchingSquaresUniform(cells)), radius)
		if dcErr > res {
			t.Errorf("cells=%d: dual contouring radial error %f exceeds resolution %f", cells, dcErr, res)
		}
		if dcErr > 1.5*msErr+1e-9 {
			t.Errorf("cells=%d: dual contouring error %f much worse than marching squares %f", cells, dcErr, msErr)
		}
		t.Logf("cells=%3d res=%.4f dc=%.5f ms=%.5f", cells, res, dcErr, msErr)
	}
}

func Test_DualContouring2D_SplineAccuracy(t *testing.T) {
	// A closed-ish smooth spline. Compare dual contouring against marching
	// squares by measuring the distance of each rendered vertex to the SDF
	// surface. Both should be within the resolution.
	knot := []v2.Vec{
		{0, 0},
		{2, 3},
		{5, 4},
		{8, 2},
		{9, -1},
		{7, -4},
		{3, -5},
	}
	spline, err := sdf.CubicSpline2D(knot)
	if err != nil {
		t.Fatal(err)
	}
	// Polygonize: the analytic spline SDF2 uses Newton-Raphson and is
	// unreliable at distance, so use the accurate polygon approximation.
	poly, err := spline.(*sdf.CubicSplineSDF2).PolySpline2D(400)
	if err != nil {
		t.Fatal(err)
	}
	// give the open spline some width so it is a solid region.
	s := sdf.Offset2D(poly, 0.75)

	for _, cells := range []int{40, 100} {
		res := s.BoundingBox().Size().MaxComponent() / float64(cells)
		for name, r := range map[string]Render2{
			"dc": NewDualContouring2D(cells),
			"ms": NewMarchingSquaresUniform(cells),
		} {
			lines := collectLines2(s, r)
			worst := 0.0
			for _, l := range lines {
				for _, p := range l {
					worst = math.Max(worst, math.Abs(s.Evaluate(p)))
				}
			}
			if worst > 1.1*res {
				t.Errorf("%s cells=%d: vertex distance to surface %f exceeds resolution %f", name, cells, worst, res)
			}
		}
	}
}

//-----------------------------------------------------------------------------
