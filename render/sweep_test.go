//-----------------------------------------------------------------------------
/*

Sweep Rendering Testing

*/
//-----------------------------------------------------------------------------

package render

import (
	"math"
	"testing"

	"github.com/deadsy/sdfx/sdf"
	v3 "github.com/deadsy/sdfx/vec/v3"
)

//-----------------------------------------------------------------------------

// countOpenEdges returns the number of mesh edges not shared by exactly 2 triangles.
func countOpenEdges(mesh []*sdf.Triangle3) int {
	type edge struct{ a, b v3.Vec }
	// quantize a vertex so shared corners compare equal
	key := func(p v3.Vec) v3.Vec {
		const k = 1e6
		return v3.Vec{math.Round(p.X * k), math.Round(p.Y * k), math.Round(p.Z * k)}
	}
	less := func(a, b v3.Vec) bool {
		if a.X != b.X {
			return a.X < b.X
		}
		if a.Y != b.Y {
			return a.Y < b.Y
		}
		return a.Z < b.Z
	}
	count := make(map[edge]int)
	for _, t := range mesh {
		for i := 0; i < 3; i++ {
			a, b := key(t[i]), key(t[(i+1)%3])
			if a == b {
				continue
			}
			// undirected edge
			if less(b, a) {
				a, b = b, a
			}
			count[edge{a, b}]++
		}
	}
	open := 0
	for _, n := range count {
		if n != 2 {
			open++
		}
	}
	return open
}

// sweepSeal returns a closed-loop sweep (a wavy seal ring).
func sweepSeal(t *testing.T) sdf.SDF3 {
	circle, err := sdf.Circle2D(1)
	if err != nil {
		t.Fatalf("%s", err)
	}
	var path []v3.Vec
	for i := 0; i < 24; i++ {
		theta := sdf.Tau * float64(i) / 24
		path = append(path, v3.Vec{
			10 * math.Cos(theta),
			10 * math.Sin(theta),
			math.Sin(3 * theta),
		})
	}
	s, err := sdf.Sweep3D(circle, path, sdf.SweepSmooth, true, 0)
	if err != nil {
		t.Fatalf("%s", err)
	}
	return s
}

// sweepPipe returns a capped open sweep (a bent pipe with a uniform twist).
func sweepPipe(t *testing.T) sdf.SDF3 {
	circle, err := sdf.Circle2D(1.5)
	if err != nil {
		t.Fatalf("%s", err)
	}
	path := []v3.Vec{{-10, 0, 0}, {-10, 8, 0}, {-4, 14, 0}, {6, 14, 0}, {12, 8, 0}, {12, 0, 0}}
	s, err := sdf.Sweep3D(circle, path, sdf.SweepSmooth, false, sdf.Pi)
	if err != nil {
		t.Fatalf("%s", err)
	}
	return s
}

//-----------------------------------------------------------------------------

func Test_Sweep3D_render(t *testing.T) {
	const cells = 200
	solids := []struct {
		name string
		s    sdf.SDF3
	}{
		{"seal", sweepSeal(t)},
		{"pipe", sweepPipe(t)},
	}
	renderers := []struct {
		name string
		r    Render3
	}{
		{"uniform", NewMarchingCubesUniform(cells)},
		{"octree", NewMarchingCubesOctree(cells)},
	}
	for _, solid := range solids {
		for _, renderer := range renderers {
			mesh := ToTriangles(solid.s, renderer.r)
			if len(mesh) == 0 {
				t.Errorf("%s/%s: empty mesh", solid.name, renderer.name)
				continue
			}
			// all vertices should be finite and near the bounding box
			bb := solid.s.BoundingBox().Enlarge(v3.Vec{1, 1, 1})
			for _, tri := range mesh {
				for _, p := range tri {
					if math.IsNaN(p.X) || math.IsNaN(p.Y) || math.IsNaN(p.Z) {
						t.Fatalf("%s/%s: NaN vertex", solid.name, renderer.name)
					}
					if !bb.Contains(p) {
						t.Fatalf("%s/%s: vertex %v outside bounding box %v", solid.name, renderer.name, p, bb)
					}
				}
			}
			// the mesh should be closed (every edge shared by 2 triangles)
			if n := countOpenEdges(mesh); n != 0 {
				t.Errorf("%s/%s: %d open edges", solid.name, renderer.name, n)
			}
		}
	}
}

//-----------------------------------------------------------------------------
