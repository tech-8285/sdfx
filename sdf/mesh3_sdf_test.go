//-----------------------------------------------------------------------------
/*

Mesh 3D Testing

*/
//-----------------------------------------------------------------------------

package sdf

import (
	"math"
	"testing"

	v3 "github.com/deadsy/sdfx/vec/v3"
)

//-----------------------------------------------------------------------------

// boxMesh returns the 12 triangles of an axis-aligned box from min to max,
// wound with outward normals.
func boxMesh(min, max v3.Vec) []*Triangle3 {
	x0, y0, z0 := min.X, min.Y, min.Z
	x1, y1, z1 := max.X, max.Y, max.Z
	v := [8]v3.Vec{
		{x0, y0, z0}, // 0
		{x1, y0, z0}, // 1
		{x1, y1, z0}, // 2
		{x0, y1, z0}, // 3
		{x0, y0, z1}, // 4
		{x1, y0, z1}, // 5
		{x1, y1, z1}, // 6
		{x0, y1, z1}, // 7
	}
	return []*Triangle3{
		// z- (normal -z)
		{v[0], v[1], v[2]}, {v[0], v[2], v[3]},
		// z+ (normal +z)
		{v[4], v[7], v[6]}, {v[4], v[6], v[5]},
		// y- (normal -y)
		{v[0], v[4], v[5]}, {v[0], v[5], v[1]},
		// y+ (normal +y)
		{v[2], v[6], v[7]}, {v[2], v[7], v[3]},
		// x- (normal -x)
		{v[0], v[3], v[7]}, {v[0], v[7], v[4]},
		// x+ (normal +x)
		{v[1], v[5], v[6]}, {v[1], v[6], v[2]},
	}
}

// reverseWinding returns a copy of the mesh with flipped triangle winding.
func reverseWinding(mesh []*Triangle3) []*Triangle3 {
	out := make([]*Triangle3, len(mesh))
	for i, t := range mesh {
		rt := *t
		rt[1], rt[2] = rt[2], rt[1]
		out[i] = &rt
	}
	return out
}

//-----------------------------------------------------------------------------

func Test_Mesh3_SignedDistanceBox(t *testing.T) {
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	s, err := Mesh3D(mesh)
	if err != nil {
		t.Fatalf("Mesh3D returned an error for a closed box: %v", err)
	}

	testSet := []struct {
		p    v3.Vec
		want float64
	}{
		{v3.Vec{2, 0, 0}, 1},             // outside along x
		{v3.Vec{0, -3, 0}, 2},            // outside along y
		{v3.Vec{0, 0, 5}, 4},             // outside along z
		{v3.Vec{1, 1, 1}, 0},             // corner on the surface
		{v3.Vec{3, 4, 5}, math.Sqrt(29)}, // outside corner region
		{v3.Vec{0, 0, 0}, -1},            // center: -min half-size
		{v3.Vec{0.5, 0, 0}, -0.5},        // inside
		{v3.Vec{-0.9, 0.2, 0.2}, -0.1},
	}

	for _, test := range testSet {
		got := s.Evaluate(test.p)
		if !EqualFloat64(got, test.want, 1e-6) {
			t.Errorf("Evaluate(%v) = %f, want %f", test.p, got, test.want)
		}
	}
}

func Test_Mesh3_SlowMatchesFast(t *testing.T) {
	mesh := boxMesh(v3.Vec{-2, -1, -0.5}, v3.Vec{2, 1, 0.5})
	fast, err := Mesh3D(mesh)
	if err != nil {
		t.Fatalf("Mesh3D error: %v", err)
	}
	slow, err := Mesh3DSlow(mesh)
	if err != nil {
		t.Fatalf("Mesh3DSlow error: %v", err)
	}

	b := NewBox3(v3.Vec{0, 0, 0}, v3.Vec{8, 8, 8})
	for i := 0; i < 2000; i++ {
		p := b.Random().Sub(v3.Vec{4, 4, 4})
		df := fast.Evaluate(p)
		ds := slow.Evaluate(p)
		if !EqualFloat64(df, ds, 1e-6) {
			t.Errorf("point %v: fast %f != slow %f", p, df, ds)
		}
		// the sign must agree, which is the inside/outside classification
		if math.Signbit(df) != math.Signbit(ds) {
			t.Errorf("point %v: sign mismatch fast %f slow %f", p, df, ds)
		}
	}
}

func Test_Mesh3_OrientationNormalized(t *testing.T) {
	// an inward-wound box is still valid: the constructor reorients it
	mesh := reverseWinding(boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1}))
	s, err := Mesh3D(mesh)
	if err != nil {
		t.Fatalf("Mesh3D error: %v", err)
	}
	if got := s.Evaluate(v3.Vec{0, 0, 0}); got >= 0 {
		t.Errorf("center of reoriented box should be inside, got %f", got)
	}
	if got := s.Evaluate(v3.Vec{5, 0, 0}); got <= 0 {
		t.Errorf("outside of reoriented box should be positive, got %f", got)
	}

	// the exposed mesh carries the corrected outward winding
	ms := s.(*MeshSDF3)
	if signedMeshVolume(ms.Mesh()) <= 0 {
		t.Errorf("exposed mesh should have positive signed volume")
	}
}

func Test_Mesh3_OffsetAndShell(t *testing.T) {
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	s, err := Mesh3D(mesh)
	if err != nil {
		t.Fatalf("Mesh3D error: %v", err)
	}

	// positive offset grows the solid: the surface moves outward by 0.5
	grown := Offset3D(s, 0.5)
	if got := grown.Evaluate(v3.Vec{1.2, 0, 0}); got >= 0 {
		t.Errorf("point inside grown solid should be negative, got %f", got)
	}
	// raw box distance at 1.6 is 0.6; offset field subtracts 0.5
	if got := grown.Evaluate(v3.Vec{1.6, 0, 0}); !EqualFloat64(got, 0.1, 1e-6) {
		t.Errorf("grown offset distance wrong: got %f", got)
	}
	// the grown surface sits at x = 1.5
	if got := grown.Evaluate(v3.Vec{1.5, 0, 0}); !EqualFloat64(got, 0, 1e-6) {
		t.Errorf("grown surface should sit at 1.5, got %f", got)
	}

	// negative offset shrinks the solid while remaining a coherent SDF
	shrunk := Offset3D(s, -0.5)
	// raw center distance is -1; the offset field subtracts -0.5
	if got := shrunk.Evaluate(v3.Vec{0, 0, 0}); !EqualFloat64(got, -0.5, 1e-6) {
		t.Errorf("shrunk center distance wrong: got %f", got)
	}
	// the shrunk surface is at x = 0.5, so 0.4 remains inside
	if got := shrunk.Evaluate(v3.Vec{0.4, 0, 0}); got >= 0 {
		t.Errorf("point should remain inside shrunk solid, got %f", got)
	}

	// shell is a thin coherent layer around the original surface
	shell, err := Shell3D(s, 0.2)
	if err != nil {
		t.Fatalf("Shell3D error: %v", err)
	}
	// deep inside the original solid: outside the shell (positive)
	if got := shell.Evaluate(v3.Vec{0, 0, 0}); got <= 0 {
		t.Errorf("shell should be empty deep inside the solid, got %f", got)
	}
	// near the original surface: inside the shell (negative)
	if got := shell.Evaluate(v3.Vec{1.02, 0, 0}); got >= 0 {
		t.Errorf("near-surface point should be inside the shell, got %f", got)
	}
}

func Test_Mesh3_MeshAttributesPreserved(t *testing.T) {
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	s, err := Mesh3D(mesh)
	if err != nil {
		t.Fatalf("Mesh3D error: %v", err)
	}
	ms := s.(*MeshSDF3)
	if len(ms.Mesh()) != len(mesh) {
		t.Errorf("triangle count changed: got %d want %d", len(ms.Mesh()), len(mesh))
	}
	if len(ms.Mesh()) != len(ms.info) {
		t.Errorf("info/mesh size mismatch")
	}
}

func Test_Mesh3_Tetrahedron(t *testing.T) {
	// regular tetrahedron around the origin
	v := [4]v3.Vec{
		{1, 1, 1},
		{1, -1, -1},
		{-1, 1, -1},
		{-1, -1, 1},
	}
	mesh := []*Triangle3{
		{v[0], v[2], v[1]},
		{v[0], v[1], v[3]},
		{v[0], v[3], v[2]},
		{v[1], v[2], v[3]},
	}
	s, err := Mesh3D(mesh)
	if err != nil {
		t.Fatalf("Mesh3D error: %v", err)
	}
	if got := s.Evaluate(v3.Vec{0, 0, 0}); got >= 0 {
		t.Errorf("tetrahedron center should be inside, got %f", got)
	}
	if got := s.Evaluate(v3.Vec{5, 5, 5}); got <= 0 {
		t.Errorf("far point should be outside, got %f", got)
	}
}

func Test_Mesh3_OutsidePointsAlwaysPositive(t *testing.T) {
	// points well outside the bounding box must always classify as outside
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	s, _ := Mesh3D(mesh)
	outside := []v3.Vec{
		{-5, 0, 0}, {5, 0, 0}, {0, 5, 0}, {0, -5, 0},
		{0, 0, 5}, {0, 0, -5}, {5, 5, 5}, {-5, -5, -5},
		{2, -3, 0.7}, {-2.5, 1.5, 3.1},
	}
	for _, p := range outside {
		if got := s.Evaluate(p); got <= 0 {
			t.Errorf("outside point %v should be positive, got %f", p, got)
		}
	}
}

func Test_Mesh3_FarPointsMatchSlow(t *testing.T) {
	// points far outside the mesh exercise both the growing-box distance
	// fallback and the ray candidate coverage
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	fast, _ := Mesh3D(mesh)
	slow, _ := Mesh3DSlow(mesh)
	far := []v3.Vec{
		{100, 0, 0}, {-100, 50, 25}, {0, 1000, 0},
		{50, -75, 125}, {-1000, -1000, -1000},
	}
	for _, p := range far {
		df := fast.Evaluate(p)
		ds := slow.Evaluate(p)
		if !EqualFloat64(df, ds, 1e-6) || df <= 0 {
			t.Errorf("far point %v: fast=%f slow=%f", p, df, ds)
		}
	}
}

func Test_Mesh3_SurfaceZeroCrossing(t *testing.T) {
	// the signed field must cross zero at the box faces
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	s, _ := Mesh3D(mesh)
	if s.Evaluate(v3.Vec{0.99, 0, 0}) >= 0 {
		t.Error("just inside the +x face should be negative")
	}
	if s.Evaluate(v3.Vec{1.01, 0, 0}) <= 0 {
		t.Error("just outside the +x face should be positive")
	}
}

func Test_Mesh3_NormalDirectionMatchesTopology(t *testing.T) {
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	s, _ := Mesh3D(mesh)
	ms := s.(*MeshSDF3)
	// every validated triangle normal must point toward the exterior: a
	// point offset from the face center along the normal is outside
	for _, tri := range ms.Mesh() {
		center := tri[0].Add(tri[1]).Add(tri[2]).MulScalar(1.0 / 3.0)
		outside := center.Add(tri.Normal().MulScalar(0.2))
		if s.Evaluate(outside) <= 0 {
			t.Errorf("normal of %v points inward", *tri)
		}
	}
}

//-----------------------------------------------------------------------------
