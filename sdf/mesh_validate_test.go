//-----------------------------------------------------------------------------
/*

Mesh topology validation and triangle intersection testing.

*/
//-----------------------------------------------------------------------------

package sdf

import (
	"testing"

	v3 "github.com/deadsy/sdfx/vec/v3"
)

//-----------------------------------------------------------------------------

func Test_ValidateManifold_ClosedBox(t *testing.T) {
	mesh := boxMesh(v3.Vec{0, 0, 0}, v3.Vec{2, 2, 2})
	clean, err := ValidateManifoldMesh(mesh)
	if err != nil {
		t.Fatalf("closed box should validate: %v", err)
	}
	if len(clean) != len(mesh) {
		t.Errorf("triangle count changed: %d -> %d", len(mesh), len(clean))
	}
	if signedMeshVolume(clean) <= 0 {
		t.Errorf("validated box should have positive signed volume")
	}
}

func Test_ValidateManifold_OpenMeshRejected(t *testing.T) {
	mesh := boxMesh(v3.Vec{0, 0, 0}, v3.Vec{2, 2, 2})
	// drop one triangle: all three of its edges lose their opposing
	// triangle and become boundary edges
	open := append(mesh[:1], mesh[2:]...)
	_, err := ValidateManifoldMesh(open)
	if err == nil {
		t.Fatal("open mesh should be rejected")
	}
	topo, ok := err.(*MeshTopologyError)
	if !ok {
		t.Fatalf("expected *MeshTopologyError, got %T: %v", err, err)
	}
	if topo.BoundaryEdges != 3 {
		t.Errorf("expected 3 boundary edges, got %d", topo.BoundaryEdges)
	}
}

func Test_ValidateManifold_MissingFaceReportsThreeBoundaries(t *testing.T) {
	// removing both triangles of the z- face opens a square hole bounded by
	// its four edges; the shared diagonal between the two triangles
	// disappears too, leaving the four perimeter edges (reported once each
	// over their undirected form)
	mesh := boxMesh(v3.Vec{0, 0, 0}, v3.Vec{2, 2, 2})
	hole := append(mesh[:0], mesh[2:]...)
	_, err := ValidateManifoldMesh(hole)
	if err == nil {
		t.Fatal("mesh with a missing face should be rejected")
	}
	topo := err.(*MeshTopologyError)
	if topo.BoundaryEdges != 4 {
		t.Errorf("expected 4 boundary edges around the square hole, got %d",
			topo.BoundaryEdges)
	}
}

func Test_ValidateManifold_NonManifoldRejected(t *testing.T) {
	mesh := boxMesh(v3.Vec{0, 0, 0}, v3.Vec{2, 2, 2})
	// append an extra triangle sharing an edge of the first face
	extra := *mesh[0]
	mesh = append(mesh, &extra)
	_, err := ValidateManifoldMesh(mesh)
	if err == nil {
		t.Fatal("non-manifold mesh should be rejected")
	}
	if _, ok := err.(*MeshTopologyError); !ok {
		t.Fatalf("expected *MeshTopologyError, got %T", err)
	}
}

func Test_ValidateManifold_SelfIntersectionRejected(t *testing.T) {
	mesh := boxMesh(v3.Vec{-1, -1, -1}, v3.Vec{1, 1, 1})
	// a triangle slicing through the interior of the box, sharing no
	// vertices with the surface
	slice := &Triangle3{
		{-2, 0.5, 0.5},
		{2, 0.5, 0.5},
		{0, 0.5, -2},
	}
	// make a closed second "pocket" by adding the slice twice with reversed
	// winding so the edge manifold check stays closed, while the geometry
	// still self-intersects the box
	rev := *slice
	rev[1], rev[2] = rev[2], rev[1]
	bad := append(mesh, slice, &rev)
	_, err := ValidateManifoldMesh(bad)
	if err == nil {
		t.Fatal("self-intersecting mesh should be rejected")
	}
	topo, ok := err.(*MeshTopologyError)
	if !ok {
		t.Fatalf("expected *MeshTopologyError, got %T", err)
	}
	if topo.IntersectingPairs == 0 {
		t.Errorf("expected intersecting pairs to be reported, got %+v", topo)
	}
}

func Test_ValidateManifold_DegenerateRejected(t *testing.T) {
	_, err := Mesh3D([]*Triangle3{
		{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
		{{0, 0, 0}, {0, 0, 0}, {1, 1, 0}},
	})
	if err == nil {
		t.Fatal("degenerate-only mesh should be rejected")
	}
	if _, ok := err.(*MeshTopologyError); !ok {
		t.Fatalf("expected *MeshTopologyError, got %T", err)
	}
}

func Test_ValidateManifold_EmptyRejected(t *testing.T) {
	if _, err := Mesh3D(nil); err == nil {
		t.Fatal("empty mesh should be rejected")
	}
}

func Test_ValidateManifold_DropsDegenerateAndKeepsSolid(t *testing.T) {
	mesh := boxMesh(v3.Vec{0, 0, 0}, v3.Vec{2, 2, 2})
	// a zero-area triangle mixed in with a valid solid
	degenerate := &Triangle3{{5, 5, 5}, {5, 5, 5}, {5, 5, 5}}
	s, err := Mesh3D(append(mesh, degenerate))
	if err != nil {
		t.Fatalf("valid mesh with stray degenerate triangle should validate: %v", err)
	}
	if len(s.(*MeshSDF3).Mesh()) != len(mesh) {
		t.Errorf("degenerate triangle should have been dropped")
	}
}

//-----------------------------------------------------------------------------

func Test_TriangleIntersect_Crossing(t *testing.T) {
	// two triangles crossing at right angles
	a := &Triangle3{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}}
	b := &Triangle3{{0, -0.5, -1}, {0, -0.5, 1}, {0, 0.8, 0}}
	if !a.intersectTriangle(b) {
		t.Error("crossing triangles should intersect")
	}
}

func Test_TriangleIntersect_Separate(t *testing.T) {
	a := &Triangle3{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}}
	b := &Triangle3{{-1, -1, 3}, {1, -1, 3}, {0, 1, 3}}
	if a.intersectTriangle(b) {
		t.Error("separate triangles should not intersect")
	}
}

func Test_TriangleIntersect_CoplanarOverlap(t *testing.T) {
	a := &Triangle3{{0, 0, 0}, {2, 0, 0}, {0, 2, 0}}
	b := &Triangle3{{1, 1, 0}, {3, 1, 0}, {1, 3, 0}}
	if !a.intersectTriangle(b) {
		t.Error("overlapping coplanar triangles should intersect")
	}
}

func Test_TriangleIntersect_CoplanarSeparate(t *testing.T) {
	a := &Triangle3{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	b := &Triangle3{{3, 3, 0}, {4, 3, 0}, {3, 4, 0}}
	if a.intersectTriangle(b) {
		t.Error("disjoint coplanar triangles should not intersect")
	}
}

func Test_TriangleOverlapsBox(t *testing.T) {
	tri := &Triangle3{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}}
	if !triangleOverlapsBox(tri, NewBox3(v3.Vec{0, 0, 0}, v3.Vec{0.5, 0.5, 0.5})) {
		t.Error("triangle crossing the box should overlap it")
	}
	if triangleOverlapsBox(tri, NewBox3(v3.Vec{0, 0, 5}, v3.Vec{1, 1, 1})) {
		t.Error("triangle far below the box should not overlap it")
	}
	if !triangleOverlapsBox(tri, NewBox3(v3.Vec{0, 0, 0}, v3.Vec{4, 4, 4})) {
		t.Error("triangle fully inside the box should overlap it")
	}
}

func Test_RayCrossCount(t *testing.T) {
	// triangle facing +z
	tri := &Triangle3{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}}
	if n := tri.rayCrossCount(v3.Vec{0, 0, -1}, v3.Vec{0, 0, 1}); n != 1 {
		t.Errorf("ray through the triangle should cross once, got %d", n)
	}
	if n := tri.rayCrossCount(v3.Vec{0, 0, 1}, v3.Vec{0, 0, 1}); n != 0 {
		t.Errorf("ray pointing away should not cross, got %d", n)
	}
	if n := tri.rayCrossCount(v3.Vec{5, 5, -1}, v3.Vec{0, 0, 1}); n != 0 {
		t.Errorf("ray missing the triangle should not cross, got %d", n)
	}
}

//-----------------------------------------------------------------------------
