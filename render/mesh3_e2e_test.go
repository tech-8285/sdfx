package render

import (
	"testing"

	"github.com/deadsy/sdfx/sdf"
	v3 "github.com/deadsy/sdfx/vec/v3"
)

// e2eBoxMesh builds an outward-wound box triangle mesh.
func e2eBoxMesh(min, max v3.Vec) []*sdf.Triangle3 {
	x0, y0, z0 := min.X, min.Y, min.Z
	x1, y1, z1 := max.X, max.Y, max.Z
	v := [8]v3.Vec{
		{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0},
		{x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1},
	}
	return []*sdf.Triangle3{
		{v[0], v[1], v[2]}, {v[0], v[2], v[3]},
		{v[4], v[7], v[6]}, {v[4], v[6], v[5]},
		{v[0], v[4], v[5]}, {v[0], v[5], v[1]},
		{v[2], v[6], v[7]}, {v[2], v[7], v[3]},
		{v[0], v[3], v[7]}, {v[0], v[7], v[4]},
		{v[1], v[5], v[6]}, {v[1], v[6], v[2]},
	}
}

// Test_MeshSDF3_EndToEnd verifies a closed triangle mesh converts to an SDF3
// whose offset and shell variants render to watertight geometry, and that an
// open mesh is rejected rather than silently treated as a solid.
func Test_MeshSDF3_EndToEnd(t *testing.T) {
	mesh := e2eBoxMesh(v3.Vec{-10, -10, -10}, v3.Vec{10, 10, 10})
	s, err := sdf.Mesh3D(mesh)
	if err != nil {
		t.Fatalf("closed mesh should validate: %v", err)
	}
	r := NewMarchingCubesUniform(48)

	// a small offset moves the zero surface off the exact input faces;
	// the rendered result must be a closed solid
	for name, model := range map[string]sdf.SDF3{
		"grown":  sdf.Offset3D(s, 2),
		"shrunk": sdf.Offset3D(s, -2),
	} {
		tris := CollectTriangles(model, r)
		if !IsWatertight(tris) {
			t.Errorf("%s: rendered mesh is not watertight (%d tris, %d boundary edges)",
				name, len(tris), CountBoundaryEdges(tris))
		}
	}

	shell, err := sdf.Shell3D(s, 2)
	if err != nil {
		t.Fatalf("Shell3D error: %v", err)
	}
	if tris := CollectTriangles(shell, r); !IsWatertight(tris) {
		t.Errorf("shell: rendered mesh is not watertight (%d tris, %d boundary edges)",
			len(tris), CountBoundaryEdges(tris))
	}

	if _, err := sdf.Mesh3D(mesh[:len(mesh)-1]); err == nil {
		t.Error("open mesh should be rejected rather than treated as a solid")
	}
}
