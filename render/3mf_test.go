//-----------------------------------------------------------------------------
/*

3MF Export Testing

These tests cover vertex welding, winding preservation, topology diagnostics
and decode round trips. The key property is that geometrically identical
vertices (after the float32 quantization of the 3MF format) share one stable
index, while merely near-coincident vertices stay distinct, so the export can
never silently create degenerate or non-manifold geometry.

*/
//-----------------------------------------------------------------------------

package render

import (
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/deadsy/sdfx/sdf"
	v3 "github.com/deadsy/sdfx/vec/v3"
	"github.com/hpinc/go3mf"
)

//-----------------------------------------------------------------------------

func tri(a, b, c v3.Vec) *sdf.Triangle3 {
	return &sdf.Triangle3{a, b, c}
}

// decode3MF decodes a written 3MF file and returns its single mesh. Decoding
// without error is what a standard reader requires; the solid-body coherency
// check is applied separately by tests that actually export closed solids.
func decode3MF(t *testing.T, path string) *go3mf.Mesh {
	t.Helper()
	f, err := go3mf.OpenReader(path)
	if err != nil {
		t.Fatalf("open %s: %s", path, err)
	}
	defer f.Close()
	var model go3mf.Model
	if err := f.Decode(&model); err != nil {
		t.Fatalf("decode %s: %s", path, err)
	}
	if len(model.Resources.Objects) != 1 || model.Resources.Objects[0].Mesh == nil {
		t.Fatalf("%s: expected one mesh object", path)
	}
	if len(model.Build.Items) != 1 || model.Build.Items[0].ObjectID != model.Resources.Objects[0].ID {
		t.Fatalf("%s: build item does not reference the mesh object", path)
	}
	mesh := model.Resources.Objects[0].Mesh
	// hard constraints any reader relies on: distinct, in-range indices
	nodeCount := uint32(len(mesh.Vertices.Vertex))
	for i, tri := range mesh.Triangles.Triangle {
		if tri.V1 == tri.V2 || tri.V1 == tri.V3 || tri.V2 == tri.V3 {
			t.Fatalf("%s: triangle %d has repeated indices", path, i)
		}
		if tri.V1 >= nodeCount || tri.V2 >= nodeCount || tri.V3 >= nodeCount {
			t.Fatalf("%s: triangle %d has an out-of-range index", path, i)
		}
	}
	return mesh
}

// edgeCounts returns how many triangles use each undirected edge.
func edgeCounts(m *go3mf.Mesh) map[edgeKey]uint32 {
	counts := make(map[edgeKey]uint32)
	for _, tr := range m.Triangles.Triangle {
		idx := [3]uint32{tr.V1, tr.V2, tr.V3}
		for j := 0; j < 3; j++ {
			counts[newEdgeKey(idx[j], idx[(j+1)%3])]++
		}
	}
	return counts
}

// tetrahedron returns a closed, consistently oriented manifold mesh:
// 4 vertices, 4 triangles, every edge shared by exactly two triangles.
func tetrahedron() []*sdf.Triangle3 {
	a := v3.Vec{0, 0, 0}
	b := v3.Vec{1, 0, 0}
	c := v3.Vec{0, 1, 0}
	d := v3.Vec{0, 0, 1}
	return []*sdf.Triangle3{
		tri(a, c, b), // z = 0 face, outward normal -z
		tri(a, b, d), // y = 0 face, outward normal -y
		tri(a, d, c), // x = 0 face, outward normal -x
		tri(b, c, d), // slanted face, outward normal (+x,+y,+z)
	}
}

//-----------------------------------------------------------------------------

// A clean closed solid must export with no diagnostics, weld every coincident
// vertex into one shared index, and decode as a connected manifold.
func Test_3MF_ManifoldWelding(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifold.3mf")
	report, err := Save3MF(path, tetrahedron())
	if err != nil {
		t.Fatalf("Save3MF: %s", err)
	}
	if !report.OK() {
		t.Fatalf("expected clean report, got:\n%s", report)
	}
	if report.Vertices != 4 || report.WrittenTriangles != 4 {
		t.Fatalf("want 4 vertices/4 triangles, got %d/%d",
			report.Vertices, report.WrittenTriangles)
	}

	mesh := decode3MF(t, path)
	if len(mesh.Vertices.Vertex) != 4 {
		t.Fatalf("want 4 welded vertices, got %d", len(mesh.Vertices.Vertex))
	}
	if len(mesh.Triangles.Triangle) != 4 {
		t.Fatalf("want 4 triangles, got %d", len(mesh.Triangles.Triangle))
	}
	for e, n := range edgeCounts(mesh) {
		if n != 2 {
			t.Fatalf("edge %v used %d times, want 2 (manifold)", e, n)
		}
	}
	// the decoded mesh must pass the library's coherency check
	if err := mesh.ValidateCoherency(); err != nil {
		t.Fatalf("decoded mesh not coherent: %s", err)
	}
}

// Two triangles sharing an edge must reference the same vertex indices, and
// the input winding (normal direction) must be preserved verbatim.
func Test_3MF_SharedVerticesAndWinding(t *testing.T) {
	path := filepath.Join(t.TempDir(), "shared.3mf")
	o := v3.Vec{0, 0, 0}
	p := v3.Vec{1, 0, 0}
	q := v3.Vec{0, 1, 0}
	r := v3.Vec{1, 1, 0}
	// both triangles wind counter-clockwise seen from +z and share edge p-q.
	in := []*sdf.Triangle3{tri(o, p, q), tri(p, r, q)}

	report, err := Save3MF(path, in)
	if err != nil {
		t.Fatalf("Save3MF: %s", err)
	}
	if report.Vertices != 4 {
		t.Fatalf("want 4 welded vertices, got %d", report.Vertices)
	}

	mesh := decode3MF(t, path)
	tr := mesh.Triangles.Triangle
	// the shared edge p-q is triangle 0's V2-V3 and triangle 1's V1-V3;
	// both must resolve to the same pair of welded vertex indices
	edge0 := newEdgeKey(tr[0].V2, tr[0].V3)
	edge1 := newEdgeKey(tr[1].V1, tr[1].V3)
	if edge0 != edge1 {
		t.Fatalf("shared edge not welded: %v and %v", edge0, edge1)
	}
	if len(mesh.Vertices.Vertex) != 4 {
		t.Fatalf("shared vertices were copied instead of welded: got %d",
			len(mesh.Vertices.Vertex))
	}
	// every normal must still point to +z (winding preserved)
	for i, face := range tr {
		a := mesh.Vertices.Vertex[face.V1]
		b := mesh.Vertices.Vertex[face.V2]
		c := mesh.Vertices.Vertex[face.V3]
		normalZ := (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
		if normalZ <= 0 {
			t.Fatalf("triangle %d winding flipped (normal z = %v)", i, normalZ)
		}
	}
}

// Vertices closer than one micron are distinct input vertices and must stay
// distinct: the exporter must not merge on a positional tolerance.
func Test_3MF_NearCoincidentVerticesStayDistinct(t *testing.T) {
	path := filepath.Join(t.TempDir(), "near.3mf")
	// 1e-7 mm is well inside go3mf's old 1 micron welding grid but is a
	// perfectly distinct float32 coordinate.
	near := v3.Vec{1e-7, 0, 0}
	in := []*sdf.Triangle3{
		tri(v3.Vec{0, 0, 0}, v3.Vec{1, 0, 0}, v3.Vec{0, 1, 0}),
		tri(near, v3.Vec{1, 0, 0}, v3.Vec{0, 1, 0}),
	}
	report, err := Save3MF(path, in)
	if err != nil {
		t.Fatalf("Save3MF: %s", err)
	}
	if report.Vertices != 4 {
		t.Fatalf("near-coincident vertices were merged: want 4 vertices, got %d",
			report.Vertices)
	}
	mesh := decode3MF(t, path)
	if len(mesh.Vertices.Vertex) != 4 {
		t.Fatalf("want 4 distinct decoded vertices, got %d",
			len(mesh.Vertices.Vertex))
	}
	if mesh.Vertices.Vertex[0] == mesh.Vertices.Vertex[3] {
		t.Fatal("distinct sub-micron vertices collapsed to one point on disk")
	}
}

// A disconnected "triangle soup" with no shared vertices is still valid 3MF
// and must export and decode unchanged.
func Test_3MF_DisconnectedVerticesExport(t *testing.T) {
	path := filepath.Join(t.TempDir(), "soup.3mf")
	in := []*sdf.Triangle3{
		tri(v3.Vec{0, 0, 0}, v3.Vec{1, 0, 0}, v3.Vec{0, 1, 0}),
		tri(v3.Vec{10, 10, 10}, v3.Vec{11, 10, 10}, v3.Vec{10, 11, 10}),
	}
	report, err := Save3MF(path, in)
	if err != nil {
		t.Fatalf("Save3MF: %s", err)
	}
	if report.Vertices != 6 || report.WrittenTriangles != 2 {
		t.Fatalf("want 6 vertices/2 triangles, got %d/%d",
			report.Vertices, report.WrittenTriangles)
	}
	if len(report.NonManifoldEdges) != 0 {
		t.Fatalf("disconnected soup has no non-manifold edges: %s", report)
	}
	// six boundary edges are expected for an open surface and must not fail
	// the export
	if len(report.BoundaryEdges) != 6 {
		t.Fatalf("want 6 boundary edges, got %d", len(report.BoundaryEdges))
	}
	mesh := decode3MF(t, path)
	if len(mesh.Vertices.Vertex) != 6 || len(mesh.Triangles.Triangle) != 2 {
		t.Fatal("decoded disconnected mesh changed shape")
	}
}

// Degenerate input triangles (identical vertices or non-finite coordinates)
// must be reported and skipped without poisoning the file; the remaining good
// geometry still exports to a readable 3MF.
func Test_3MF_DegenerateTrianglesDiagnosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "degenerate.3mf")
	in := []*sdf.Triangle3{
		tri(v3.Vec{0, 0, 0}, v3.Vec{1, 0, 0}, v3.Vec{0, 1, 0}),
		tri(v3.Vec{2, 0, 0}, v3.Vec{2, 0, 0}, v3.Vec{2, 1, 0}), // repeated vertex
		tri(v3.Vec{math.NaN(), 0, 0}, v3.Vec{3, 0, 0}, v3.Vec{3, 1, 0}),
	}
	report, err := Save3MF(path, in)
	if err != nil {
		t.Fatalf("Save3MF should still succeed with one good triangle: %s", err)
	}
	if len(report.DegenerateTriangles) != 2 {
		t.Fatalf("want 2 degenerate diagnostics, got %d",
			len(report.DegenerateTriangles))
	}
	if report.WrittenTriangles != 1 {
		t.Fatalf("want 1 written triangle, got %d", report.WrittenTriangles)
	}
	if report.UnreferencedVertices == 0 {
		t.Fatal("vertices owned only by skipped triangles should be reported")
	}
	mesh := decode3MF(t, path)
	for _, face := range mesh.Triangles.Triangle {
		if face.V1 == face.V2 || face.V2 == face.V3 || face.V1 == face.V3 {
			t.Fatal("degenerate triangle was written to the file")
		}
	}
}

// Exact duplicate faces are diagnosed and written only once.
func Test_3MF_DuplicateFacesDiagnosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dup.3mf")
	face := tri(v3.Vec{0, 0, 0}, v3.Vec{1, 0, 0}, v3.Vec{0, 1, 0})
	rotated := tri(v3.Vec{1, 0, 0}, v3.Vec{0, 1, 0}, v3.Vec{0, 0, 0}) // cyclic
	in := []*sdf.Triangle3{face, face, rotated}

	report, err := Save3MF(path, in)
	if err != nil {
		t.Fatalf("Save3MF: %s", err)
	}
	if len(report.DuplicateFaces) != 2 {
		t.Fatalf("want 2 duplicate diagnostics, got %d",
			len(report.DuplicateFaces))
	}
	if report.WrittenTriangles != 1 {
		t.Fatalf("duplicate face written more than once: got %d triangles",
			report.WrittenTriangles)
	}
	mesh := decode3MF(t, path)
	if len(mesh.Triangles.Triangle) != 1 || len(mesh.Vertices.Vertex) != 3 {
		t.Fatal("decoded mesh does not match de-duplicated face")
	}
}

// An edge shared by three triangles is a non-manifold connection and must be
// diagnosed, but the triangles are still exported as an explicit triangle soup.
func Test_3MF_NonManifoldEdgeDiagnosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nonmanifold.3mf")
	a := v3.Vec{0, 0, 0}
	b := v3.Vec{1, 0, 0}
	c := v3.Vec{0, 1, 0}
	d := v3.Vec{0, 0, 1}
	e := v3.Vec{-1, -1, 0}
	in := []*sdf.Triangle3{
		tri(a, b, c),
		tri(a, b, d),
		tri(a, b, e), // edge a-b now used by three triangles
	}
	report, err := Save3MF(path, in)
	if err != nil {
		t.Fatalf("Save3MF: %s", err)
	}
	if len(report.NonManifoldEdges) == 0 {
		t.Fatalf("non-manifold edge not detected: %s", report)
	}
	if report.WrittenTriangles != 3 {
		t.Fatal("diagnosed triangles must still be exported")
	}
	// the file remains readable by a standard decoder
	mesh := decode3MF(t, path)
	if len(mesh.Triangles.Triangle) != 3 {
		t.Fatal("non-manifold triangles were silently dropped")
	}
}

// An all-degenerate input cannot be represented at all and must return a hard
// error instead of an empty, unreadable file.
func Test_3MF_NoRepresentableGeometry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.3mf")
	in := []*sdf.Triangle3{
		tri(v3.Vec{0, 0, 0}, v3.Vec{0, 0, 0}, v3.Vec{0, 0, 0}),
	}
	report, err := Save3MF(path, in)
	if err == nil {
		t.Fatal("expected hard error for geometry with no valid triangles")
	}
	if report == nil || len(report.DegenerateTriangles) != 1 {
		t.Fatal("degenerate diagnostic should accompany the hard error")
	}
	if _, statErr := os.Stat(path); statErr == nil {
		t.Fatal("no file should be left behind on hard failure")
	}
}

// Float32 coordinates must survive the write/read cycle bit-for-bit; the
// exporter must not round distinct vertices to four decimal places.
func Test_3MF_FloatPrecisionRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "precision.3mf")
	src := v3.Vec{123.456789, -0.000123456, 98765.4375}
	in := []*sdf.Triangle3{
		tri(v3.Vec{0, 0, 0}, v3.Vec{1, 0, 0}, src),
	}
	if _, err := Save3MF(path, in); err != nil {
		t.Fatalf("Save3MF: %s", err)
	}
	mesh := decode3MF(t, path)
	want := toPoint3D(src)
	got := mesh.Vertices.Vertex[2]
	if got != want {
		t.Fatalf("precision lost: want %v, got %v", want, got)
	}
}

// The streaming channel path used by To3MF must deliver a report and produce
// a file equivalent to the direct Save3MF path.
func Test_3MF_StreamWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream.3mf")
	var wg sync.WaitGroup
	out, reports, err := write3MF(&wg, path)
	if err != nil {
		t.Fatalf("write3MF: %s", err)
	}
	out <- tetrahedron()
	close(out)
	wg.Wait()
	report := <-reports
	if report.Err != nil {
		t.Fatalf("stream writer error: %s", report.Err)
	}
	if !report.OK() {
		t.Fatalf("expected clean report, got:\n%s", report)
	}
	mesh := decode3MF(t, path)
	if len(mesh.Vertices.Vertex) != 4 || len(mesh.Triangles.Triangle) != 4 {
		t.Fatal("streamed mesh has wrong size")
	}
}

//-----------------------------------------------------------------------------
