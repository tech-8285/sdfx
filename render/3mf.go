//-----------------------------------------------------------------------------
/*

Output a 3D triangle mesh to a 3MF file.

https://3mf.io/specification/

Notes:

3D manufacturing files (3mf) generally contain meta data about the 3d object.
The files produced by this code are very basic. They are the equivalent of an
STL in 3MF format. That is: just the triangle mesh with a default 1mm unit.

File sizes for 3MF are around 7x smaller than an STL with the same mesh.

3MF files are not identical from run to run. 3MF files are a zipped archive.
The contents of the archive *are* the same but the containing zip file differs.

Topology handling:

Vertices are welded only when they are geometrically identical after the
float64 -> float32 quantization required by the 3MF format. No tolerance is
applied: distinct input vertices are never merged, so welding can never create
degenerate triangles or fuse edges that were separate in the input mesh.
Triangle winding and vertex ordering are preserved exactly, so face normals
match the input. Degenerate input triangles, duplicate faces and non-manifold
edges are reported in the MeshReport rather than being hidden by silently
copying or discarding vertices.

*/
//-----------------------------------------------------------------------------

package render

import (
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/deadsy/sdfx/sdf"
	v3 "github.com/deadsy/sdfx/vec/v3"
	"github.com/hpinc/go3mf"
)

//-----------------------------------------------------------------------------

// maxReportIssues caps how many per-issue diagnostics are retained, so a
// badly broken mesh cannot make the report itself unbounded.
const maxReportIssues = 32

// MeshReport describes topology problems found while building a 3MF mesh.
//
// A report is always returned, even when problems were found: meshes that are
// technically non-manifold are still valid 3MF triangle soups and are written
// out unchanged (apart from unrepresentable triangles, which cannot be encoded
// and are skipped). The counts let a caller decide how to react.
type MeshReport struct {
	// DegenerateTriangles lists input triangle indices whose vertices are
	// identical after float32 quantization or contain non-finite coordinates.
	// These triangles cannot be represented in 3MF and are not written.
	DegenerateTriangles []int
	// DuplicateFaces lists input triangle indices whose three vertex indices
	// match a triangle already added (same vertices, any cyclic ordering).
	// The duplicate is written only once.
	DuplicateFaces []int
	// NonManifoldEdges lists edges (as pairs of vertex indices, low first)
	// shared by more than two triangles. A closed solid has every edge
	// shared by exactly two triangles.
	NonManifoldEdges [][2]uint32
	// BoundaryEdges lists edges used by exactly one triangle. A watertight
	// solid has no boundary edges; an open surface is valid 3MF but not a
	// closed solid body.
	BoundaryEdges [][2]uint32
	// UndersizedMesh is set when the mesh has fewer than 3 vertices or 4
	// triangles, below the 3MF recommendation for a solid body. Such meshes
	// remain readable triangle soups, so this is a warning, not a failure.
	UndersizedMesh bool
	// UnreferencedVertices counts vertices not referenced by any written
	// triangle. They are retained in the file rather than silently dropped.
	UnreferencedVertices int
	// InputTriangles is the number of triangles received.
	InputTriangles int
	// WrittenTriangles is the number of triangles written to the mesh.
	WrittenTriangles int
	// Vertices is the number of (welded) vertices in the mesh.
	Vertices int
	// Err is a hard failure that prevented the file from being written.
	// Topology warnings are carried by the other fields instead.
	Err error
}

// OK reports whether the mesh is a well-formed, watertight manifold with no
// diagnostic conditions.
func (r *MeshReport) OK() bool {
	return len(r.DegenerateTriangles) == 0 &&
		len(r.DuplicateFaces) == 0 &&
		len(r.NonManifoldEdges) == 0 &&
		len(r.BoundaryEdges) == 0 &&
		!r.UndersizedMesh &&
		r.UnreferencedVertices == 0
}

// String renders the report as human readable diagnostics.
func (r *MeshReport) String() string {
	if r.OK() {
		return fmt.Sprintf("3mf mesh: %d vertices, %d triangles, manifold",
			r.Vertices, r.WrittenTriangles)
	}
	s := fmt.Sprintf(
		"3mf mesh: %d vertices, %d triangles written (%d input):",
		r.Vertices, r.WrittenTriangles, r.InputTriangles)
	if n := len(r.DegenerateTriangles); n != 0 {
		s += fmt.Sprintf("\n  %d degenerate/unrepresentable triangles skipped: %v",
			n, sampleIndices(r.DegenerateTriangles))
	}
	if n := len(r.DuplicateFaces); n != 0 {
		s += fmt.Sprintf("\n  %d duplicate faces: %v", n, sampleIndices(r.DuplicateFaces))
	}
	if n := len(r.NonManifoldEdges); n != 0 {
		s += fmt.Sprintf("\n  %d non-manifold edges (shared by >2 triangles): %v",
			n, sampleEdges(r.NonManifoldEdges))
	}
	if n := len(r.BoundaryEdges); n != 0 {
		s += fmt.Sprintf("\n  %d boundary edges (open surface, not a closed solid): %v",
			n, sampleEdges(r.BoundaryEdges))
	}
	if r.UndersizedMesh {
		s += "\n  undersized mesh (3MF solid bodies want at least 3 vertices and 4 triangles)"
	}
	if r.UnreferencedVertices != 0 {
		s += fmt.Sprintf("\n  %d unreferenced vertices retained", r.UnreferencedVertices)
	}
	return s
}

func sampleIndices(in []int) []int {
	if len(in) <= maxReportIssues {
		return in
	}
	return append(append([]int{}, in[:maxReportIssues]...), -1)
}

func sampleEdges(in [][2]uint32) [][2]uint32 {
	if len(in) <= maxReportIssues {
		return in
	}
	return in[:maxReportIssues]
}

//-----------------------------------------------------------------------------

// edgeKey is an undirected edge between two vertex indices, low index first.
type edgeKey struct {
	a, b uint32
}

// faceKey is a triangle described by its three vertex indices. It is stored
// with the indices rotated so the smallest is first, making it independent of
// cyclic vertex ordering while keeping reversed winding distinct.
type faceKey struct {
	a, b, c uint32
}

func newFaceKey(v [3]uint32) faceKey {
	// rotate so the smallest index leads; the cyclic order is preserved.
	if v[1] < v[0] && v[1] <= v[2] {
		return faceKey{v[1], v[2], v[0]}
	}
	if v[2] < v[0] && v[2] < v[1] {
		return faceKey{v[2], v[0], v[1]}
	}
	return faceKey{v[0], v[1], v[2]}
}

func newEdgeKey(a, b uint32) edgeKey {
	if a > b {
		a, b = b, a
	}
	return edgeKey{a, b}
}

// isFinitePoint reports whether all three coordinates are finite IEEE 754
// values (not NaN or +/-Inf).
func isFinitePoint(p go3mf.Point3D) bool {
	for _, x := range p {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// meshBuilder3MF accumulates triangles into a go3mf Mesh with exact vertex
// welding and topology diagnostics.
type meshBuilder3MF struct {
	mesh    *go3mf.Mesh
	indexOf map[go3mf.Point3D]uint32 // exact float32 coordinate -> vertex index
	faces   map[faceKey]struct{}     // triangles already written
	edges   map[edgeKey]uint32       // edge -> number of triangles using it
	report  MeshReport
}

func newMeshBuilder3MF(mesh *go3mf.Mesh) *meshBuilder3MF {
	return &meshBuilder3MF{
		mesh:    mesh,
		indexOf: make(map[go3mf.Point3D]uint32),
		faces:   make(map[faceKey]struct{}),
		edges:   make(map[edgeKey]uint32),
	}
}

// addVertex returns the index of the vertex at p, welding only vertices whose
// quantized float32 coordinates are bit-for-bit identical. New vertices are
// appended, so indices follow the stable first-seen order of the input stream.
func (mb *meshBuilder3MF) addVertex(p go3mf.Point3D) uint32 {
	if idx, ok := mb.indexOf[p]; ok {
		return idx
	}
	idx := uint32(len(mb.mesh.Vertices.Vertex))
	mb.mesh.Vertices.Vertex = append(mb.mesh.Vertices.Vertex, p)
	mb.indexOf[p] = idx
	return idx
}

// addTriangle adds one input triangle. The input winding is preserved.
// Degenerate and duplicate triangles are reported and skipped; everything
// else is written even if it makes the mesh non-manifold.
func (mb *meshBuilder3MF) addTriangle(t *sdf.Triangle3) {
	mb.report.InputTriangles++

	points := [3]go3mf.Point3D{
		toPoint3D(t[0]),
		toPoint3D(t[1]),
		toPoint3D(t[2]),
	}

	// Non-finite or collapsed geometry has no representation as a 3MF
	// triangle. Report it rather than silently dropping it.
	for _, p := range points {
		if !isFinitePoint(p) {
			mb.report.DegenerateTriangles = append(
				mb.report.DegenerateTriangles, mb.report.InputTriangles-1)
			return
		}
	}

	idx := [3]uint32{
		mb.addVertex(points[0]),
		mb.addVertex(points[1]),
		mb.addVertex(points[2]),
	}
	if idx[0] == idx[1] || idx[1] == idx[2] || idx[2] == idx[0] {
		mb.report.DegenerateTriangles = append(
			mb.report.DegenerateTriangles, mb.report.InputTriangles-1)
		return
	}

	// Exact duplicate faces are written once; the report records every
	// repeated input triangle.
	key := newFaceKey(idx)
	if _, ok := mb.faces[key]; ok {
		mb.report.DuplicateFaces = append(
			mb.report.DuplicateFaces, mb.report.InputTriangles-1)
		return
	}
	mb.faces[key] = struct{}{}

	// Preserve the input ordering (and therefore winding/normal direction).
	mb.mesh.Triangles.Triangle = append(mb.mesh.Triangles.Triangle,
		go3mf.Triangle{V1: idx[0], V2: idx[1], V3: idx[2]})
	mb.report.WrittenTriangles++

	for j := 0; j < 3; j++ {
		mb.edges[newEdgeKey(idx[j], idx[(j+1)%3])]++
	}
}

// finalize computes edge and vertex diagnostics and validates the assembled
// mesh against the 3MF constraints. It returns the report and any hard error
// that would make the file unreadable. Topology warnings live in the report,
// not the error.
func (mb *meshBuilder3MF) finalize() (*MeshReport, error) {
	r := &mb.report
	r.Vertices = len(mb.mesh.Vertices.Vertex)

	for e, count := range mb.edges {
		switch {
		case count == 1:
			if len(r.BoundaryEdges) < maxReportIssues {
				r.BoundaryEdges = append(r.BoundaryEdges, [2]uint32{e.a, e.b})
			}
		case count > 2:
			if len(r.NonManifoldEdges) < maxReportIssues {
				r.NonManifoldEdges = append(r.NonManifoldEdges, [2]uint32{e.a, e.b})
			}
		}
	}

	// Vertices created by skipped degenerate triangles have no face
	// referencing them. Keep them in the file but make the loss explicit.
	referenced := make([]bool, r.Vertices)
	for _, tri := range mb.mesh.Triangles.Triangle {
		referenced[tri.V1] = true
		referenced[tri.V2] = true
		referenced[tri.V3] = true
	}
	for _, used := range referenced {
		if !used {
			r.UnreferencedVertices++
		}
	}

	// Hard 3MF constraints: anything written must decode in a standard
	// reader. A mesh with no representable triangles has no encoding at all.
	if r.Vertices < 3 || r.WrittenTriangles == 0 {
		return r, fmt.Errorf("3mf mesh has no representable triangles (%d input, %d vertices)",
			r.InputTriangles, r.Vertices)
	}

	// Smaller meshes are still valid triangle soups but do not satisfy the
	// solid-body recommendation (at least 3 vertices and 4 triangles).
	if r.Vertices < 3 || r.WrittenTriangles < 4 {
		r.UndersizedMesh = true
	}

	// Defense in depth: every triangle must reference three distinct,
	// in-range vertices. The builder guarantees this, but a failure here
	// would produce an unreadable file, so check explicitly.
	nodeCount := uint32(r.Vertices)
	for i, tri := range mb.mesh.Triangles.Triangle {
		if tri.V1 == tri.V2 || tri.V1 == tri.V3 || tri.V2 == tri.V3 {
			return r, fmt.Errorf("3mf triangle %d has repeated vertex indices (%d,%d,%d)",
				i, tri.V1, tri.V2, tri.V3)
		}
		if tri.V1 >= nodeCount || tri.V2 >= nodeCount || tri.V3 >= nodeCount {
			return r, fmt.Errorf("3mf triangle %d references a vertex out of range (%d,%d,%d > %d)",
				i, tri.V1, tri.V2, tri.V3, nodeCount)
		}
	}
	return r, nil
}

// build3MFMesh accumulates a triangle slice into mesh and returns its report.
func build3MFMesh(mesh *go3mf.Mesh, triangles []*sdf.Triangle3) (*MeshReport, error) {
	mb := newMeshBuilder3MF(mesh)
	for _, t := range triangles {
		mb.addTriangle(t)
	}
	return mb.finalize()
}

//-----------------------------------------------------------------------------

// toPoint3D converts a 3D float vector to a go3mf 3D vector.
func toPoint3D(a v3.Vec) go3mf.Point3D {
	return go3mf.Point3D{float32(a.X), float32(a.Y), float32(a.Z)}
}

//-----------------------------------------------------------------------------

// Save3MF writes a triangle mesh to a 3MF file with exact vertex welding and
// topology diagnostics. The returned report is non-nil on success and
// describes any degenerate triangles, duplicate faces, non-manifold edges or
// open boundaries in the mesh; a non-nil error means no readable file was
// produced.
func Save3MF(path string, triangles []*sdf.Triangle3) (*MeshReport, error) {
	f, err := go3mf.CreateWriter(path)
	if err != nil {
		return nil, err
	}

	var model go3mf.Model
	var mesh go3mf.Mesh

	obj := &go3mf.Object{Mesh: &mesh}
	obj.ID = model.Resources.UnusedID()
	model.Resources.Objects = append(model.Resources.Objects, obj)
	model.Build.Items = append(model.Build.Items, &go3mf.Item{ObjectID: obj.ID})

	report, err := build3MFMesh(&mesh, triangles)
	if err != nil {
		f.Close()
		os.Remove(path)
		return report, err
	}

	// Emit full float32 precision instead of the library default of four
	// decimal places, so distinct float32 vertices stay distinct on disk and
	// survive an exact decode round trip.
	f.Encoder.FloatPrecision = -1

	if encErr := f.Encode(&model); encErr != nil {
		f.Close()
		os.Remove(path)
		return report, encErr
	}
	if closeErr := f.Close(); closeErr != nil {
		os.Remove(path)
		return report, closeErr
	}
	return report, nil
}

//-----------------------------------------------------------------------------

// write3MF writes a stream of triangles to a 3MF file.
//
// The returned report channel receives exactly one value when the stream
// closes: the topology report, or a report plus a non-nil Err on hard failure.
// It is buffered so the writer goroutine never blocks on a caller that stops
// reading early.
func write3MF(wg *sync.WaitGroup, path string) (
	chan<- []*sdf.Triangle3, <-chan *MeshReport, error,
) {

	f, err := go3mf.CreateWriter(path)
	if err != nil {
		return nil, nil, err
	}

	// External code writes triangles to this channel.
	// This goroutine reads the channel and writes triangles to the file.
	c := make(chan []*sdf.Triangle3)
	reports := make(chan *MeshReport, 1)

	var model go3mf.Model
	var mesh go3mf.Mesh

	// add the mesh to the model
	obj := &go3mf.Object{Mesh: &mesh}
	obj.ID = model.Resources.UnusedID()
	model.Resources.Objects = append(model.Resources.Objects, obj)
	model.Build.Items = append(model.Build.Items, &go3mf.Item{ObjectID: obj.ID})

	mb := newMeshBuilder3MF(&mesh)

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer f.Close()
		// read triangles from the channel and add them to the model
		for ts := range c {
			for _, t := range ts {
				mb.addTriangle(t)
			}
		}

		report, err := mb.finalize()
		if report == nil {
			report = &MeshReport{}
		}
		if err != nil {
			report.Err = err
			os.Remove(path)
			reports <- report
			return
		}

		// Emit full float32 precision instead of the library default of
		// four decimal places, so distinct float32 vertices stay distinct
		// on disk and survive an exact decode round trip.
		f.Encoder.FloatPrecision = -1
		if err := f.Encode(&model); err != nil {
			report.Err = err
			os.Remove(path)
			reports <- report
			return
		}
		reports <- report
	}()

	return c, reports, nil
}

//-----------------------------------------------------------------------------
