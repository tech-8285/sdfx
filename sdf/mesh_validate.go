//-----------------------------------------------------------------------------
/*

Triangle mesh topology validation.

A mesh can only define a signed distance field (and therefore participate in
offset/shell operations) when it is a closed, consistently oriented,
non-intersecting 2-manifold. The functions in this file verify those
conditions so that open or otherwise broken input fails explicitly instead
of being silently treated as a solid.

*/
//-----------------------------------------------------------------------------

package sdf

import (
	"math"

	v3 "github.com/deadsy/sdfx/vec/v3"
	"github.com/dhconnelly/rtreego"
)

//-----------------------------------------------------------------------------

// meshTolerance is the relative (to the mesh extent) distance below which two
// vertices are considered coincident when validating mesh topology.
const meshTolerance = 1e-9

// MeshTopologyError describes a way in which a triangle mesh fails to be a
// closed, consistently oriented 2-manifold.
type MeshTopologyError struct {
	// Reason is a human readable description of the first problem found.
	Reason string
	// BoundaryEdges counts edges used by exactly one triangle. Zero for a
	// closed (watertight) mesh.
	BoundaryEdges int
	// NonManifoldEdges counts edges shared by more than two triangles, or
	// shared by two triangles with inconsistent winding.
	NonManifoldEdges int
	// DegenerateTriangles counts triangles with coincident vertices or a
	// zero area.
	DegenerateTriangles int
	// IntersectingPairs counts pairs of triangles that intersect each other
	// away from shared topology (self-intersection).
	IntersectingPairs int
}

func (e *MeshTopologyError) Error() string {
	return "mesh is not a closed oriented 2-manifold: " + e.Reason
}

//-----------------------------------------------------------------------------

// qVertex is a quantized vertex position, used to merge vertices that are
// coincident within the validation tolerance.
type qVertex struct {
	x, y, z int64
}

// qEdge is a directed edge of quantized vertices.
type qEdge struct {
	a, b qVertex
}

// canonicalEdge reports whether the edge runs from the lexicographically
// smaller vertex to the larger one, giving each undirected edge a single
// canonical directed record.
func canonicalEdge(e qEdge) bool {
	switch {
	case e.a.x != e.b.x:
		return e.a.x < e.b.x
	case e.a.y != e.b.y:
		return e.a.y < e.b.y
	default:
		return e.a.z < e.b.z
	}
}

func quantizeMeshVertex(p v3.Vec, scale float64) qVertex {
	return qVertex{
		x: int64(math.Round(p.X * scale)),
		y: int64(math.Round(p.Y * scale)),
		z: int64(math.Round(p.Z * scale)),
	}
}

// signedMeshVolume returns the signed volume of the mesh using the divergence
// theorem. A consistently outward-oriented closed mesh has a positive volume.
func signedMeshVolume(mesh []*Triangle3) float64 {
	var vol float64
	for _, t := range mesh {
		vol += t[0].Dot(t[1].Cross(t[2])) / 6.0
	}
	return vol
}

// ValidateManifoldMesh checks that mesh is a closed, consistently oriented
// 2-manifold without self-intersections. On success it returns the mesh with
// a consistent outward winding orientation. Degenerate triangles are dropped
// before validation. If the mesh cannot define a solid, a *MeshTopologyError
// is returned describing why.
func ValidateManifoldMesh(mesh []*Triangle3) ([]*Triangle3, error) {
	if len(mesh) == 0 {
		return nil, &MeshTopologyError{Reason: "no triangles"}
	}

	// work out the bounding box and a tolerance-scaled quantization factor
	bb := mesh[0].BoundingBox()
	for _, t := range mesh {
		bb = bb.Extend(t.BoundingBox())
	}
	extent := bb.Size().MaxComponent()
	if extent <= 0 {
		return nil, &MeshTopologyError{Reason: "zero size bounding box"}
	}
	scale := 1.0 / (meshTolerance * extent)

	// drop degenerate triangles and quantize their vertices
	clean := make([]*Triangle3, 0, len(mesh))
	quantized := make([][3]qVertex, 0, len(mesh))
	degenerate := 0
	for _, t := range mesh {
		if t.Degenerate(meshTolerance * extent) {
			degenerate++
			continue
		}
		if t.Normal().Length2() == 0 {
			degenerate++
			continue
		}
		clean = append(clean, t)
		quantized = append(quantized, [3]qVertex{
			quantizeMeshVertex(t[0], scale),
			quantizeMeshVertex(t[1], scale),
			quantizeMeshVertex(t[2], scale),
		})
	}
	if len(clean) < 4 {
		return nil, &MeshTopologyError{
			Reason:              "fewer than 4 non-degenerate triangles cannot enclose a volume",
			DegenerateTriangles: degenerate,
		}
	}

	// every directed edge must be matched by exactly one edge with the
	// opposite direction (closed + consistently oriented manifold)
	type edgeStat struct {
		count    int // number of triangles using this directed edge
		reversed int // number of triangles using the reverse direction
	}
	edges := make(map[qEdge]*edgeStat, len(clean)*3)
	get := func(e qEdge) *edgeStat {
		if s, ok := edges[e]; ok {
			return s
		}
		s := &edgeStat{}
		edges[e] = s
		return s
	}
	for _, q := range quantized {
		for k := 0; k < 3; k++ {
			a, b := q[k], q[(k+1)%3]
			get(qEdge{a, b}).count++
			get(qEdge{b, a}).reversed++
		}
	}

	boundary, nonManifold := 0, 0
	for e, s := range edges {
		// count each undirected edge once: process only the canonical
		// (a <= b) directed record
		if !canonicalEdge(e) {
			continue
		}
		total := s.count + s.reversed
		switch {
		case total == 2 && s.count == 1 && s.reversed == 1:
			// good: one triangle each side, opposite winding
		case total == 1:
			// a single triangle uses the edge: it is open
			boundary++
		default:
			// more than two triangles share the edge, or its orientations
			// are inconsistent
			nonManifold++
		}
	}
	if boundary != 0 || nonManifold != 0 {
		return nil, &MeshTopologyError{
			Reason:              "mesh is open or non-manifold",
			BoundaryEdges:       boundary,
			NonManifoldEdges:    nonManifold,
			DegenerateTriangles: degenerate,
		}
	}

	// normalize orientation: signed volume must be positive for outward
	// winding (counter-clockwise vertices as seen from outside)
	out := clean
	if signedMeshVolume(clean) < 0 {
		out = make([]*Triangle3, len(clean))
		for i, t := range clean {
			rt := *t
			rt[1], rt[2] = rt[2], rt[1]
			out[i] = &rt
		}
	}

	// self-intersection check: triangles may only overlap at shared vertices
	// or shared edges.
	if n := countIntersectingPairs(out, scale); n != 0 {
		return nil, &MeshTopologyError{
			Reason:              "mesh contains self-intersecting triangles",
			IntersectingPairs:   n,
			DegenerateTriangles: degenerate,
		}
	}

	return out, nil
}

//-----------------------------------------------------------------------------
// Self-intersection detection.
//
// An r-tree gives broad-phase candidate pairs whose bounding boxes overlap;
// the exact triangle/triangle test then confirms a real intersection. Pairs
// that share a vertex or edge are topology, not self-intersection.

func countIntersectingPairs(mesh []*Triangle3, scale float64) int {
	spatials := make([]rtreego.Spatial, len(mesh))
	entries := make([]triEntry, len(mesh))
	for i, t := range mesh {
		b := t.BoundingBox()
		rect, _ := rtreego.NewRectFromPoints(v3ToPoint(b.Min), v3ToPoint(b.Max))
		entries[i] = triEntry{tri: t, id: i, rect: rect}
		spatials[i] = &entries[i]
	}
	tree := rtreego.NewTree(3, 3, 5, spatials...)

	// quantized vertices of each triangle, used to recognize shared topology
	quantized := make([][3]qVertex, len(mesh))
	for i, t := range mesh {
		for k := 0; k < 3; k++ {
			quantized[i][k] = quantizeMeshVertex(t[k], scale)
		}
	}
	sharesTopology := func(a, b int) bool {
		for _, va := range quantized[a] {
			for _, vb := range quantized[b] {
				if va == vb {
					return true
				}
			}
		}
		return false
	}

	count := 0
	for i, t := range mesh {
		rect := entries[i].rect
		for _, s := range tree.SearchIntersect(rect) {
			j := s.(*triEntry).id
			if j <= i {
				continue
			}
			if sharesTopology(i, j) {
				continue
			}
			if t.intersectTriangle(mesh[j]) {
				count++
			}
		}
	}
	return count
}

//-----------------------------------------------------------------------------
