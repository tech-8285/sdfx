//-----------------------------------------------------------------------------
/*

2D Dual Contouring Renderer

Convert an SDF2 boundary to a set of closed line contours.

Marching squares finds the points where the surface crosses the edges of a
sampling grid and joins them with straight segments. That is a piecewise
linear approximation of the surface, so a sharp corner gets chamfered and the
position of the contour is only accurate to the grid resolution.

Dual contouring takes the dual approach:

1) one vertex is placed inside every grid cell that the surface passes through.
2) the vertex position is chosen from the distance field gradient (the surface
   normals) at the points where the surface crosses the cell edges. This is a
   least squares (QEF) problem: find the point that best satisfies
   n.(p - c) = 0 for each edge crossing c with normal n.
3) cells that share a surface crossing edge have their vertices joined.

Because the gradient of a polygon/box distance field is constant along a
straight edge, the QEF solution for a cell containing a corner is the corner
itself, and for a cell containing a straight edge it is a point on that edge.
Corners and flat sides are therefore preserved instead of being cut off.
Joining across shared crossing edges means every emitted contour is a closed
loop, and the traversal direction is chosen so that solid is on the left:
outer contours and holes have opposite winding.

A quadtree is used to find the cells that the surface passes through, so the
distance field is only sampled near the surface.

Resources:

https://www.mattkeeter.com/projects/contours/
https://www.graphics.rwth-aachen.de/publication/131/feature1.pdf
https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/

*/
//-----------------------------------------------------------------------------

package render

import (
	"fmt"
	"math"

	"github.com/deadsy/sdfx/sdf"
	"github.com/deadsy/sdfx/vec/conv"
	v2 "github.com/deadsy/sdfx/vec/v2"
	"github.com/deadsy/sdfx/vec/v2i"
)

//-----------------------------------------------------------------------------
// Grid layout
//
// Grid corners are indexed (i, j); cells are indexed (i, j) and span the
// corners (i, j) to (i+1, j+1). y increases with j.

// local edge ids for a cell.
const (
	dc2Bottom = 0 // corners (i,j) -> (i+1,j)
	dc2Top    = 1 // corners (i,j+1) -> (i+1,j+1)
	dc2Left   = 2 // corners (i,j) -> (i,j+1)
	dc2Right  = 3 // corners (i+1,j) -> (i+1,j+1)
)

// dc2Cell is the dual vertex (or vertices) placed in a grid cell.
// vmap maps each local edge to a vertex index, or -1 if the edge has no
// surface crossing. A cell with four crossings is ambiguous (the contour
// splits into two separate arcs) and gets two vertices.
type dc2Cell struct {
	n     int       // number of vertices (0, 1 or 2)
	verts [2]v2.Vec // vertex positions
	vmap  [4]int8   // local edge -> vertex index
}

// dc2Edge is a grid edge with a surface crossing.
type dc2Edge struct {
	horizontal bool   // true for a horizontal edge, false for vertical
	i, j       int    // corner index of the edge start
	p          v2.Vec // crossing point
	n          v2.Vec // outward (away from the solid) surface normal
}

// dc2EdgeKey identifies a grid edge.
type dc2EdgeKey struct {
	horizontal bool
	i, j       int
}

// dc2Node is a quadtree node. Its side is (1 << n) cells.
type dc2Node struct {
	v v2i.Vec // origin corner of the square
	n uint    // level of the square, size = 1 << n
}

// dc2Grid holds the sampled distance field and the derived contour data.
type dc2Grid struct {
	s        sdf.SDF2                // the SDF2 being rendered
	origin   v2.Vec                  // grid origin (lower left corner)
	cellSize float64                 // size of a grid cell
	nx, ny   int                     // number of cells in x and y
	hdiag    []float64               // lookup table of square half diagonals
	eps      float64                 // epsilon for gradient evaluation
	dist     map[v2i.Vec]float64     // cache of corner distances
	cells    map[v2i.Vec]dc2Cell     // cells the surface passes through
	edgeMap  map[dc2EdgeKey]*dc2Edge // cache of grid edge crossings
	edges    []*dc2Edge              // grid edges that cross the surface
}

func newDC2Grid(s sdf.SDF2, origin v2.Vec, cellSize float64, levels uint) *dc2Grid {
	n := 1 << levels
	g := dc2Grid{
		s:        s,
		origin:   origin,
		cellSize: cellSize,
		nx:       n,
		ny:       n,
		hdiag:    make([]float64, levels+1),
		dist:     make(map[v2i.Vec]float64),
		cells:    make(map[v2i.Vec]dc2Cell),
		edgeMap:  make(map[dc2EdgeKey]*dc2Edge),
	}
	// build a lut for square half diagonal lengths
	for i := range g.hdiag {
		side := math.Ldexp(1, i) * cellSize
		g.hdiag[i] = 0.5 * math.Sqrt(2.0*side*side)
	}
	// The gradient is evaluated with a small step relative to the cell size.
	g.eps = 1e-3 * cellSize
	if g.eps <= 0 {
		g.eps = 1e-9
	}
	return &g
}

//-----------------------------------------------------------------------------

// cornerPos returns the position of grid corner (i,j).
func (g *dc2Grid) cornerPos(i, j int) v2.Vec {
	return v2.Vec{X: g.origin.X + float64(i)*g.cellSize, Y: g.origin.Y + float64(j)*g.cellSize}
}

// cornerVal returns the distance at grid corner (i,j).
func (g *dc2Grid) cornerVal(i, j int) float64 {
	key := v2i.Vec{X: i, Y: j}
	if d, ok := g.dist[key]; ok {
		return d
	}
	d := g.s.Evaluate(g.cornerPos(i, j))
	g.dist[key] = d
	return d
}

// inside returns true if a distance value is inside the solid.
func dc2Inside(d float64) bool {
	return d < 0
}

// isEmpty returns true if a quadtree node contains no SDF surface.
func (g *dc2Grid) isEmpty(node *dc2Node) bool {
	// evaluate the SDF at the center of the square
	half := 1 << (node.n - 1)
	d := g.cornerVal(node.v.X+half, node.v.Y+half)
	// compare to the center/corner distance
	return math.Abs(d) >= g.hdiag[node.n]
}

// descend processes a quadtree node, recursing down to cell level.
func (g *dc2Grid) descend(node *dc2Node) {
	if node.n == 0 {
		// This is a single cell: it always needs processing.
		g.processCell(node.v.X, node.v.Y)
		return
	}
	if g.isEmpty(node) {
		return
	}
	n := node.n - 1
	s := 1 << n
	g.descend(&dc2Node{node.v, n})
	g.descend(&dc2Node{node.v.Add(v2i.Vec{X: s, Y: 0}), n})
	g.descend(&dc2Node{node.v.Add(v2i.Vec{X: s, Y: s}), n})
	g.descend(&dc2Node{node.v.Add(v2i.Vec{X: 0, Y: s}), n})
}

//-----------------------------------------------------------------------------

// gradient2 returns the normalized distance field gradient at p.
func (g *dc2Grid) gradient2(p v2.Vec) v2.Vec {
	e := g.eps
	n := v2.Vec{
		X: g.s.Evaluate(v2.Vec{X: p.X + e, Y: p.Y}) - g.s.Evaluate(v2.Vec{X: p.X - e, Y: p.Y}),
		Y: g.s.Evaluate(v2.Vec{X: p.X, Y: p.Y + e}) - g.s.Evaluate(v2.Vec{X: p.X, Y: p.Y - e}),
	}
	l := n.Length()
	if l == 0 || math.IsNaN(l) || math.IsInf(l, 0) {
		return v2.Vec{}
	}
	return n.MulScalar(1 / l)
}

// edgeRoot finds the zero crossing of the distance field on the grid edge
// between p0 and p1. The edge is known to cross (d0 and d1 differ in sign).
func (g *dc2Grid) edgeRoot(p0, p1 v2.Vec, d0 float64) v2.Vec {
	// Bisect on the sign of the distance. This is exact for a distance field
	// along the edge and gives a crossing point far more accurate than linear
	// interpolation when the edge is not perpendicular to the surface.
	a, b := 0.0, 1.0
	fa := d0
	for k := 0; k < 12; k++ {
		m := 0.5 * (a + b)
		fm := g.s.Evaluate(p0.Add(p1.Sub(p0).MulScalar(m)))
		if (fm < 0) == (fa < 0) {
			a = m
			fa = fm
		} else {
			b = m
		}
	}
	return p0.Add(p1.Sub(p0).MulScalar(0.5 * (a + b)))
}

// hEdge returns the crossing data for horizontal grid edge (i,j), or nil if
// the edge does not cross the surface.
func (g *dc2Grid) hEdge(i, j int) *dc2Edge {
	key := dc2EdgeKey{horizontal: true, i: i, j: j}
	if e, ok := g.edgeMap[key]; ok {
		return e
	}
	d0 := g.cornerVal(i, j)
	d1 := g.cornerVal(i+1, j)
	var e *dc2Edge
	if dc2Inside(d0) != dc2Inside(d1) {
		p0 := g.cornerPos(i, j)
		p1 := g.cornerPos(i+1, j)
		p := g.edgeRoot(p0, p1, d0)
		n := g.gradient2(p)
		if n.Length() == 0 {
			// Fallback: point along the edge away from the solid.
			n = v2.Vec{X: 1, Y: 0}
			if dc2Inside(d1) {
				n = v2.Vec{X: -1, Y: 0}
			}
		}
		e = &dc2Edge{horizontal: true, i: i, j: j, p: p, n: n}
		g.edges = append(g.edges, e)
	}
	g.edgeMap[key] = e
	return e
}

// vEdge returns the crossing data for vertical grid edge (i,j), or nil if the
// edge does not cross the surface.
func (g *dc2Grid) vEdge(i, j int) *dc2Edge {
	key := dc2EdgeKey{horizontal: false, i: i, j: j}
	if e, ok := g.edgeMap[key]; ok {
		return e
	}
	d0 := g.cornerVal(i, j)
	d1 := g.cornerVal(i, j+1)
	var e *dc2Edge
	if dc2Inside(d0) != dc2Inside(d1) {
		p0 := g.cornerPos(i, j)
		p1 := g.cornerPos(i, j+1)
		p := g.edgeRoot(p0, p1, d0)
		n := g.gradient2(p)
		if n.Length() == 0 {
			// Fallback: point along the edge away from the solid.
			n = v2.Vec{X: 0, Y: 1}
			if dc2Inside(d1) {
				n = v2.Vec{X: 0, Y: -1}
			}
		}
		e = &dc2Edge{horizontal: false, i: i, j: j, p: p, n: n}
		g.edges = append(g.edges, e)
	}
	g.edgeMap[key] = e
	return e
}

//-----------------------------------------------------------------------------

// dc2Crossing is a surface crossing on one of the edges of a cell.
type dc2Crossing struct {
	edge int    // local edge id
	p    v2.Vec // crossing point
	n    v2.Vec // surface normal (outward)
}

// quadrant returns the sub-box of a cell around corner c (0..3), in the same
// corner order as the cell corners (bottom-left, bottom-right, top-right,
// top-left).
func quadrant(cellMin, cellMax v2.Vec, c int) (v2.Vec, v2.Vec) {
	midX := 0.5 * (cellMin.X + cellMax.X)
	midY := 0.5 * (cellMin.Y + cellMax.Y)
	switch c {
	case 0: // bottom-left
		return cellMin, v2.Vec{X: midX, Y: midY}
	case 1: // bottom-right
		return v2.Vec{X: midX, Y: cellMin.Y}, v2.Vec{X: cellMax.X, Y: midY}
	case 2: // top-right
		return v2.Vec{X: midX, Y: midY}, cellMax
	default: // top-left
		return v2.Vec{X: cellMin.X, Y: midY}, v2.Vec{X: midX, Y: cellMax.Y}
	}
}

// processCell computes the dual vertex (or vertices) for cell (i,j).
func (g *dc2Grid) processCell(i, j int) {
	key := v2i.Vec{X: i, Y: j}
	if _, ok := g.cells[key]; ok {
		return
	}
	corners := [4]bool{
		dc2Inside(g.cornerVal(i, j)),
		dc2Inside(g.cornerVal(i+1, j)),
		dc2Inside(g.cornerVal(i+1, j+1)),
		dc2Inside(g.cornerVal(i, j+1)),
	}
	if corners[0] == corners[1] && corners[1] == corners[2] && corners[2] == corners[3] {
		// The surface does not pass through this cell.
		g.cells[key] = dc2Cell{}
		return
	}
	var cross [4]dc2Crossing
	count := 0
	add := func(e *dc2Edge, local int) {
		cross[count] = dc2Crossing{edge: local, p: e.p, n: e.n}
		count++
	}
	if e := g.hEdge(i, j); e != nil {
		add(e, dc2Bottom)
	}
	if e := g.hEdge(i, j+1); e != nil {
		add(e, dc2Top)
	}
	if e := g.vEdge(i, j); e != nil {
		add(e, dc2Left)
	}
	if e := g.vEdge(i+1, j); e != nil {
		add(e, dc2Right)
	}
	if count == 0 {
		g.cells[key] = dc2Cell{}
		return
	}
	cell := dc2Cell{}
	for k := range cell.vmap {
		cell.vmap[k] = -1
	}
	cellMin := g.cornerPos(i, j)
	cellMax := g.cornerPos(i+1, j+1)

	if count == 4 {
		// Saddle cell: the two arcs each enclose one of the two diagonal
		// corners that lie on the opposite side of the surface from the cell
		// center. Give each arc its own vertex, restricted to the quadrant
		// around that corner, so separate contours do not get pinched
		// together.
		centerInside := dc2Inside(g.s.Evaluate(v2.Vec{
			X: 0.5 * (cellMin.X + cellMax.X),
			Y: 0.5 * (cellMin.Y + cellMax.Y),
		}))
		pocket := [4]bool{
			corners[0] != centerInside,
			corners[1] != centerInside,
			corners[2] != centerInside,
			corners[3] != centerInside,
		}
		// corner pairs for the four local edges.
		edgeCorners := [4][2]int{
			dc2Bottom: {0, 1},
			dc2Top:    {2, 3},
			dc2Left:   {0, 3},
			dc2Right:  {1, 2},
		}
		var group [4][]dc2Crossing
		for k := 0; k < count; k++ {
			a, b := edgeCorners[cross[k].edge][0], edgeCorners[cross[k].edge][1]
			c := a
			if !pocket[a] {
				c = b
			}
			group[c] = append(group[c], cross[k])
		}
		index := int8(0)
		for c := 0; c < 4; c++ {
			if len(group[c]) == 0 {
				continue
			}
			if cell.n >= 2 {
				// Should not happen, but don't overflow the vertex array.
				break
			}
			qMin, qMax := quadrant(cellMin, cellMax, c)
			cell.verts[cell.n] = g.qef(group[c], qMin, qMax)
			for _, x := range group[c] {
				cell.vmap[x.edge] = index
			}
			cell.n++
			index++
		}
		g.cells[key] = cell
		return
	}

	// Two crossings (the common case): a single arc through the cell.
	cell.verts[0] = g.qef(cross[:count], cellMin, cellMax)
	for k := 0; k < count; k++ {
		cell.vmap[cross[k].edge] = 0
	}
	cell.n = 1
	g.cells[key] = cell
}

// qef positions a dual vertex so that it best satisfies the surface planes
// defined by a set of edge crossings. The solution is regularized towards the
// mean crossing point, which makes straight edges well conditioned, and is
// finally projected onto the zero level set to correct curvature error.
func (g *dc2Grid) qef(cross []dc2Crossing, cellMin, cellMax v2.Vec) v2.Vec {
	var a11, a12, a22, bx, by float64
	var mx, my float64
	for _, c := range cross {
		nx, ny := c.n.X, c.n.Y
		a11 += nx * nx
		a12 += nx * ny
		a22 += ny * ny
		w := nx*c.p.X + ny*c.p.Y
		bx += w * nx
		by += w * ny
		mx += c.p.X
		my += c.p.Y
	}
	k := float64(len(cross))
	mass := v2.Vec{X: mx / k, Y: my / k}
	// Regularize. This leaves the normal direction of the solution almost
	// untouched (so corners and edges stay sharp) while pinning down the
	// unconstrained tangential direction of a straight edge.
	lambda := 1e-3 * 0.5 * (a11 + a22)
	if lambda <= 0 {
		lambda = 1e-9
	}
	a11 += lambda
	a22 += lambda
	bx += lambda * mass.X
	by += lambda * mass.Y
	var v v2.Vec
	det := a11*a22 - a12*a12
	if math.Abs(det) < 1e-12 {
		v = mass
	} else {
		v.X = (bx*a22 - by*a12) / det
		v.Y = (a11*by - a12*bx) / det
	}
	v = v.Clamp(cellMin, cellMax)
	return g.projectToSurface(v, cellMin, cellMax)
}

// projectToSurface moves a vertex onto the zero level set of the distance
// field. The QEF solution lies slightly outside a convex curved surface
// (the tangent planes intersect outside the surface); this removes that bias.
// A sharp corner already has a near zero distance and is not moved.
func (g *dc2Grid) projectToSurface(p, cellMin, cellMax v2.Vec) v2.Vec {
	maxStep := 0.5 * g.cellSize
	for k := 0; k < 3; k++ {
		d := g.s.Evaluate(p)
		if math.Abs(d) <= 1e-12 {
			break
		}
		n := g.gradient2(p)
		if n.Length() == 0 {
			break
		}
		step := d
		if step > maxStep {
			step = maxStep
		} else if step < -maxStep {
			step = -maxStep
		}
		p = p.Sub(n.MulScalar(step))
		p = p.Clamp(cellMin, cellMax)
	}
	return p
}

//-----------------------------------------------------------------------------

// cell returns the dual cell for cell (i,j), processing it on demand.
func (g *dc2Grid) cell(i, j int) (dc2Cell, bool) {
	if i < 0 || j < 0 || i >= g.nx || j >= g.ny {
		return dc2Cell{}, false
	}
	key := v2i.Vec{X: i, Y: j}
	c, ok := g.cells[key]
	if !ok {
		g.processCell(i, j)
		c = g.cells[key]
	}
	return c, true
}

// cellVertex returns the vertex a cell contributes to one of its local edges.
func (c *dc2Cell) cellVertex(edge int) (v2.Vec, bool) {
	k := c.vmap[edge]
	if k < 0 || int(k) >= c.n {
		return v2.Vec{}, false
	}
	return c.verts[k], true
}

// emit walks the grid edges that cross the surface and joins the dual
// vertices of the adjacent cells. Every crossing edge is between exactly two
// cells, so each vertex gets two incident segments and the output is a set of
// closed contours. The segment direction is chosen so the solid is on the
// left, which makes outer contours and holes wind in opposite directions.
func (g *dc2Grid) emit(output sdf.Line2Writer) {
	lines := make([]*sdf.Line2, 0, 256)
	flush := func() {
		if len(lines) != 0 {
			output.Write(lines)
			lines = make([]*sdf.Line2, 0, 256)
		}
	}
	addLine := func(a, b v2.Vec) {
		l := sdf.Line2{a, b}
		if l.Degenerate(1e-12) {
			return
		}
		lines = append(lines, &l)
		if len(lines) >= 256 {
			flush()
		}
	}

	// The edge list can grow while we walk it: a cell may be processed on
	// demand, which registers its crossing edges.
	for k := 0; k < len(g.edges); k++ {
		e := g.edges[k]
		if e.horizontal {
			// Adjacent cells are below (i,j-1) and above (i,j).
			below, ok0 := g.cell(e.i, e.j-1)
			above, ok1 := g.cell(e.i, e.j)
			if !ok0 || !ok1 {
				continue
			}
			v0, ok0 := below.cellVertex(dc2Top)
			v1, ok1 := above.cellVertex(dc2Bottom)
			if !ok0 || !ok1 {
				continue
			}
			left := dc2Inside(g.cornerVal(e.i, e.j))
			right := dc2Inside(g.cornerVal(e.i+1, e.j))
			if left && !right {
				addLine(v0, v1)
			} else if !left && right {
				addLine(v1, v0)
			}
			continue
		}
		// Adjacent cells are left (i-1,j) and right (i,j).
		leftCell, ok0 := g.cell(e.i-1, e.j)
		rightCell, ok1 := g.cell(e.i, e.j)
		if !ok0 || !ok1 {
			continue
		}
		v0, ok0 := leftCell.cellVertex(dc2Right)
		v1, ok1 := rightCell.cellVertex(dc2Left)
		if !ok0 || !ok1 {
			continue
		}
		below := dc2Inside(g.cornerVal(e.i, e.j))
		above := dc2Inside(g.cornerVal(e.i, e.j+1))
		if !below && above {
			addLine(v0, v1)
		} else if below && !above {
			addLine(v1, v0)
		}
	}
	flush()
	output.Close()
}

//-----------------------------------------------------------------------------

// dualContouring2D generates line segments for an SDF2 using dual contouring.
func dualContouring2D(s sdf.SDF2, resolution float64, output sdf.Line2Writer) {
	// Scale the bounding box about the center to make sure the boundaries
	// aren't on the object surface, and so every surface crossing edge has
	// a cell on both sides.
	bb := s.BoundingBox()
	bb = bb.ScaleAboutCenter(1.01)
	longAxis := bb.Size().MaxComponent()
	if resolution <= 0 || longAxis <= 0 {
		output.Close()
		return
	}
	// how many quadtree levels to cover the long axis at the required resolution?
	var levels uint
	if resolution < longAxis {
		levels = uint(math.Ceil(math.Log2(longAxis / resolution)))
	}
	// create the dual contouring state
	g := newDC2Grid(s, bb.Min, resolution, levels)
	// process the quadtree, start at the top level
	g.descend(&dc2Node{v: v2i.Vec{X: 0, Y: 0}, n: levels})
	// join the vertices across the shared crossing edges
	g.emit(output)
}

//-----------------------------------------------------------------------------

// DualContouring2D renders using 2D dual contouring. Vertices are positioned
// from the distance field gradient, which preserves sharp corners and straight
// edges, and produces closed contours.
type DualContouring2D struct {
	meshCells int // number of cells on the longest axis of bounding box. e.g 200
}

// NewDualContouring2D returns a Render2 object.
func NewDualContouring2D(meshCells int) *DualContouring2D {
	return &DualContouring2D{
		meshCells: meshCells,
	}
}

// Info returns a string describing the rendered area.
func (r *DualContouring2D) Info(s sdf.SDF2) string {
	bbSize := s.BoundingBox().Size()
	resolution := bbSize.MaxComponent() / float64(r.meshCells)
	cells := conv.V2ToV2i(bbSize.MulScalar(1 / resolution))
	return fmt.Sprintf("%dx%d, resolution %.2f", cells.X, cells.Y, resolution)
}

// Render produces a 2d line mesh over the bounding area of an sdf2.
func (r *DualContouring2D) Render(s sdf.SDF2, output sdf.Line2Writer) {
	bbSize := s.BoundingBox().Size()
	resolution := bbSize.MaxComponent() / float64(r.meshCells)
	dualContouring2D(s, resolution, output)
}

//-----------------------------------------------------------------------------
