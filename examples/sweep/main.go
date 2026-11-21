//-----------------------------------------------------------------------------
/*

Sweep a 2D cross-section along a 3D path.

*/
//-----------------------------------------------------------------------------

package main

import (
	"log"
	"math"

	"github.com/deadsy/sdfx/render"
	"github.com/deadsy/sdfx/sdf"
	v2 "github.com/deadsy/sdfx/vec/v2"
	v3 "github.com/deadsy/sdfx/vec/v3"
)

//-----------------------------------------------------------------------------

// pipe is a hollow bent pipe: a circle swept along a smooth 3D path,
// with a smaller concentric sweep subtracted.
func pipe() (sdf.SDF3, error) {
	path := []v3.Vec{
		{-30, -20, 0},
		{-30, 10, 0},
		{-20, 25, 10},
		{0, 30, 20},
		{20, 25, 10},
		{30, 10, 0},
		{30, -20, 0},
	}
	outer, err := sdf.Circle2D(6)
	if err != nil {
		return nil, err
	}
	inner, err := sdf.Circle2D(4)
	if err != nil {
		return nil, err
	}
	s0, err := sdf.Sweep3D(outer, path, sdf.SweepSmooth, false, 0)
	if err != nil {
		return nil, err
	}
	s1, err := sdf.Sweep3D(inner, path, sdf.SweepSmooth, false, 0)
	if err != nil {
		return nil, err
	}
	return sdf.Difference3D(s0, s1), nil
}

// handle is a grab handle: a rounded rectangle swept along a polyline
// with a uniform twist, unioned with mounting bosses.
func handle() (sdf.SDF3, error) {
	path := []v3.Vec{
		{-25, 0, 0},
		{-25, 0, 15},
		{-15, 0, 25},
		{15, 0, 25},
		{25, 0, 15},
		{25, 0, 0},
	}
	grip := sdf.Box2D(v2.Vec{8, 5}, 1.5)
	s, err := sdf.Sweep3D(grip, path, sdf.SweepPolyline, false, sdf.Pi/2)
	if err != nil {
		return nil, err
	}
	boss, err := sdf.Cylinder3D(6, 8, 0)
	if err != nil {
		return nil, err
	}
	b0 := sdf.Transform3D(boss, sdf.Translate3d(v3.Vec{-25, 0, -1}))
	b1 := sdf.Transform3D(boss, sdf.Translate3d(v3.Vec{25, 0, -1}))
	return sdf.Union3D(s, b0, b1), nil
}

// seal is a wavy sealing ring: a circle swept around a closed smooth path.
func seal() (sdf.SDF3, error) {
	circle, err := sdf.Circle2D(2.5)
	if err != nil {
		return nil, err
	}
	var path []v3.Vec
	for i := 0; i < 24; i++ {
		theta := sdf.Tau * float64(i) / 24
		path = append(path, v3.Vec{
			25 * math.Cos(theta),
			25 * math.Sin(theta),
			2 * math.Sin(3*theta),
		})
	}
	return sdf.Sweep3D(circle, path, sdf.SweepSmooth, true, 0)
}

//-----------------------------------------------------------------------------

func main() {
	s0, err := pipe()
	if err != nil {
		log.Fatalf("error: %s", err)
	}
	render.ToSTL(s0, "pipe.stl", render.NewMarchingCubesOctree(300))

	s1, err := handle()
	if err != nil {
		log.Fatalf("error: %s", err)
	}
	render.ToSTL(s1, "handle.stl", render.NewMarchingCubesOctree(300))

	s2, err := seal()
	if err != nil {
		log.Fatalf("error: %s", err)
	}
	render.ToSTL(s2, "seal.stl", render.NewMarchingCubesOctree(300))
}

//-----------------------------------------------------------------------------
