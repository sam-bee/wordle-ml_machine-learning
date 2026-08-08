// Command smoke executes a tiny GoMLX graph on the configured backend.
package main

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
)

func euclideanDistance(a, b *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(a, b))))
}

func main() {
	backend := compute.MustNew()
	exec := MustNewExec1(backend, euclideanDistance)
	result := exec.MustCall(
		[]float32{1, 2},
		[]float32{4, 6},
	)

	value, ok := result.Value().(float32)
	if !ok || math.Abs(float64(value-5)) > 1e-5 {
		panic(fmt.Sprintf("unexpected GoMLX result: %T(%v), want float32(5)", result.Value(), result.Value()))
	}

	fmt.Printf("GoMLX smoke passed: backend=%s (%s), distance=%.1f\n",
		backend.Name(), backend.Description(), value)
}
