package main

// #cgo pkg-config: python3
// #cgo LDFLAGS: -L/Library/Frameworks/Python.framework/Versions/3.12/lib  -lpython3.12 -ldl -framework CoreFoundation
// #define PY_SSIZE_T_CLEAN
// #include <Python.h>
import "C"

// export isGameOver
func isGameOver(board [9][9]int) int {
	// Check if player 1 has any pieces in the first row
	for _, cell := range board[0] {
		if cell == 1 {
			return 1
		}
	}

	// Check if player 2 has any pieces in the last row
	for _, cell := range board[8] {
		if cell == 2 {
			return 2
		}
	}

	// No player has won
	return 0
}

func main() {}
