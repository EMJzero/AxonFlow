from typing import TypeVar, Generator

T = TypeVar('T')

"""
Compute the manhattan distance betwenn two points 'pt1' and 'pt2' in an n-dimensional lattice.
"""
def manhattan(pt1 : tuple[int, ...], pt2 : tuple[int, ...]) -> int:
    if len(pt1) != len(pt2):
        raise Exception("The two points must have the same dimensionality.")
    
    return sum(abs(coord1 - coord2) for coord1, coord2 in zip(pt1, pt2))

"""
Iterate over all coordinates in a rectangular sub-lattice diagonally.
Starts from (x_beg, y_beg) [included] and proceeds over minor diagonals towards (x_end, y_end) [excluded].

Note: this coincides with an enumeration by increasing manhattan distance from (x_beg, y_beg).
"""
def iter_major_diagonals(x_beg : int, y_beg : int, x_end : int, y_end : int) -> Generator[tuple[int, int], None, None]:
    x_sign = 1 if x_beg < x_end else -1
    y_sign = 1 if y_beg < y_end else -1
    width = abs(x_end - x_beg)
    height = abs(y_beg - y_end)
    for i in range(height):
        for j in range(min(i + 1, width)):
            yield (x_beg + x_sign*j, y_beg + y_sign*(i - j))
    for j in range(1, width):
        for i in range(min(width - j, height)):
            yield (x_beg + x_sign*(j + i), y_end - y_sign*(1 + i))

"""
Development versions:

Iterate over all coordinates in a rectangular sub-lattice diagonally.
Starts from (0, 0) and proceeds from the bottom left to the top right over minor diagonals.
def iter_major_diagonals(width : int, height : int) -> Generator[tuple[int, int], None, None]:
    for i in range(height):
        #print("")
        for j in range(min(i + 1, width)):
            #print(f"(y : {i - j}, x : {j})")
            yield (j, i - j)
    #print("modechange")
    for j in range(1, width):
        #print("")
        for i in range(min(width - j, height)):
            #print(f"(y : {height - 1 - i}, x : {j + i})")
            yield (j + i, height - 1 - i)

Generic.
def itersquare(x_beg, y_beg, x_end, y_end):
    x_sign = 1 if x_beg < x_end else -1
    y_sign = 1 if y_beg < y_end else -1
    width = abs(x_end - x_beg)
    height = abs(y_beg - y_end)
    for i in range(height):
        print("")
        for j in range(min(i + 1, width)):
            print(f"(y : {y_beg + y_sign*(i - j)}, x : {x_beg + x_sign*j}) dist: {manhattan((x_beg + x_sign*j, y_beg + y_sign*(i - j)), (x_beg, y_beg))}")
    print("modechange")
    for j in range(1, width):
        print("")
        for i in range(min(width - j, height)):
            print(f"(y : {y_end - y_sign*(1 + i)}, x : {x_beg + x_sign*(j + i)}) dist: {manhattan((x_beg + x_sign*(j + i), y_end - y_sign*(1 + i)), (x_beg, y_beg))}")
"""