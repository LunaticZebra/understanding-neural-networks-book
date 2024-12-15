class HexCell:
    def __init__(self, q, r):
        self.q = q  # axial coordinate q
        self.r = r  # axial coordinate r
        self.neighbors = []  # list to store neighboring hex cells

    # Neighbor directions for axial coordinates (q, r)
    DIRECTIONS = [
        (1, 0),  # East
        (1, -1),  # North-East
        (0, -1),  # North-West
        (-1, 0),  # West
        (1, 1),  # South-East
        (0, 1)  # South-West
    ]

    def get_neighbors_unfiltered(self):
        # Generate the neighboring hex cells based on the direction vectors
        return [(self.q + dq, self.r + dr) for dq, dr in HexCell.DIRECTIONS]

    def __repr__(self):
        return f"HexCell({self.q}, {self.r})"


class HexGrid:
    def __init__(self, width, height):
        self.width = width  # number of columns
        self.height = height  # number of rows
        self.grid = {}
        self.generate_grid()

    def generate_grid(self):
        for r in range(self.height):
            for q in range(self.width):
                self.grid[(q, r)] = HexCell(q, r)

    def get_cell(self, q, r):
        return self.grid.get((q, r))

    def get_neighbors(self, q, r):
        cell = self.get_cell(q, r)
        if cell:
            neighbors = cell.get_neighbors_unfiltered()
            return filter(lambda coords: 0 <= coords[0] < self.width and 0 <= coords[1] < self.height, neighbors)
        return []

    def get_neighbourhood(self, q, r, neighbourhood_size):
        visited = set()

        def get_neighbourhood_inner(q, r, neighbourhood_size, neighbourhood):

            if (q, r) in visited or neighbourhood_size <= 0:
                return

            visited.add((q, r))

            neighbours = self.get_neighbors(q, r)
            neighbourhood.update(neighbours)

            if neighbourhood_size:
                for n in neighbours:
                    get_neighbourhood_inner(n.q, n.r, neighbourhood_size-1, neighbourhood)

        neighbourhood = set()
        get_neighbourhood_inner(q, r, neighbourhood_size, neighbourhood)

        return neighbourhood

    def display(self):
        for r in range(self.height):
            for q in range(self.width):
                print(f"({q},{r})", end=" ")
            print()
