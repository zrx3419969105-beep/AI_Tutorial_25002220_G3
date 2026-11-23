"""
Artificial Intelligence - Assignment
Genetic Algorithm for Delivery Route Optimisation

The depot is A. The van has to visit B, C, D, E, F and G once each and then
go back to A. I use a simple GA to search for a short route.

Run it with:  python ga_delivery_route.py
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. The distance table from the assignment (in kilometres)
#    Same order of rows and columns: A B C D E F G
# ---------------------------------------------------------------
CITIES = ["A", "B", "C", "D", "E", "F", "G"]
POS = {city: i for i, city in enumerate(CITIES)}

DIST = [
    [0, 4, 8, 6, 7, 9, 5],   # A
    [4, 0, 5, 7, 3, 8, 6],   # B
    [8, 5, 0, 4, 6, 3, 7],   # C
    [6, 7, 4, 0, 5, 6, 8],   # D
    [7, 3, 6, 5, 0, 4, 6],   # E
    [9, 8, 3, 6, 4, 0, 5],   # F
    [5, 6, 7, 8, 6, 5, 0],   # G
]

DELIVERY_POINTS = ["B", "C", "D", "E", "F", "G"]   # A is not in the list


def dist_between(city1, city2):
    """Look up the distance between two cities in the table."""
    return DIST[POS[city1]][POS[city2]]


def route_distance(route):
    """Total km for A -> route -> A (the return trip is included)."""
    total = dist_between("A", route[0])
    for i in range(len(route) - 1):
        total = total + dist_between(route[i], route[i + 1])
    total = total + dist_between(route[-1], "A")
    return total


# ---------------------------------------------------------------
# 2. GA operators
# ---------------------------------------------------------------
def make_population(pop_size):
    """Start with random orders of the six locations."""
    population = []
    for _ in range(pop_size):
        route = DELIVERY_POINTS.copy()
        random.shuffle(route)
        population.append(route)
    return population


def tournament_select(population, k=3):
    """Pick 3 routes at random and keep the shortest one as a parent."""
    contestants = random.sample(population, k)
    contestants.sort(key=route_distance)
    return contestants[0]


def order_crossover(parent1, parent2):
    """Order crossover: copy a slice from parent1, fill the rest in
    the order they appear in parent2. The child always has all six cities."""
    cut1 = random.randint(1, len(parent1) - 2)
    cut2 = random.randint(cut1, len(parent1) - 1)

    child = [None] * len(parent1)

    # step 1: copy the slice from parent1, same positions
    child[cut1:cut2] = parent1[cut1:cut2]

    # step 2: take the remaining cities from parent2, in its order,
    # and drop them into the empty positions of the child (wrapping around)
    remaining = [city for city in parent2 if city not in child]
    positions = [i for i in range(len(child)) if child[i] is None]
    # the empty positions are the ones after cut2, then before cut1
    positions.sort(key=lambda i: (i - cut2) % len(child))
    for pos, city in zip(positions, remaining):
        child[pos] = city

    return child


def mutate(route):
    """Swap two random cities. The route stays valid, just a different order."""
    i = random.randint(0, len(route) - 1)
    j = random.randint(0, len(route) - 1)
    while j == i:
        j = random.randint(0, len(route) - 1)
    route[i], route[j] = route[j], route[i]
    return route


# ---------------------------------------------------------------
# 3. Main GA loop
# ---------------------------------------------------------------
def run_ga(seed=42, pop_size=60, generations=100, mutation_rate=0.25,
           n_elite=2):
    random.seed(seed)

    population = make_population(pop_size)
    best_route = min(population, key=route_distance)
    best_dist = route_distance(best_route)
    history = []       # best route so far, each generation
    avg_history = []   # average distance of the whole population

    for gen in range(generations):
        # sort the population so the best routes are in front
        population.sort(key=route_distance)

        # elitism: copy the best routes unchanged
        new_population = population[:n_elite]

        # build the rest of the next generation
        while len(new_population) < pop_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            child = order_crossover(parent1, parent2)

            # mutation, only sometimes
            if random.random() < mutation_rate:
                child = mutate(child)

            new_population.append(child)

        population = new_population

        # keep track of the best route found so far
        current_best = min(population, key=route_distance)
        if route_distance(current_best) < best_dist:
            best_dist = route_distance(current_best)
            best_route = current_best.copy()

        history.append(best_dist)

        # average length of the routes in the population
        total = sum(route_distance(route) for route in population)
        avg_history.append(total / len(population))

        if (gen + 1) % 10 == 0:
            print("Generation", gen + 1, "| best distance so far:", best_dist, "km")

    return best_route, best_dist, history, avg_history


def main():
    print("Delivery Route Optimisation with a Genetic Algorithm")
    print("Depot: A | Delivery points: B C D E F G")
    print()

    best_route, best_dist, history, avg_history = run_ga()

    full_route = ["A"] + best_route + ["A"]

    print()
    print("Best route found:")
    print("  " + " -> ".join(full_route))
    print("Total distance:", best_dist, "km")
    print()

    # small convergence chart, saved next to this file
    output_folder = Path(__file__).resolve().parent
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o", markersize=3,
             label="best route so far")
    plt.plot(range(1, len(avg_history) + 1), avg_history, marker="s", markersize=3,
             label="population average")
    plt.xlabel("Generation")
    plt.ylabel("Distance (km)")
    plt.title("GA convergence for the delivery routes")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "convergence.png", dpi=160)
    print()
    print("Convergence chart saved to convergence.png")


if __name__ == "__main__":
    main()
