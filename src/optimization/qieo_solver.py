import numpy as np

class QIEOSolver:
    """
    Quantum-Inspired Evolutionary Optimization (QIEO) Solver.
    Uses quantum representation (Q-bits) and quantum rotation gate updates
    to solve combinatorial and continuous optimization problems.
    """
    def __init__(self, n_bits, population_size=20, max_iterations=50, theta_max=0.05 * np.pi):
        self.n_bits = n_bits
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.theta_max = theta_max
        
        # Initialize quantum population (Q-bits)
        # Each Q-bit is represented by a pair [alpha, beta] where |alpha|^2 + |beta|^2 = 1.
        # Initially, all states are in equal superposition: alpha = beta = 1/sqrt(2).
        self.q_pop = np.ones((population_size, n_bits, 2)) / np.sqrt(2)
        
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []

    def measure_state(self, q_state):
        """Collapses quantum state to a classical binary string."""
        # Calculate probabilities of collapsing to 0 (|alpha|^2)
        probs_zero = q_state[:, 0] ** 2
        r = np.random.rand(self.n_bits)
        # Collapse to 0 if r < probs_zero, else 1
        return (r >= probs_zero).astype(int)

    def generate_classical_population(self):
        """Generates classical binary strings for the entire population."""
        classical_pop = []
        for i in range(self.population_size):
            classical_pop.append(self.measure_state(self.q_pop[i]))
        return np.array(classical_pop)

    def apply_rotation_gate(self, i, j, theta):
        """Applies quantum rotation gate to Q-bit j of individual i."""
        alpha, beta = self.q_pop[i, j, 0], self.q_pop[i, j, 1]
        
        # Rotation matrix multiplication
        # [alpha_new] = [cos(theta) -sin(theta)] [alpha]
        # [beta_new]  = [sin(theta)  cos(theta)] [beta]
        alpha_new = alpha * np.cos(theta) - beta * np.sin(theta)
        beta_new = alpha * np.sin(theta) + beta * np.cos(theta)
        
        self.q_pop[i, j, 0] = alpha_new
        self.q_pop[i, j, 1] = beta_new

    def get_rotation_angle(self, x, best_x, fitness, best_fitness, bit_idx):
        """
        Determines rotation angle theta_j according to the lookup table strategy.
        Rotates Q-bit state towards the target (best_x) configuration.
        """
        # Lookup table values based on standard QIEO literature
        if fitness < best_fitness:
            # Target is best_x
            target = best_x[bit_idx]
            current = x[bit_idx]
            
            if current == 0 and target == 1:
                # Rotate towards 1 (increase beta, decrease alpha)
                return self.theta_max
            elif current == 1 and target == 0:
                # Rotate towards 0 (decrease beta, increase alpha)
                return -self.theta_max
        return 0.0

    def optimize(self, objective_func):
        """
        Runs the optimization loop.
        objective_func takes a binary array and returns a scalar fitness value (higher is better).
        """
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        # Superposition setup
        self.q_pop = np.ones((self.population_size, self.n_bits, 2)) / np.sqrt(2)
        
        for iteration in range(self.max_iterations):
            # 1. Measurement -> Generate classical binary strings
            classical_pop = self.generate_classical_population()
            
            # 2. Evaluate fitness
            fitnesses = []
            for i in range(self.population_size):
                fit = objective_func(classical_pop[i])
                fitnesses.append(fit)
                
                # Update best global solution
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = classical_pop[i].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # 3. Update Q-bits using rotation gates
            for i in range(self.population_size):
                for j in range(self.n_bits):
                    theta = self.get_rotation_angle(
                        classical_pop[i], 
                        self.best_solution, 
                        fitnesses[i], 
                        self.best_fitness, 
                        j
                    )
                    if theta != 0:
                        self.apply_rotation_gate(i, j, theta)
                        
        return self.best_solution, self.best_fitness, self.fitness_history
