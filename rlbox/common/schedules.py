import numpy as np


def exponential_decay(epsilon_final, stop_exploration):
    ''' Calculate epsilon based on an exponential interpolation '''
    epsilon_step = -np.log(epsilon_final) / stop_exploration

    def get_epsilon(step):
        if step <= stop_exploration:
            return np.exp(-epsilon_step * step)
        else:
            return epsilon_final

    return get_epsilon


def linear_decay(epsilon_final, stop_exploration, epsilon_start=1):
    ''' Calculates epsilon based on a linear interpolation '''
    epsilon_step = -(epsilon_start - epsilon_final) / stop_exploration
    epsilon_steps = []

    def get_epsilon(step):
        if step <= stop_exploration:
            return epsilon_step * step + epsilon_start
        else:
            return epsilon_final

    return get_epsilon


def piecewise_linear_decay(boundaries, values, initial_value=1):
    ''' Linear interpolates between boundaries '''
    boundaries = [0] + boundaries
    final_epsilons = [initial_value * value for value in values]
    final_epsilons = [initial_value] + final_epsilons

    decay_steps = [
        end_step - start_step
        for start_step, end_step in zip(boundaries[:-1], boundaries[1:])
    ]

    decay_rates = [
        -(start_e - final_e) / decay_step
        for start_e, final_e, decay_step in zip(final_epsilons[:-1], final_epsilons[1:],
                                                decay_steps)
    ]

    def get_epsilon(x):
        for boundary, x0, m, y0 in zip(boundaries[1:], boundaries[:-1], decay_rates,
                                       final_epsilons):
            if x <= boundary:
                return m * (x - x0) + y0

        # Outside of boundary
        return final_epsilons[-1]

    return get_epsilon
