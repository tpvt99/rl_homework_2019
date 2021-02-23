from autograder import Autograder

q_i = {'name': None, 'prefix': None, 'key': None, 'thresh': None, 'weight': None}

q1_rnd = {'name': 'Q1_rnd', 'prefix': ['q1', 'rnd'], 'key': 'Eval_AverageReturn', 'thresh': -60, 'weight': 1/9}
q1_random = {'name': 'Q1_random', 'prefix': ['q1', 'random'], 'key': 'Eval_AverageReturn', 'thresh': -80, 'weight': 1/9}
q1_alg = {'name': 'Q1_alg', 'prefix': ['q1', 'alg'], 'key': 'Eval_AverageReturn', 'thresh': -90, 'weight': 1/9}

q2_dqn = {'name': 'Q2_dqn', 'prefix': ['q2', 'dqn'], 'key': 'Eval_AverageReturn', 'thresh': -45, 'weight': 1/9}
q2_cql = {'name': 'Q2_cql', 'prefix': ['q2', 'cql'], 'key': 'Eval_AverageReturn', 'thresh': -40, 'weight': 1/9}
q2_numsteps = {'name': 'Q2_numsteps', 'prefix': ['q2', 'numsteps'], 'key': 'Eval_AverageReturn', 'thresh': -90, 'weight': 1/9}
q2_alpha = {'name': 'Q2_alpha', 'prefix': ['q2', 'alpha'], 'key': 'Eval_AverageReturn', 'thresh': -40, 'weight': 1/9}

q3_med = {'name': 'Q3_med', 'prefix': ['q3', 'med'], 'key': 'Eval_AverageReturn', 'thresh': -40, 'weight': 1/9}
q3_hard = {'name': 'Q3_hard', 'prefix': ['q3', 'hard'], 'key': 'Eval_AverageReturn', 'thresh': -60, 'weight': 1/9}

HW5_scheme = [q1_rnd, q1_random, q1_alg, q2_dqn, q2_cql, q2_numsteps, q2_alpha, q3_med, q3_hard]

if __name__ == '__main__':
    autograder = Autograder(HW5_scheme, make_visible=False)
    autograder.process_submission()
