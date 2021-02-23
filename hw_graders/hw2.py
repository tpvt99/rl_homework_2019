from autograder import Autograder

q_i = {'name': None, 'prefix': None, 'key': 'Eval_AverageReturn', 'thresh': None, 'weight': None}


q1_1a = {'name': 'Q1.1a', 'prefix': 'q1_sb_no_rtg_dsa', 'key': 'Eval_AverageReturn', 'thresh': 150, 'weight': 1/24}
q1_1b = {'name': 'Q1.1b', 'prefix': 'q1_sb_rtg_dsa', 'key': 'Eval_AverageReturn', 'thresh': 180, 'weight': 1/24}
q1_1c = {'name': 'Q1.1c', 'prefix': 'q1_sb_rtg_na', 'key': 'Eval_AverageReturn', 'thresh': 180, 'weight': 1/24}

q1_2a = {'name': 'Q1.2a', 'prefix': 'q1_lb_no_rtg_dsa', 'key': 'Eval_AverageReturn', 'thresh': 180, 'weight': 1/24}
q1_2b = {'name': 'Q1.2b', 'prefix': 'q1_lb_rtg_dsa', 'key': 'Eval_AverageReturn', 'thresh': 180, 'weight': 1/24}
q1_2c = {'name': 'Q1.2c', 'prefix': 'q1_lb_rtg_na', 'key': 'Eval_AverageReturn', 'thresh': 180, 'weight': 1/24}

q2 = {'name': 'Q2', 'prefix': 'q2', 'key': 'Eval_AverageReturn', 'thresh': 900, 'weight': 1/4}

q3 = {'name': 'Q3', 'prefix': 'q3', 'key': 'Eval_AverageReturn', 'thresh': 150, 'weight': 1/4}

q4a = {'name': 'Q4a', 'prefix': 'q4_search', 'key': 'Eval_AverageReturn', 'thresh': 150, 'weight': 1/8}
q4b = {'name': 'Q4b', 'prefix': 'q4', 'key': 'Eval_AverageReturn', 'thresh': 170, 'weight': 1/8}

HW2_scheme = [q1_1a, q1_1b, q1_1c, q1_2a, q1_2b, q1_2c, q2, q3, q4a, q4b]

if __name__ == '__main__':
    autograder = Autograder(HW2_scheme, make_visible=False)
    autograder.process_submission()
