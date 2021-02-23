from output_messages import *
import tensorflow as tf
import traceback
import os
import numpy as np
import math
import json
import glob

def crawler(dirname):
    """ This function returns the paths of all folders submitted """
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(crawler(dirname))
    return subfolders

def parse_file(file, key):
    """ This function returns the paths of all folders submitted """
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == key:
                eval_returns.append(v.simple_value)
    return eval_returns
    
def get_section_results(file_list, section):
    """
    This function returns a score and message for the section.
    Note: The score specific score values for errors are important,
        as it determines which errors overide each other.
    """
    thresh = section['thresh']
    prefix = section['prefix']
    key = section['key']

    # No Directory Catch
    if not len(file_list):
        return -1, no_runs_found_message_func(prefix)
        
    # Corrupted Event File Catch
    try:
        eval_returns = parse_file(file_list[0], key)
    except tf.errors.DataLossError:
        error_message = traceback.format_exc()
        full_message = corruped_error_message + error_message
        return -0.5, full_message
    except Exception:
        error_message = traceback.format_exc()
        full_message = unknown_error_message + error_message
        return -0.5, full_message

    # Event File Is Empty Catch
    if not eval_returns:
        return -0.75, no_ef_found_message_func(key)
    
    # Grade Performance + Determine Message
    performance = np.max(np.array(eval_returns))
    score = int(performance >= thresh)
    message = full_credit_message if score == 1 else low_score_message_func(thresh)
    return score, message


class Autograder:
    def __init__(self, scheme, make_visible=False):
        self.scheme = scheme
        self.visibility = 'visible' if make_visible else 'after_published'
        self.log_dir = '/autograder/'
        self.submission_dir = self.log_dir + 'submission/'
        self.result_dir = self.log_dir + 'results/'
        self.all_paths = crawler(self.submission_dir)
        self.score_dict = {section['name']: -1 for section in self.scheme}
        self.output_dict = {section['name']: no_runs_found_message_func(section['prefix']) for section in self.scheme}
        self.add_joke = True
        weight_sum = sum([section['weight'] for section in self.scheme])
        assert abs(weight_sum - 1) < 1e-5

    def process_submission(self):
        results = self.grade_submission()
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        with open(os.path.join(self.result_dir, 'results.json'), 'w') as outfile:
            json.dump(results, outfile)
            
    def add_joke_to_results(self, results):
        if self.visibility != 'visible':
            return
        
        if results['score'] >= 100:
            results['output'] = cs_joke_message
        else:
            results['output'] = joke_message
    
    def grade_submission(self):
        self.process_submission_dicts()
        self.clip_grades()
        results = self.dict_to_results()
        grade = sum([section['weight'] * self.score_dict[section['name']]
                    for section in self.scheme])
        results['score'] = round(grade * 100)
        results["visibility"] = self.visibility
        
        if self.add_joke:
            self.add_joke_to_results(results)
            
        return results

    def dict_to_results(self):
        results = {'tests': [], 'score': 0}
        for key in self.score_dict.keys():
            output = {
                'name': key,
                "visibility": self.visibility,
                'score': self.score_dict[key],
                "max_score": 1.0,
                'output': self.output_dict[key]
            }
            results['tests'].append(output)
        return results
    
    def clip_grades(self):
        for section in self.scheme:
            clipped_score = max(0, min(1, self.score_dict[section['name']]))
            self.score_dict[section['name']] = clipped_score

    def process_filepath(self, folder_path, section):
        name = section['name']
        file_list = glob.glob(folder_path + '/events*')
        score, message = get_section_results(file_list, section)
        
        if score > self.score_dict[name]:
            self.score_dict[name] = score
            self.output_dict[name] = message
            
    def is_exp_instance(self, prefix, exp_folder):
        if isinstance(prefix, str): prefix = [prefix]
        return all([p in exp_folder for p in prefix])

    def process_submission_dicts(self):
        for folder_path in self.all_paths:
            exp_folder = folder_path.split('/')[-1]
            for section in self.scheme:
                if self.is_exp_instance(section['prefix'], exp_folder):
                    self.process_filepath(folder_path, section)
