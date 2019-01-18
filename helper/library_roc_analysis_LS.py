#!/usr/bin/python
import contextlib
import math
import numpy as np
import os
import subprocess
import sys

last_file = 10
current_file = 0
actives = '/path/to/actives.ldb'
decoys = '/path/to/decoys.ldb'
directory = '/path/to/pharmacophore/library'
output_file_name = 'enrichment_factors.txt'
minimum_of_actives = 5  # minimum actives to be found to run a roc curve analysis in percent


def sdf_parser(sdf_path, molecule_count_only=False):
    actives_indices = []
    actives_scores = []
    decoys_scores = []
    hit_type_pointer = False
    index_pointer = False
    score_pointer = False
    hit_type = None
    index = None
    score = None
    mol_counter = 0
    with open(sdf_path, 'r') as sdf_text:
        if molecule_count_only:
            for line in sdf_text.readlines():
                if '$$$$' in line:
                    mol_counter += 1
            return mol_counter
        else:
            for line in sdf_text.readlines():
                if score_pointer:
                    score = float(line.strip())
                if hit_type_pointer:
                    hit_type = line.strip()
                if index_pointer:
                    index = line.strip()
                if '> <Mol. Index>' in line:
                    index_pointer = True
                else:
                    index_pointer = False
                if '> <Active/Decoy>' in line:
                    hit_type_pointer = True
                else:
                    hit_type_pointer = False
                if '> <Pharmacophore-Fit Score>' in line:
                    score_pointer = True
                else:
                    score_pointer = False
                if '$$$$' in line:
                    if hit_type == 'active':
                        actives_scores.append(score)
                        actives_indices.append(index)
                    else:
                        decoys_scores.append(score)
            return [sorted(actives_scores), sorted(decoys_scores), sorted(actives_indices)]


def auc_step(fp, auc_tp_step, number_of_actives, number_of_decoys, fraction):
    auc_fp_size = (fraction - (fp / number_of_decoys))
    if auc_fp_size < 0:
        return 0
    else:
        return auc_fp_size * (auc_tp_step / number_of_actives)


def area_under_the_curve(actives_scores, decoys_scores, number_of_actives, number_of_decoys, fraction):
    tp, fp = 0, 0
    auc_collector = 0
    while len(actives_scores + decoys_scores) > 0:
        if tp >= number_of_actives * fraction:
            break
        if len(actives_scores) > 0:
            auc_tp_step = 1
            if tp + 1 > number_of_actives * fraction:
                auc_tp_step = (number_of_actives * fraction) - tp
            if len(decoys_scores) > 0:
                if actives_scores[-1] > decoys_scores[-1]:
                    auc_collector += auc_step(fp, auc_tp_step, number_of_actives, number_of_decoys, fraction)
                    tp += 1
                    actives_scores = actives_scores[:-1]
                else:
                    fp += 1
                    decoys_scores = decoys_scores[:-1]
            else:
                tp += 1
                actives_scores = actives_scores[:-1]
                auc_collector += auc_step(fp, auc_tp_step, number_of_actives, number_of_decoys, fraction)
        else:
            fp += 1
            decoys_scores = decoys_scores[:-1]
    while tp < number_of_actives * fraction:
        auc_tp_step = 1
        if tp + 1 > number_of_actives * fraction:
            auc_tp_step = (number_of_actives * fraction) - tp
        tp += 1
        if tp == number_of_actives:
            fp = number_of_decoys
        else:
            fp += (number_of_decoys - fp) / (number_of_actives - tp)
        auc_collector += auc_step(fp, auc_tp_step, number_of_actives, number_of_decoys, fraction)
    return round(auc_collector / (fraction ** 2), 2)


def enrichment_factor(score_array, number_of_actives, number_of_decoys, fraction):
    absolute_fraction = int(math.ceil((number_of_actives + number_of_decoys) * fraction))
    tp = score_array[: absolute_fraction, 1].sum()
    fp = absolute_fraction - tp
    if absolute_fraction > len(score_array):
        fp = len(score_array) - tp
    return round((tp / (tp + fp)) / (number_of_actives / (number_of_actives + number_of_decoys)), 1)


def roc_analyzer(current_file, directory, number_of_actives, number_of_decoys):
    file_name = '/'.join([directory, str(current_file)])
    if sdf_parser('.'.join([file_name, 'sdf']), True) == 0:
        return None
    actives_scores, decoys_scores, actives_indices = sdf_parser('.'.join([file_name, 'sdf']))
    score_array = np.vstack((np.array([[x, 1] for x in actives_scores]), np.array([[x, 0] for x in decoys_scores])))
    score_array = score_array[np.flipud(score_array[:, 0].argsort())]
    EFs = []
    AUCs = []
    for fraction in [0.01, 0.05, 0.1, 1]:
        EFs.append(str(enrichment_factor(score_array, number_of_actives, number_of_decoys, fraction)))
        AUCs.append(str(area_under_the_curve(actives_scores, decoys_scores, number_of_actives, number_of_decoys,
                                             fraction)))
    actives_rate = round((len(actives_scores) / number_of_actives) * 100, 1)
    result = [str(current_file)] + EFs + AUCs + [str(actives_rate)] + [str(actives_indices)]
    print('\rPharmacophore {0}: EF1={1} EF10={3} AUC10={7} AUC100={8} Actives={9}%'.format(*result[:-1]))
    return result


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
           yield
        finally:
           sys.stdout = old_stdout


number_of_actives = 0
header = ['Pharmacophore', 'EF1', 'EF5', 'EF10', 'EF100', 'AUC1', 'AUC5', 'AUC10', 'AUC100', 'Actives [%]',
          'Actives Indices']
with open('/'.join([directory, output_file_name]), 'w') as result_file:
    result_file.write('\t'.join(header) + '\n')
    while current_file <= last_file:
        sys.stdout.write('\rAnalyzing pharmacophore {} of {}.'.format(current_file, last_file))
        roc_analysis = False
        file_name = '/'.join([directory, str(current_file)])
        with suppress_stdout():
            subprocess.run('iscreen -q {0}.pml -d {1}:active -o {0}.sdf -l {0}.log'.format(file_name, actives).split(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with open('{}.log'.format(file_name, 'r')) as log_file:
                if 'iscreen finished successfully' in log_file.read():
                    log_file.seek(0)
                    number_of_actives = 0
                    for line in log_file.readlines():
                        if 'Using database' in line:
                            try:
                                number_of_actives = int(line.split()[-2])
                                break
                            except ValueError:
                                print('\rNumber of actives in database cannot be found! Please check {}.log'.format(
                                    file_name))
                                sys.exit()
                    if number_of_actives > 0:
                        if round(number_of_actives * (minimum_of_actives / 100)) <= sdf_parser(
                                '{}.sdf'.format(file_name), True):
                            roc_analysis = True
                        else:
                            current_file += 1
                    else:
                        print('\rNumber of actives in database is found to be 0! Please check {0}.log and {1}'.format(
                            file_name, actives))
                        sys.exit()
        except FileNotFoundError:
            pass
        if roc_analysis:
            with suppress_stdout():
                subprocess.run('iscreen -q {0}.pml -d {1}:active,{2}:inactive -o {0}.sdf -l {0}.log -R {0}.png'.format(
                               file_name, actives, decoys).split(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            try:
                with open('{}.log'.format(file_name, 'r')) as log_file:
                    if 'iscreen finished successfully' in log_file.read():
                        log_file.seek(0)
                        database_counter = 0
                        number_of_actives = 0
                        number_of_decoys = 0
                        for line in log_file.readlines():
                            if 'Using database' in line:
                                if database_counter == 0:
                                    try:
                                        number_of_actives = int(line.split()[-2])
                                        database_counter += 1
                                    except ValueError:
                                        print('\rNumber of actives in database cannot be found! Please check '
                                              '{}.log'.format(file_name))
                                        sys.exit()
                                else:
                                    try:
                                        number_of_decoys = int(line.split()[-2])
                                        break
                                    except ValueError:
                                        print('\rNumber of decoys in database cannot be found! Please check '
                                              '{}.log'.format(file_name))
                                        sys.exit()
                        if number_of_actives > 0 and number_of_decoys > 0:
                            result = roc_analyzer(current_file, directory, number_of_actives, number_of_decoys)
                            if result is not None:
                                current_file += 1
                                result_file.write('\t'.join(result) + '\n')
                        else:
                            print('\rNumber of actives or decoys in databases is found to be 0! Please check {0}.log,' +
                                  '{1} and {2}.'.format(file_name, actives, decoys))
                            sys.exit()
            except FileNotFoundError:
                pass
