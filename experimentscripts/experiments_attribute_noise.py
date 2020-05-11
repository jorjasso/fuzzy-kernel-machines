# example :
# python experiments_attribute_noise.py -d iris -c knn -i 10 -tn nn -o ../experiments/attribute_noise -no 5 -ni 5
import argparse
import sys
from functions_experiment_attribute_noise import *

# parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_name", required=True,
                help="dataset names in ['iris','wine','sonar','glass','heart','ecoli','ionosphere','wdbc','pima',"
                     "'contraceptive','yeast','segment','spambase','page-blocks','satimage', 'thyroid','ring','twonorm','penbased']")
ap.add_argument("-c", "--classifier", required=True, help="classifier in 'NSFS_NS','NSFS_KBF_symmetric',"
                                                          " 'C4.5','bagC45','svmRBF','knn',"
                                                          "'FuzzyPatternClassifier','MultimodalEvolutionaryClassifier',"
                                                          " 'FuzzyPatternTreeTopDownClassifier','FuzzyReductionRuleClassifier',"
                                                          "'FuzzyPatternClassifierGA','FuzzyPatternTreeClassifier','lr','rf','mlp','sgd'")

ap.add_argument("-i", "--n_iter", required=True, help="Number of parameter settings that are sampled for RandomsearchCV (see n_iter parameter from this procedure) n_iter trades off runtime vs quality of the solution.")
ap.add_argument("-tn", "--type_noise", required=True, help="noise type in {nn,cn,nc}")
ap.add_argument("-o", "--output_dir", required=True, help="output path")
ap.add_argument("-no", "--n_splits_outter", required=True, help="numnber of splits for outter cv")
ap.add_argument("-ni", "--n_splits_inner", required=True, help="number of splits for inner cv")


args = vars(ap.parse_args())

# args
experiment_description={'noise_level':[0,5,10,15,20],
                        'type_noise' :args['type_noise'], #options are nn, nc, cn
                        'dataset_name':args['dataset_name'],
                        'n_iter':int(args['n_iter']),
                        'output_dir':args['output_dir'],
                        'n_splits_outter':int(args['n_splits_outter']),
                        'n_splits_inner':int(args['n_splits_inner']),
                        'classifier':args['classifier']
                        }

#perform experiments
do_experiments(experiment_description)
print('finished')
sys.exit()
quit()
raise SystemExit
