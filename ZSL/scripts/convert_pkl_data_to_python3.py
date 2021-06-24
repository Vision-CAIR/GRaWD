"""
Original pkl files were written with python2.7 so
can't be easily loaded in python 3+.
Let's convert them to normal python 3 pickle files
"""

import os
import pickle
import shutil
import argparse


FILES_TO_CONVERT = [
    'CUB2011/CUB_Porter_7551D_TFIDF_new.mat',
    'CUB2011/CUB_Porter_7551D_TFIDF_term.mat',
    'CUB2011/labels_test_hard.pkl',
    'CUB2011/labels_test.pkl',
    'CUB2011/labels_train_hard.pkl',
    'CUB2011/labels_train.pkl',
    'CUB2011/pfc_feat_test_hard.mat',
    'CUB2011/pfc_feat_test.mat',
    'CUB2011/pfc_feat_train_hard.mat',
    'CUB2011/pfc_feat_train.mat',
    'CUB2011/train_test_split_easy.mat',
    'CUB2011/train_test_split_hard.mat',

    'NABird/labels_test_hard.pkl',
    'NABird/labels_test.pkl',
    'NABird/labels_train_hard.pkl',
    'NABird/labels_train.pkl',
    'NABird/NAB_Porter_13217D_TFIDF_new.mat',
    'NABird/NAB_Porter_13217D_TFIDF_term.mat',
    'NABird/pfc_feat_test_easy.mat',
    'NABird/pfc_feat_test_hard.mat',
    'NABird/pfc_feat_train_easy.mat',
    'NABird/pfc_feat_train_hard.mat',
    'NABird/train_test_split_NABird_easy.mat',
    'NABird/train_test_split_NABird_hard.mat',
]


def main():
    for file_name in FILES_TO_CONVERT:
        f_src_name = f'old-data/{file_name}'
        f_trg_name = f'data/{file_name}'

        os.makedirs(os.path.dirname(f_trg_name), exist_ok=True)

        if f_src_name[-4:] != '.pkl':
            # Just copying non-pkl files
            shutil.copyfile(f_src_name, f_trg_name)
        else:
            print(f'Copying {f_src_name} => {f_trg_name}')

            with open(f_src_name, 'rb') as f_src:
                data = pickle.load(f_src, encoding='latin-1')

            with open(f_trg_name, 'wb') as f_trg:
                pickle.dump(data, f_trg)


if __name__ == "__main__":
    main()
