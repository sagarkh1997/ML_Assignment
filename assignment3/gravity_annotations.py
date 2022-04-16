import simpledorff as sd
import numpy as np
import pandas as pd
import sys

def calculate_kripendorff_alpha(csv_file):
    df = pd.read_csv(csv_file)
    df_copy = df
    df_copy.drop(columns=['final_annotation','Unnamed: 5'],inplace=True)
    df_reshape = pd.melt(df_copy.reset_index(),id_vars=['tweet_id'],value_vars=['annotator_1','annotator_2','annotator_3'],var_name='annotators',value_name='annotation')
    df_reshape.dropna(subset=['annotation'],inplace=True)
    indexes = df_reshape[df_reshape['annotation']=='X'].index
    df_reshape.drop(indexes,inplace=True)
    df_reshape.to_csv('SimpleDorff.csv')
    alpha = sd.calculate_krippendorffs_alpha_for_df(df_reshape,experiment_col='tweet_id',annotator_col='annotators',class_col='annotation')
    return alpha

def calculate_pairwise_agreement(csv_file):
    df = pd.read_csv('gravity_annotations.csv')
    index12 = df[df['annotator_3'].isna()].index
    index23 = df[df['annotator_1'].isna()].index
    index13 = df[df['annotator_2'].isna()].index
    index12_agreement = df[df['annotator_1']==df['annotator_2']].index
    index23_agreement = df[df['annotator_2']==df['annotator_3']].index
    index13_agreement = df[df['annotator_1']==df['annotator_3']].index
    pair12_agreement = len(index12_agreement)/(len(df) - len(index12))
    pair23_agreement = len(index23_agreement)/(len(df) - len(index23))
    pair13_agreement = len(index13_agreement)/(len(df) - len(index13))
    pairwise_agreement = (pair12_agreement+pair23_agreement+pair13_agreement)/3
    return pairwise_agreement


def main():
    print('Hello')
    csv_file = sys.argv[1]
    alpha = calculate_kripendorff_alpha(csv_file)
    pairwise_agreement = calculate_pairwise_agreement(csv_file)
    print('{0}  {1}'.format(alpha,pairwise_agreement))


if __name__=="__main__":
    main()