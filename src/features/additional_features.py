""" Method for sofa score calculation """
from definitions import *
import numpy as np
import pandas as pd


class CalculateSofa():
    """
    Calculates the SOFA, qSOFA and septic shock classes from the data.
    """
    def qsofa_score(self, df):
        """
        1 point for each of the following:
            RR > 22/min
            SBP <= 100mmHg
            Altered mental status    # Have no way of checking this, needs GCS
        """
        qsofa = pd.DataFrame(index=df.index)
        qsofa['Resp'] = df['Resp'] >= 22
        qsofa['SBP'] = df['SBP'] <= 100
        return qsofa.sum(axis=1)


    def sofa_updater(self, data, range, reverse=False):
        s = pd.Series(index=df.index)
        for i in range(len(range) - 1):
            upper, lower = range[i], range[i+1]
            s[lower <= data < upper] = i
        return s


    def sofa_score(self, df):
        """
        Refer to https://www.mdcalc.com/sequential-organ-failure-assessment-sofa-score#evidence
        :param df:
        :return:
        """
        sofa = pd.DataFrame(index=df.index)
        sofa['Platelets'] = 0
        sofa['Platelets'][(400 <= df['Platelets']) & (df['Platelets'] < np.inf)] = 0
        sofa['Platelets'][(300 <= df['Platelets']) & (df['Platelets'] < 400)] = 1
        sofa['Platelets'][(200 <= df['Platelets']) & (df['Platelets'] < 300)] = 2
        sofa['Platelets'][(100 <= df['Platelets']) & (df['Platelets'] < 200)] = 3
        sofa['Platelets'][df['Platelets'] < 100] = 4

        sofa['Bilirubin'] = 0
        sofa['Bilirubin'][df['Bilirubin_total'] < 1.2] = 0
        sofa['Bilirubin'][(1.2 <= df['Bilirubin_total']) & ((df['Bilirubin_total']) <= 1.9)] = 1
        sofa['Bilirubin'][(1.9 < df['Bilirubin_total']) & ((df['Bilirubin_total']) <= 5.9)] = 2
        sofa['Bilirubin'][(5.9 < df['Bilirubin_total']) & ((df['Bilirubin_total']) <= 11.9)] = 3
        sofa['Bilirubin'][(11.9 < df['Bilirubin_total'])] = 4

        sofa['Creatinine'] = 0
        sofa['Creatinine'][df['Creatinine'] < 1.2] = 0
        sofa['Creatinine'][(1.2 <= df['Creatinine']) & (df['Creatinine'] < 1.9)] = 1
        sofa['Creatinine'][(1.9 <= df['Creatinine']) & (df['Creatinine'] < 3.5)] = 2
        sofa['Creatinine'][(3.5 <= df['Creatinine']) & (df['Creatinine'] < 5)] = 3
        sofa['Creatinine'][(df['Creatinine'] >= 5)] = 4

        sofa['MAP'][df['MAP'] < 70] = 1

        return sofa.sum(axis=1)

    def transform(self, df):
        df['qSOFA'] = self.qsofa_score(df)
        df['SOFA'] = self.sofa_score(df)
        return df


class SIRSLabeller():
    """
    Evaluate timepoints to see if they satisfy any 2 SIRS criteria. These are:
        1. Temp > 38degC  OR  Temp < 36degC
        2. HR > 90
        3. RespRate > 20  OR  PaCO2 < 32mmHg
        4. WBC > 12,000/mm^3  OR  WBC < 4,000/mm^3  OR  > 10% bands
    """
    def transform(self, df):
        # Create a dataframe that stores true false for each category
        df_sirs = pd.DataFrame(index=df.index, columns=['temp', 'hr', 'rr.paco2', 'wbc'])
        df_sirs['temp'] = ((df['Temp'] > 38) | (df['Temp'] < 36))
        df_sirs['hr'] = df['HR'] > 90
        df_sirs['rr.paco2'] = ((df['Resp'] > 20) | (df['PaCO2'] < 32))
        df_sirs['wbc'] = ((df['WBC'] < 4) | (df['WBC'] > 12))

        # Sum each row, if >= 2 then mar as SIRS
        df_sum = df_sirs.sum(axis=1) >= 2

        # Add to df as an sirs column
        df['SIRS'] = df_sum

        return df


class InSightFeatures(BaseIDTransformer):
    """
    Gets the insight features for each unique id
    """
    def __init__(self, window_size=3):
        self.window_size = window_size  # Num prev entries to look at. The paper used 3.

    def transform_id(self, df):
        # Get the values at time now and two timepoints prior
        df_values = self.add_previous_times(df, self.window_size)

        # Get the differences x0 - x1 and x1 - x2
        df_differences = self.add_previous_differences(df, self.window_size - 1)

        # Concat together to make InSight features
        df_insight_features = pd.concat([df_values, df_differences], axis=1)

        return df_insight_features


    def add_previous_times(df, n_times):
        """
        Adds the previous times to the current time for each row in a dataframe. Relabels cols accordingly. Rows now have
        the form:
            row_i -> x0_t, ... xk_t, x0_t-1, ..., xk_t-n_times
        :param df: the dataframe
        :param n_times: The number of times to have in the same row
        :return: df with time vals and appropriately labelled columns
        """
        df_new = pd.concat([df.shift(i) for i in range(n_times)], axis=1)
        df_new.columns = [column + '.prev({})'.format(i) for i in range(n_times, 0, -1) for column in df.columns]
        return df_new


    def add_previous_differences(df, n_differences):
        """
        Adds differences between rows. If a rows x0, x1, x2 and n_differences set to 2, returns a df with rows that contain
        entries x0 - x1, x1 - x2.
        :param df: dataframe
        :param n_differences: number of differences to add
        :return: differences df
        """
        df_new = pd.concat([df.shift(i) - df.shift(i+1) for i in range(n_differences)], axis=1)
        df_new.columns = [column + '.diff({},{})'.format(i-1, i) for i in range(n_differences, 0, -1) for column in df.columns]
        return df_new


# People with SIRS criteria are twice as likely to have sepsis
if __name__ == '__main__':
    df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
    # df = CalculateSofa().transform(df)
    df = SIRSLabeller().transform(df)