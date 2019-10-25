import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SystematicStrats():

    def __init__(self, df, stocks_list, prices):
        self.df = df
        self.stock_list = stocks_list
        self.ranked_df = self.contruct_ranking()
        self.prices = prices

    def contruct_ranking(self):
        col_names = self.df.columns[1:]
        mask = self.df[col_names].applymap(lambda x: isinstance(x, (int, float)))
        self.df[col_names] = self.df[col_names].where(mask)
        self.df = self.df.ffill()
        self.df.set_index('Ticker', inplace=True)
        self.df[col_names] = self.df[col_names].apply(lambda x: x.astype('float'))

        # return only filtered df
        self.df = self.df[self.df.index.str.contains('|'.join(self.stock_list))]
        dfr = pd.DataFrame()
        for z in col_names:
            # df[str(z)+str('_R')] = 0
            dfr[str(z)] = self.df[z].rank()

        return dfr

    def generate_signals(self, nlong):
        self.ranked_dfT = self.ranked_df.transpose()
        self.ranked_dfT.index = pd.to_datetime(self.ranked_dfT.index)
        colnames = list(self.ranked_dfT.columns)
        colnames.sort()
        self.ranked_dfT = self.ranked_dfT[colnames]

        self.ranked_dfT_s = self.ranked_dfT.shift()
        #self.ranked_dfT_s.replace(np.nan, 0, inplace=True)
        df_array = self.ranked_dfT_s.values

        #long portfolio
        ones_array_l = (df_array < nlong + 1).astype(int)
        signals_l = pd.DataFrame(ones_array_l, index=self.ranked_dfT.index)
        signals_l.columns = colnames

        #short portfolio
        ones_array_s = (df_array > 17 - nlong).astype(int)
        signals_s = pd.DataFrame(ones_array_s, index=self.ranked_dfT.index)
        signals_s.columns = colnames
        return signals_l,signals_s


    def generate_systematic_strat_returns(self, sig_l, sig_s, nlong):

        rtn_l = self.generate_rtns_long_short(sig_l,nlong)
        rtn_s = self.generate_rtns_long_short(sig_s,nlong)

        return rtn_l,rtn_s

    def generate_rtns_long_short(self, sig, nlong):
        ones_array = sig.values
        ones_array = (ones_array * (1 / nlong)).round(2)
        prices_r = self.prices.reindex(self.ranked_dfT.index)
        prices_chg_r = prices_r.pct_change()
        prices_chg_r.replace(np.nan, 0, inplace=True)
        prices_chg_r_array = prices_chg_r.values

        rtns = np.multiply(prices_chg_r_array, ones_array).round(4)
        rtns_df_ls = pd.DataFrame(rtns, index=prices_chg_r.index)

        return rtns_df_ls


if __name__ == '__main__':

    n_long = 3  # number of long positions
    filename = 'Russell3000.xlsx'
    Lfile = pd.ExcelFile(filename)

    prices = pd.read_csv('Russell_price.csv', index_col='date', parse_dates=True)
    stocks = list(prices.columns)
    stocks.sort()
    prices = prices[stocks]

    ebitda = pd.read_excel(Lfile, 'EBITDA', skiprows=1)
    eps = pd.read_excel(Lfile, 'EPS', skiprows=1)
    fcf = pd.read_excel(Lfile, 'FCF', skiprows=1)
    rev = pd.read_excel(Lfile, 'REVENUE', skiprows=1)
    opt = pd.read_excel(Lfile, 'OPERATING PROFIT AFTER TAX', skiprows=1)
    # qoq = pd.read_excel(Lfile, 'QoQ Return', skiprows=1)

    # run strategy

    combo = pd.DataFrame()
    features = [eps, ebitda, fcf, rev, opt] #change as required

    for f in features:
        sys_strats = SystematicStrats(f, stocks, prices)
        signals_l,signals_s = sys_strats.generate_signals(n_long)
        sys_strat_returns_l,sys_strat_returns_s = sys_strats.generate_systematic_strat_returns(signals_l,signals_s, n_long)

        rtn_ts_l = sys_strat_returns_l.sum(axis=1)
        rtn_ts_s = sys_strat_returns_s.sum(axis=1)

        cum_rtns_l = np.cumproduct(rtn_ts_l + 1)
        cum_rtns_s = np.cumproduct(rtn_ts_s + 1)
        cum_rtn_ls = cum_rtns_l + cum_rtns_s
        rtn_ls_chg = cum_rtn_ls.pct_change()
        cum_rtn_ls_chg = np.cumproduct(rtn_ls_chg+1)-1
        cum_rtn_ls_chg.replace(np.nan, 0, inplace=True)

        combo = pd.concat([combo, cum_rtn_ls_chg], axis=1)

    # extract S&P data
    spx = pd.read_csv('spx_long.csv', index_col='Date', parse_dates=True)['Adj Close']
    spx = spx.reindex(combo.index)
    spx = spx.pct_change()
    cum_spx = np.cumproduct(spx + 1) - 1
    cum_spx.replace(np.nan, 0, inplace=True)
    combo = pd.concat([combo, cum_spx], axis=1)

    combo.columns = ['EPS', 'EBITDA', 'FCF', 'REVENUE', 'OPT','SPX'] #change as required
    combo.plot()
    plt.legend(combo.columns)
    plt.title('Systematic Long Only Strategy Based On Ranking Russell3000')
    plt.show()

    print(combo)






