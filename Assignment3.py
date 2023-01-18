import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.preprocessing import MinMaxScaler 

import sklearn.metrics as skmet



def read(name):
    '''
    reading data from read function
    '''
    # read data
    data = pd.read_csv(name,skiprows=4)
   
    
    origdata=data.drop(['Country Code', 'Indicator Code'],axis=1)
    
    #for population, total
    filterpop=['Population, total']
    filterpop=origdata.loc[origdata['Indicator Name'].isin(filterpop)]
    countrypop=['Aruba','Australia']
    countrypop=filterpop.loc[filterpop['Country Name'].isin(countrypop)]
    
    #for urban population
    filterind=['Urban population']
    filterind=origdata.loc[origdata['Indicator Name'].isin(filterind)]
    countryfilter=['Aruba','Australia']
    countryfilter=filterind.loc[filterind['Country Name'].isin(countryfilter)]
    
    #one country aruba
    countfilt=['Aruba']
    countfilt=filterind.loc[filterind['Country Name'].isin(countfilt)]
    
    #one country Australia
    countfillt=['Australia']
    countfillt=filterpop.loc[filterpop['Country Name'].isin(countfillt)]
    
    
    
    cols = ['Country Name', 'Indicator Name',
           '1960','1961',  '1962', '1963', '1964', '1965', '1966', '1968',
           '1969','1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
           '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
           '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016','2017','2018','2019','2020','2021']
    #this line reorients the dataframe
    df_ind = countryfilter[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(1)
    df_indf = countfilt[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(1)
    df_inda = countfillt[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(1)
    
    df_cot = countryfilter[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(0)
    df_cot_pop = countrypop[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(0)
  
    
   
    return origdata,df_ind,df_cot,df_cot_pop,df_indf,df_inda




def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1900))
    return f

def logistics(t, scale, growth, t0):
    """ Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   

def kmean_urban(dfind,datacot,df_indf):
    
    dataindicator=dfind
    datacountry=datacot
    dataonecountry=df_indf
    
    scaler= MinMaxScaler()
    scaler=scaler.fit(datacountry[['Aruba']])
    datacountry['Aruba'] = scaler.transform(datacountry[['Aruba']])
    scaler=scaler.fit(datacountry[['Australia']])
    datacountry['Australia'] = scaler.transform(datacountry[['Australia']])
    
    for ic in range(2, 7):
        # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(datacountry)
        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(datacountry, labels))
    
    kmeans = cluster.KMeans(n_clusters=3)
    
    kmeans.fit(datacountry) # fit done on country name and level_1 as a year
    
    labels = kmeans.labels_
    cen= kmeans.cluster_centers_
    datacountry['cluster']=labels
    #clusters
    
    d1=datacountry[datacountry.cluster==0]
    d2=datacountry[datacountry.cluster==1]
    d3=datacountry[datacountry.cluster==2]
   
  
    
    plt.scatter(d1[["Aruba"]], d1[["Australia"]] )
    plt.scatter(d2[["Aruba"]], d2[["Australia"]] )
    plt.scatter(d3[["Aruba"]], d3[["Australia"]])
   
 
    plt.scatter(cen[:, 0], cen[:, 1], s=100, c='black', label = 'Centroids')
    plt.legend(["Cluster0","Cluster1","Cluster2", "Centroid"])
    plt.title("Aruba vs Australia Urban Population")
    plt.xlabel("Aruba")
    plt.ylabel("Australia")
    plt.show()
    
    year=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1968','1969',
           '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
           '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
           '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016','2017','2018','2019','2020','2021']

    dataonecountry["Year"]=year
    
    
    dataonecountry["Year"]=pd.to_numeric(dataonecountry["Year"])
    #fitting
    
    #exponential function
    popt, covar = opt.curve_fit(exp_growth, dataonecountry["Year"],dataonecountry["Urban population"], p0=[4e8, 0.02])
    
    #add new column pop exp
    dataonecountry["pop_exp"] = exp_growth(dataonecountry["Year"], *popt)
    plt.figure()
    plt.plot(dataonecountry["Year"], dataonecountry["Urban population"], label="Urban population")
    plt.plot(dataonecountry["Year"], dataonecountry["Urban population"], label="fit")
    plt.legend()
    plt.title("Exponential Urban Population")
    plt.xlabel("Year")
    plt.ylabel("Urban Population")
    plt.show()
    
    
    #exponential function 
    popt, covar = opt.curve_fit(exp_growth, dataonecountry["Year"],dataonecountry["Urban population"], p0=(4e8, 0.02))
    sigma = np.sqrt(np.diag(covar))
    
    low, up = err_ranges(dataonecountry["Year"], exp_growth, popt, sigma)
    
    dataonecountry["urb"] = exp_growth(dataonecountry["Year"], *popt)
    plt.figure()
    plt.plot(dataonecountry["Year"], dataonecountry["Urban population"], label="Urban population")
    plt.plot(dataonecountry["Year"], dataonecountry["urb"], label="forecast")
    plt.fill_between(dataonecountry["Year"], low, up, alpha=0.7,color="yellow")
    plt.legend()
    plt.title("Logistic Urban Population")
    plt.xlabel("Year")
    plt.ylabel("Urban Population")
    plt.show()
    
    #Forcasted population
    print("Forcasted population")
    low, up = err_ranges(2020, exp_growth, popt, sigma)
    print("2020 between ", low, "and", up)
    low, up = err_ranges(2030, exp_growth, popt, sigma)
    print("2030 between ", low, "and", up)
    low, up = err_ranges(2040, exp_growth, popt, sigma)
    print("2040 between ", low, "and", up)
    
    #forecasted population with mean
    print("Forcasted population")
    low, up = err_ranges(2020, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2020:", mean, "+/-", pm)
    low, up = err_ranges(2030, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2030:", mean, "+/-", pm)
    low, up = err_ranges(2040, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2040:", mean, "+/-", pm)
    
    
    
    
    return dataonecountry
    
    
    
    
    
   
    
    
def kmean_pop(df_cot_pop,df_inda):
   
    datacountryy=df_cot_pop
    dataseccountry=df_inda
    scaler= MinMaxScaler()
    scaler=scaler.fit(datacountryy[['Aruba']])
    datacountryy['Aruba'] = scaler.transform(datacountryy[['Aruba']])
    scaler=scaler.fit(datacountryy[['Australia']])
    datacountryy['Australia'] = scaler.transform(datacountryy[['Australia']])
    
    for ic in range(2, 7):
        # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(datacountryy)
        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(datacountryy, labels))
    
    kmeans = cluster.KMeans(n_clusters=3)
    
    kmeans.fit(datacountryy) # fit done on country name and level_1 as a year
    
    labels = kmeans.labels_
    cen= kmeans.cluster_centers_
    datacountryy['cluster']=labels
    #clusters
    
    d1=datacountryy[datacountryy.cluster==0]
    d2=datacountryy[datacountryy.cluster==1]
    d3=datacountryy[datacountryy.cluster==2]
   
  
    
    plt.scatter(d1[["Aruba"]], d1[["Australia"]] )
    plt.scatter(d2[["Aruba"]], d2[["Australia"]] )
    plt.scatter(d3[["Aruba"]], d3[["Australia"]])
   
 
    plt.scatter(cen[:, 0], cen[:, 1], s=100, c='black', label = 'Centroids')
    plt.legend(["Cluster0","Cluster1","Cluster2","Centroid"])
    plt.title("Aruba vs Australia Population total")
    plt.xlabel("Aruba")
    plt.ylabel("Australia")
    plt.show()
    
    year=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1968','1969',
           '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
           '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
           '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016','2017','2018','2019','2020','2021']

    dataseccountry["Year"]=year
    
    
    dataseccountry["Year"]=pd.to_numeric(dataseccountry["Year"])
    #fitting
    
    #exponential function
    popt, covar = opt.curve_fit(exp_growth, dataseccountry["Year"],dataseccountry["Population, total"], p0=[4e8, 0.02])
    
    #add new column pop exp
    dataseccountry["pop_exp"] = exp_growth(dataseccountry["Year"], *popt)
    plt.figure()
    plt.plot(dataseccountry["Year"], dataseccountry["Population, total"], label="Population, total")
    plt.plot(dataseccountry["Year"], dataseccountry["Population, total"], label="fit")
    plt.legend()
    plt.title("Exponential Population, total")
    plt.xlabel("Year")
    plt.ylabel("Population total")
    plt.show()
    
    
    #exponential function 
    popt, covar = opt.curve_fit(exp_growth, dataseccountry["Year"],dataseccountry["Population, total"], p0=(4e8, 0.02))
    sigma = np.sqrt(np.diag(covar))
    
    low, up = err_ranges(dataseccountry["Year"], exp_growth, popt, sigma)
    
    dataseccountry["poptotal"] = exp_growth(dataseccountry["Year"], *popt)
    plt.figure()
    plt.plot(dataseccountry["Year"], dataseccountry["Population, total"], label="Population, total")
    plt.plot(dataseccountry["Year"], dataseccountry["poptotal"], label="forecast")
    plt.fill_between(dataseccountry["Year"], low, up, alpha=0.7,color="yellow")
    plt.legend()
    plt.title("Logistic Population total")
    plt.xlabel("Year")
    plt.ylabel("population, total")
    plt.show()
    
    #Forcasted population
    print("Forcasted population")
    low, up = err_ranges(2020, exp_growth, popt, sigma)
    print("2020 between ", low, "and", up)
    low, up = err_ranges(2030, exp_growth, popt, sigma)
    print("2030 between ", low, "and", up)
    low, up = err_ranges(2040, exp_growth, popt, sigma)
    print("2040 between ", low, "and", up)
    
    #forecasted population with mean
    print("Forcasted population")
    low, up = err_ranges(2020, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2020:", mean, "+/-", pm)
    low, up = err_ranges(2030, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2030:", mean, "+/-", pm)
    low, up = err_ranges(2040, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2040:", mean, "+/-", pm)
    
    return datacountryy
   
     

def australia_heatmap(data,name):
    '''
    this function show heatmap and correlations of uk between indicators for better understanding
    '''
    # get dataframe
    fdata=pd.DataFrame()
        
    # get urban population
    ukdata=data[data["Indicator Name"]=="Urban population"]
        
    # get UK data
    ukurban=ukdata[ukdata['Country Name']=="Australia"].drop(['Country Name','Indicator Name'],axis=1).T
        
    # drop nan value
    ukurban=ukurban.dropna().T
        
    fdata["Urban population"]=ukurban.iloc[0]
        
    ukdata=data[data["Indicator Name"]=='Agricultural land (sq. km)']
        
    ukurban=ukdata[ukdata['Country Name']=="Australia"].drop(['Country Name','Indicator Name'],axis=1).T
        
    ukurban=ukurban.dropna().T
        
    # get arabledata
    fdata['Agricultural land (sq. km)']=ukurban.iloc[0]
        
        
    ukdata=data[data["Indicator Name"]=='Population, total']
        
    ukurban=ukdata[ukdata['Country Name']=="Australia"].drop(['Country Name','Indicator Name'],axis=1).T
        
    ukurban=ukurban.dropna().T
        
    # get total population
    fdata['Population, total']=ukurban.iloc[0]
        
    # plot a heatmap with annotation
    ax = plt.axes()
        
        # plot heat map
    heatmap = sns.heatmap(fdata.corr(), cmap="tab10",
        annot=True,ax=ax
                
        )
        
        # set title
    ax.set_title('Australia')
        
    plt.show()

def aruba_correlation_heatmap(data,name):
    
    '''
    this function show heatmap of and correlations of morocco between indicators for better understanding
    '''
        
    # crete panda dataframe
    fdata=pd.DataFrame()
        
    #  get uk urba  population  data
    moroccodata=data[data["Indicator Name"]=="Urban population"]
        
    moroccourban=moroccodata[moroccodata['Country Name']=="Morocco"].drop(['Country Name','Indicator Name'],axis=1).T
        
    # drop nan value
    moroccourban=moroccourban.dropna().T
        
    fdata["Urban population"]=moroccourban.iloc[0]
        
    # get arable data
    moroccodata=data[data["Indicator Name"]=='Agricultural land (sq. km)']
        
    moroccourban=moroccodata[moroccodata['Country Name']=="Morocco"].drop(['Country Name','Indicator Name'],axis=1).T
        
    moroccourban=moroccourban.dropna().T
        
    fdata['Agricultural land (sq. km)']=moroccourban.iloc[0]
        
    #  get total population data
    moroccodata=data[data["Indicator Name"]=='Population, total']
        
    moroccourban=moroccodata[moroccodata['Country Name']=="Morocco"].drop(['Country Name','Indicator Name'],axis=1).T
        
    moroccourban=moroccourban.dropna().T
        
    fdata['Population, total']=moroccourban.iloc[0]
        
    # plot a heatmap with annotation
    ax = plt.axes()
        
    heatmap = sns.heatmap(fdata.corr(), cmap="tab10",
        annot=True,ax=ax
                
        )
        
    # set title
    ax.set_title('Morocco')
    plt.show()  

if __name__ == '__main__':
    origdata,df_ind,df_cot,df_cot_pop,df_indf,df_inda=read("API.csv")
    dataonecountry=kmean_urban(df_ind,df_cot,df_indf)
   
    kmean_pop(df_cot_pop,df_inda)
    australia_heatmap(origdata,"Australia")
    aruba_correlation_heatmap(origdata,"Aruba")
    """
    lst = ['HNP_StatsData','WDIData','EdStatsData']
    # lst = ['HNP_StatsData','WDIData']
    df_reind=select_restack("API.csv"):
    df = joining(df_reind)
    dff=dataruba(df)
    """
   