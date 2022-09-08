#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disclaimer:
    
 @author: Dittrich Levente
 Originally I've made this in R and Excle during my second semester in 2021/22, but wanted to convert to Python too as an element of my portfolio.
 I've made some changes compared to the original which are rather clarifications and corrections.
 I'll comment my interpreters and tougths during the file.

 The data was collected in April 18 2022.
    -HDI from the UNDP(https://hdr.undp.org/en/indicators/137506)
    -Area under organic farming from Eurostat(https://ec.europa.eu/eurostat/databrowser/view/sdg_02_40/default/table?lang=en)
 The chosen year was 2019, I've already arrenged and cleaned the data in a CSV, so it's easier to reproduce my findings.

 Edit: To obtain the HDI datas you must log in to the UNDP website, which wasn't there when I collected it in April.
"""


#needed packages
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
"""
Introduction:

 On our 'Projekt 1.' course we've got the objective to explore a part of sustainable economy.
 My team and I investigated the potential of organic farms, the spread of organic farming in Europe
 and the possibility of reducing meat consumption in Hungary could be solved entirely by organic farms instead of conventional farming.


Hypothesis:
    
 In my individual project, I am looking at how organic farms impact on living standards, and what are the barriers to the spread of organic farming.
 My hypothesis is that the higher the proportion of farms that are organic farms in a country, the better people live there.
 The initial European countries (34) studied are England, Austria, Belgium, Bulgaria,
 Bulgaria, Cyprus, Czech Republic, Denmark, Estonia, Northern Macedonia, Finland, France,
 Portugal, Romania, Spain, Sweden, Switzerland, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland,
 Turkey. Later in the model building process, some countries are justifiably excluded
 of the elements analysed.


Data, descriptive statistics:
    
 To confirm or reject my hypothesis, I used the following data:
     - HDI of European countries (UNDP, 2022)
     - Total organic farm area as a percentage of total agricultural area in European countries (Eurostat, 2022)
"""


#reading the .csv
df = pd.read_csv("HDI-BioArea.csv") #BA means Bio Area, HDI means HDI, obviously

#mean and median
HDIMean = df["HDI"].mean()
HDIMed = df["HDI"].median()
HDIMax = df["HDI"].max()
HDIMin = df["HDI"].min()
BAMean = df["BA"].mean()
BAMed = df["BA"].median()
BAMax = df["BA"].max()
BAMin = df["BA"].min()


"""
 The HDI is a generally accepted measure of agricultural production, published by the United Nations Development Programme indicator based on 3 components: 
 life expectancy at birth, literacy and combined education, and GDP per capita in purchasing power parity dollars.
 This is between 0 and 1, so it's a great way to compare countries.
 Across Europe, the HDI is very high, with the lowest value being 0.774 in Northern Macedonia and the highest in Norway, at 0.957.
 On average, the surveyed countries is 0.894 and the median is 0.898, giving a very slightly rightward-sloping distribution.
 Hungary has an HDI of 0.854. The sample standard deviation is 0.04798.
 
 The ratio of total organic farm area to total agricultural area is of total agricultural area is the percentage of total agricultural area that is covered by organic
 organic farms. This number can also range between 0% and 100%, i.e. between 0 and 1, which again is really good for comparission. This data is produced by Eurostat.
 Austria is a great example, with more than a quarter of Austria's agricultural area is organic farming.
 The whole of the European Union (including the UK, since Brexit is due in 2020), Eurostat calculates this at 7.92%.
 The European Commission aims to reach 25% area under organic farming and to make aquaculture significantly greener by 2030.
 This is to achieve the target of European Green Deal's Farm to Fork strategy (European Commission, 2022).
"""


#scatterplot with all the data
sns.set_theme()
plot1 = sns.relplot(data = df, x = "BA", y = "HDI", hue = "Country", aspect=2.5)
plot1.set(title = "HDI and Biofarm Area(%) in Europe, 2019")
plot1.set_axis_labels("Biofarm Area(%)", "HDI")
plt.figure()

#data without Iceland, Luxembourg and Malta
c = ["Iceland", "Luxembourg", "Malta"]
df2 = df[~df.Country.isin(c)]
del c

#correlation with the base data
corr = df2["HDI"].corr(df["BA"])


"""
Regression:

 However, some of the European countries provided by Eurostat are, which may be distorted, such as Malta and Luxembourg, due to their size, Iceland in 
 volcanic soils, so I have excluded them from the model, looking at 31 countries.
 In the model, Y (dependent variable) is the HDI and X (explanatory variable) is the area share of organic farms in relation to the total area.
 Plotted on a scatter plot, we can see something like a logarithmic regression with a correlation of 0.399 between the two data.
"""


#converting both variables to logarithmic form
df2["LNHDI"] = np.log(df2["HDI"])
df2["LNBA"] = np.log(df2["BA"])

#correlation and R2 with the logarithmic forms
lncorr = df2["LNHDI"].corr(df2["LNBA"])
lnr2 = lncorr**2

#scatterplot of the 31 countries in logarithmic form with regression line
plot2= sns.regplot(data = df2, x = "LNBA", y = "LNHDI", color = "teal", scatter_kws={"s": 80}, ci = None)
plot2.set(title = "ln(HDI) and ln(Biofarm Area(%)) with regression line in Europe, 2019")
plot2.set_xlabel("ln(Biofarm Area(%))")
plot2.set_ylabel("ln(HDI)")
plt.figure()


"""
 However, if we take the natural logarithm of both data, this value is now 0.576, which is quite significant.
 The R2 obtained from the logarithmic transformation of the data is 0.332, i.e. ceteris paribus the change in X affects the change in y by 33.2 percent.
 I know that the logarithms of both variables are usually taken when there are large differences between observations,
 but you can see from the first dot plot that a y = α + x^β would be a great fit to the points.
"""


#removing Netherlands and Norway
c = ["Netherlands", "Norway"]
df3 = df2[~df2.Country.isin(c)]
del c

#scatterplot of the 29 countries in logarithmic form with regression line
plot3= sns.regplot(data = df3, x = "LNBA", y = "LNHDI", color = "teal", scatter_kws={"s": 80}, ci = None)
plot3.set(title = "ln(HDI) and ln(Biofarm Area(%)) with regression line in Europe, 2019")
plot3.set_xlabel("ln(Biofarm Area(%))")
plot3.set_ylabel("ln(HDI)")


"""
 The figure shows 4 outlier data, namely England, the Netherlands, Ireland and Norway.
 In these countries, with a very high HDI, the share of organic farm area in total agricultural area is very low, between 1.62% and 4.59%,
 with a significant bias even after logarithmic transformation.
 Norway's northern location and the Scandinavian Mountains covering the country justify the low organic farm area ratio,
 I feel it valid to remove it from the sample due to unfavourable conditions, just as I did with Iceland.
 In the Netherlands, the so-called horticulture is more widespread than organic agriculture, which also places a strong emphasis on a sustainable environment,
 but cannot be considered fully organic.
 As the concept of horticulture is in itself very similar to organic farming,
 but I could not find any areal data on it (it is more characterised by small-scale production and horticulture),
 I feel that the Netherlands should be excluded from the model.
 Another factor contributing to the high HDI of the British Isles is that it was a great power in the modern era until the break-up of the colonial system,
 which is still reflected in its economy and culture. Despite the fact that England and Ireland are outliers,
 there is no significant reason to exclude them from the model, 29 countries remain under study. I'll be using 95% confidence level.
"""


#correlation and R2 with the logerithmic forms with the chosen 29 countries
finalcorr = df3["LNHDI"].corr(df3["LNBA"])
finalr2 = finalcorr**2

#F statistics of the regression
y = df3["LNHDI"]
x = df3["LNBA"]
x = sm.add_constant(x)
results = sm.OLS(y, x).fit()
print(results.summary())
del x, y


"""
 In the resulting model, the correlation improved to 0.642 and the R2 to 0.412,
 with the variance of the area share of organic farms explaining 41.2% of the variance of the HDI.
 The resulting figures show a strong relationship between the two variables, as expected.
 The significance level of F is very low (0.000176), which implies that the model is significant at the 95% confidence level.
 The constant, i.e. the intercept, is 0.834 (=e^-0.18137) when back-calculated from the logarithmic form,
 which means that if there is no biofarm at all in the country, the estimated value of HDI will be this much.
 The lower and upper range when recalculated are 0.807 and 0.862 respectively.
 For the estimated beta, the t-value (4.346762) is sufficiently high, the p-value (0.000176) is low,
 0 is not included in the estimated interval, and the nullhypothesis β=0 can be rejected.
 
 Thus, the model estimates that the HDI for countries without organic farms is 0.834,
 for every one percent increase in the share of organic farm area relative to total agricultural area, the HDI increases by 3.31%.
 Presumably, there is not a one-way causal relationship between the two variables,
 as the consumption of organic products may indicate a healthy lifestyle and awareness of sustainability in the societies of the countries concerned,
 although organic farms can produce less food than conventional farms,
 but the income needed to operate them must be covered, and so these products become more expensive.
 Only consumers with high purchasing power parity can afford more expensive products in their daily lives.
 This is particularly the case in Western and Northern European countries.
 Here, however, the two variables converge: organic farms have an impact on well-being and health,
 but a high level of well-being must be established in order for consumers to be able to buy organic products.
 The variables interact with each other.


Factors complicating the transition:

 Achieving the European Commission's target for Hungary,
 mentioned in the chapter on Data, Descriptive Statistics, is a grand undertaking with many obstacles.
 Hungary's agricultural sector is highly concentrated, with 32% of agricultural land in Hungary over 500 hectares in 2016 large farms,
 owned by 0.3% of landowners (Imre Kovách, 2019).

 Large landholdings aim at quick returns, profit maximisation and cost minimisation,
 and it is not in their interest to replace large-scale grain fields, which are well managed with chemicals,
 with well-planned (environmental knowledge required) biodiverse crop rotations that require minimal chemicals (against non-native pests)
 but rely on natural protection methods (highly effective).

"""

