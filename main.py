
import numpy as np
import scipy.optimize as sc
import pandas as pd
import matplotlib.pyplot as plt


#Function2: Calculate Portfolio historical Mean Returns and Stds
def PortfolioPerformance(weights,meanReturns,CovMatrix):
    PortMean = np.sum(meanReturns*weights)*252
    PortStd = np.sqrt((np.dot(weights.T,np.dot(CovMatrix,weights)))*252)
    return PortMean, PortStd

#Function3:Start Optimizing, by finding weights of max Sharp Ratio Portfolio. Using min method of scipy. So, we first create Negative Sharp Ratio Function
def NSR(weights,MeanReturns,CovMatrix,RiskFreeRate = 0.045):
    Portmean, PortStd = PortfolioPerformance(weights,MeanReturns,CovMatrix)
    nSR = -(Portmean - RiskFreeRate)/PortStd
    return nSR

#Function4: Optimize for weights (minimize) the function Negative Sharpe Ratio
def maxSR(MeanReturns,CovMatrix,RiskFreeRate=0.045,ConstraintSet=(0,1)):
    args = (MeanReturns,CovMatrix,RiskFreeRate)
    numStocks = len(MeanReturns)
    constraints = ({'type':'eq' , 'fun': lambda x: np.sum(x) - 1})
    bound = ConstraintSet
    bounds = tuple(bound for i in range(numStocks))
    result = sc.minimize(NSR,numStocks*[1./numStocks],args=args,method='SLSQP',bounds=bounds,constraints=constraints)
    return result

#Function5: Create formula for minimum Variance Portfolio
def MinStdF(weights,meanReturns,CovMatrix):
    return PortfolioPerformance(weights, meanReturns, CovMatrix)[1]

#Function6: Optimize (minimize) the above formula,to find minimum Std weights
def MinStd(MeanReturns,CovMatrix,ConstraintSet=(0,1)):
    args = (MeanReturns,CovMatrix)
    numStocks = len(MeanReturns)
    constraints = ({'type':'eq' , 'fun': lambda x: np.sum(x) - 1})
    bound = ConstraintSet
    bounds = tuple(bound for i in range(numStocks))
    result = sc.minimize(MinStdF,numStocks*[1./numStocks],args=args,method='SLSQP',bounds=bounds,constraints=constraints)
    return result
def portfolioreturn(weights,MeanReturns,CovMatrix):
    return PortfolioPerformance(weights, meanReturns, CovMatrix)[0]
#Function, which minimizes volatility for a given level o return.
def EfficientOptim(MeanReturns, CovMatrix, ReturnTarget, constaintset=(0,1)):
    numAssets = len(MeanReturns)
    args = (MeanReturns,CovMatrix)
    constaints = {'type':'eq','fun': lambda x: portfolioreturn(x,MeanReturns,CovMatrix)- ReturnTarget}, {'type':'eq' , 'fun': lambda x: np.sum(x) - 1}
    bound = constaintset
    bounds = tuple(bound for i in range(numAssets))
    effopt = sc.minimize(MinStdF,numAssets*[1./numAssets],args= args,method='SLSQP', bounds=bounds,constraints=constaints)
    return effopt

#Function that calculates by iritation of the above function, the efficient frontier.
def PortfolioMetrics(MeanReturns,CovMatrix,RiskFreeRate=0.045,constraintSet = (0,1)):
    maxSRPort = maxSR(MeanReturns,CovMatrix,RiskFreeRate,constraintSet)
    MaxSRReturn, MaxSRstd = PortfolioPerformance(maxSRPort['x'],MeanReturns,CovMatrix)
    MaxSRReturn, MaxSRstd = round(MaxSRReturn*100,3), round(MaxSRstd*100,3)
    PortWeights = pd.DataFrame(maxSRPort['x'],index=MeanReturns.index,columns=['Max SR Portfolio'])
    PortWeights['Max SR Portfolio'] = [round(i*100,3) for i in PortWeights['Max SR Portfolio']]
    minVarPort = MinStd(MeanReturns, CovMatrix)
    minVarReturn, minVarstd = PortfolioPerformance(minVarPort['x'], MeanReturns, CovMatrix)
    minVarReturn, minVarstd = round(minVarReturn*100,3), round(minVarstd*100,3)
    PortWeights['Least Variance Portfolio'] = minVarPort['x']
    PortWeights['Least Variance Portfolio'] = [round(i*100,3) for i in PortWeights['Least Variance Portfolio']]
    PortWeights.loc['Return'] = (MaxSRReturn, minVarReturn)
    PortWeights.loc['Standard Deviation'] = (MaxSRstd, minVarstd)
    #okey, lets do the iritation now
    TargetReturns = np.linspace(minVarReturn/100, MaxSRReturn/100, 50)
    p = 1
    for i in TargetReturns:
        b = np.array([i,EfficientOptim(MeanReturns, CovMatrix, i)['fun']])
        c = np.concatenate((EfficientOptim(MeanReturns, CovMatrix, i)['x'],b))
        c = [round(i*100,3) for i in c]
        PortWeights[p] = c
        p+=1
    return PortWeights

#Inputs
data = pd.read_excel('mutual fund returns.xlsx')
meanReturns = data.mean()
covMatrix = data.cov()
print(meanReturns)
print(covMatrix)

a = PortfolioMetrics(meanReturns,covMatrix,RiskFreeRate=0.05)
PortfolioMetrics(meanReturns,covMatrix,RiskFreeRate=0.05,constraintSet = (0,1)).to_excel('Optimal Portfolios.xlsx')
miaou1 = a.loc['Return']
miaou2 = a.loc['Standard Deviation']
plt.figure(figsize=(8, 6))
plt.scatter(miaou2, miaou1, label='Portfolios', marker='o')
plt.title('Portfolio Efficient Frontier')
plt.xlabel('Standard Deviation (%)')
plt.ylabel('Return (%)')
plt.legend()
plt.grid(True)
plt.show()
print('An excel file with the Optimal Portfolios Metrics (weights, returns and std of each optimal portfolio) has been created. Check on the assigned to python project file.')


