

import scipy
import scipy.optimize
from scipy.optimize import minimize

import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter



df=pd.read_csv("../data/01_heights_weights_genders_errors.csv") ## Data from "Machine Learning for Hackers"
print df.head()




print ' Pull both the heights and weights from the distribution as before '
x=numpy.array( df['Height'] )
xerr = df['Height errors']
#plt.figure(1,figsize=(16,8))
#plt.subplot(121) ## [row, column, fignum]
#plt.errorbar(range(np.shape(x)[0]), x, yerr=xerr, marker='.', linestyle='None')
#plt.ylabel('Height', fontsize=22)
print 'x.shape', x.shape
print 'xerr.shape', xerr.shape



y = numpy.array( df['Weight'] )
yerr = df['Weight errors']
#plt.figure(1,figsize=(16,8))
#plt.subplot(122) ## [row, column, fignum]
#plt.errorbar(range(np.shape(y)[0]), y, yerr=yerr, marker='.', color='m', linestyle='None')
#plt.ylabel('Weight', fontsize=22)
print 'y.shape', y.shape
print 'yerr.shape', yerr.shape






def DegreesOfFreedom( 
    DataPointCount = None, 
    ParameterCount = None,
    ):
    return DataPointCount  -  ParameterCount


assert( 5 == DegreesOfFreedom( 10, 5 ) )

def Model( 
    Parameters  = None, 
    Point       = None ,
    ):
    A = Parameters[0]
    B = Parameters[1]
    #C = Parameters[2]

    Result = A*Point + B
    return Result



def GetChiSquaredFunction( 
    ModelFunction = None, 
    x = None,
    y = None,
    yerr = None,
    ):

    def ChiSquaredFunction( Parameters ):
        k = 0 
        ChiSquared = 0 
        PointCount = len(x)
        while k < PointCount:
            DataPoint = x[k]
            DataValue = y[k]
            DataError = yerr[k]
            FunctionResult = ModelFunction( 
                Parameters = Parameters,
                Point = DataPoint,
                )
            ChiSquared += ( (FunctionResult - DataValue)**2 ) / ( DataError**2 )
            k = k + 1

        ChiSquared /= DegreesOfFreedom( DataPointCount = PointCount, ParameterCount = 2)
        return ChiSquared



    return ChiSquaredFunction

def FitModelFunctionToData( 
    ModelFunction = None,  
    x = None,
    y = None,
    xerr = None,
    yerr = None,
    ):


    ChiSquaredFunction = GetChiSquaredFunction(
        ModelFunction = Model ,
        x = x,
        y = y,
        yerr = yerr
        )

    BestFitParameters  = scipy.optimize.minimize( 
        fun = ChiSquaredFunction ,
        x0 = (.5,.5),
        method = 'Nelder-Mead' )

    return BestFitParameters




def Main():

    ChiSquaredFunction = GetChiSquaredFunction(
        ModelFunction = Model ,
        x = x,
        y = y,
        yerr = yerr
        )


    ExampleChiSquaredValue = ChiSquaredFunction( (1., 1.) )
    print 'ExampleChiSquaredValue', ExampleChiSquaredValue


    
    BestFitParameters = FitModelFunctionToData( 
        ModelFunction = Model,
        x = x,
        y = y,
        xerr= xerr,
        yerr = yerr,
        )

    print 'BestFitParameters', BestFitParameters



    plt.figure(1,figsize=(16,8))
    plt.errorbar(x, y, yerr, marker='.', linestyle='None')


    #fitresults = Model(Parameters = [0.05, .5, 0.1], Point = x)

    #print fitresults.shape

    #plt.plot( x, fitresults  )
    minx = numpy.min(x)
    maxx = numpy.max(x)
    print 'numpy.min(x)', minx
    print 'numpy.max(x)', maxx
    entryvalues = numpy.linspace(minx , maxx, 100)
    print 'entryvalues', entryvalues
    plt.plot( entryvalues ,  Model(BestFitParameters.x, entryvalues) )
     #0.01* (entryvalues**2.)+ 10.*(entryvalues) - 550plt.show()



    plt.show()



    print ''





Main()













