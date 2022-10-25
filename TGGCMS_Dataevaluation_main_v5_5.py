# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:09:21 2022

@author: poschmann
"""

#Overall programm handling TGGCMS files
import os
import pandas as pd
from scipy.signal import argrelmin
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
from scipy.signal import butter,filtfilt
import glob
import sys

def getlistofint(MSdf, text):
    print("'b' for back to main menu")
    intlist = input(text).split()
    correctentry = 'n'
    while correctentry == 'n':    
        for entry in intlist:
            try:
                int(entry)
                correctentry = 'y'
            except:
                if entry == 'b':
                    main(MSdf)
                else:
                    print('Wrong input')
                    intlist = input(text).split()
                
    return intlist

def getlimits(MSdf, limitinput):
    print("'b' for back to main menu")
    limits = input(limitinput).split(' ')
    start = limits[0]
    try:
        start = float(start)
        end = float(limits[-1])
        if start == end:
            start = 0.9 * start
            end = 1.1 * end
    except:
        if start == 'b':
            main(MSdf)
        else:
            print('wrong input')
            getlimits(MSdf, limitinput)
    start = float(start)
    end = float(end)
    return start, end

def getvalue(MSdf, valueinputtext, valuetype):
    valueinput = ''
    while (type(valueinput) != int) and (type(valueinput) != float):
        print("'b' for back to main menu")
        valueinput = input(valueinputtext)
        try: 
            valueinput = valuetype(valueinput)
        except:
            if valueinput == 'b':
                main(MSdf)
            else:
                print('wrong input')
    return valuetype(valueinput)

def getyninput(MSdf, inputquestion):
    print("'b' for back to main menu")
    ok = input(inputquestion)
    while (ok != 'n') and (ok != 'y'):
        if ok == 'b':
            main(MSdf)
        else:
            print('wrong input')
            ok = input(inputquestion)
    return ok

def normalizeIntensity(table):
    for item in table['Intensity']:
        table['Intensity'] = table['Intensity']/table['Intensity'].abs().max() *999
        table['Intensity'] = table['Intensity'].round(0)
    return table

def makegraph(X, Y):
    plt.figure(figsize=(10,8))
    plt.plot(X, Y, lw = 1)
    plt.show()
    plt.close()
    return

def massspecplot(X, Y, Label, title, show):
    #start Chromatogramm plot
    plt.figure(figsize=(10,8))
    #plot Chromatogramm
    plt.bar(X, Y, label = Label)
    #axis labels
    plt.xlabel('m/z')
    plt.ylabel('Intensity /counts')
    #legend
    plt.legend()
    #plotsaving
    plt.savefig(title, dpi = 300)
    if show != 0:
        plt.show()
    plt.close()
    return

def makegraphtofile(X, Y, LABEL, X_axis, Y_axis, X_rangemin, X_rangemax, Y_rangemin, Y_rangemax, title, show):
    #start  plot
    plt.figure(figsize=(10,8))
    #plot 
    plt.plot(X, Y, label = LABEL, lw = 1)
    #axis labels
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    #legend
    plt.legend()
    #plot range if value given
    if X_rangemin != False:
        plt.xlim(X_rangemin,X_rangemax)
    if Y_rangemin != False:
        plt.ylim(Y_rangemin,Y_rangemax)
    #save figure
    plt.savefig(title, dpi = 300)
    if show != 0:
        plt.show()
    plt.close()
    return

def makegraphplusscattertofile(X, Y, LABEL, X_scatter, Y_scatter, scatter_label, X_axis, Y_axis, X_rangemin, X_rangemax, Y_rangemin, Y_rangemax, title, show):
    #start  plot
    plt.figure(figsize=(10,8))
    #plot 
    plt.plot(X, Y, label = LABEL, lw = 1)
    plt.scatter(X_scatter, Y_scatter, label = scatter_label, c='r')
    #axis labels
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    #legend
    plt.legend()
    #plot range if value given
    if X_rangemin != False:
        plt.xlim(X_rangemin,X_rangemax)
    if Y_rangemin != False:
        plt.ylim(Y_rangemin,Y_rangemax)
    #save figure
    if title !=0:
        plt.savefig(title, dpi = 300)
    if show != 0:
        plt.show()
    else:
        plt.close()
    return

def massestolookat(MSdf):
    #rearrange data of each TIC
    for Nr in pd.unique(MSdf['ScanRange']):
        Range = MSdf[MSdf['ScanRange'] == Nr].drop('ScanRange', axis = 1)
        Range['Mass'] = Range['Mass'].round(0).astype(int)
        Range['RetentionTime'] = Range['RetentionTime'].round(2)
        pivRng1 = Range.pivot_table(index = 'Mass', columns = 'RetentionTime', values = 'Intensity', aggfunc='sum')
        pivRng1.fillna(0, inplace = True)
        df = pivRng1.reset_index()
        SoverNlist = []
        #iterate through every detected SIR
        for index, row in df.iterrows():
            #ignore masses below 11
            if row.iloc[0] <= 11:
                continue
            else:
                #calculate mean Intensity and stddeviation in first datapoints to determine Noise
                meanbase = row.iloc[2:72].sum()/70
                stddevbase = row.iloc[2:72].std()
                #find maximum of SIR
                maxint = row.max()    
                #devide maximum of background substracted SIR by standard deviation as value for Signal over noise ratio
                maxoverstddev = (maxint-meanbase) / stddevbase
                #generate list of masses with high S/N ratio with values above 20
                if maxoverstddev >= 20:
                    listentry = [row.iloc[0], maxoverstddev.round(2)]
                    SoverNlist.append(listentry)
        #generate file with interessting masses
        SoverNdf = pd.DataFrame(SoverNlist, columns = ['Mass', 'S/R_ratio']).set_index('Mass')
        SoverNdf.sort_values('S/R_ratio', axis = 0, ascending = False, inplace = True)
        SoverNdf.to_csv('MassesSoverNratio_in_Scanrange_'+ str(Nr)+ '.xy')
    return

def noisefilter(MSdf):
    filter_ok = 'n'
    while filter_ok == 'n':
        #generate workingcopy of datafile
        MSdf_operate=MSdf.copy(deep=True)
        #ask for inputvalues for the Butterworth filter
        fs = 0.7 * getvalue(MSdf, 'Minimum Peak Width in Seconds (~1.0): ', float)#  int(input('Minimum Peak Width in Seconds (~1.0): '))
        cutoff = getvalue(MSdf, 'Noise Frequency in Hz (~0.2): 0.', int) /10 #int(input ('Noise Frequency in Hz (~0.2): 0.')) /10      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz                       
        # Filter requirements.
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2       # sin wave can be approx represented as quadratic
        normal_cutoff = cutoff / nyq
        #generate dataframe to put noisefiltered data in
        datalist = [] 
        RetTimelist = []
        #scan through each TIC
        for Nr in pd.unique(MSdf_operate['ScanRange']): 
            #rearange data to work on 
            Range = MSdf_operate.loc[MSdf_operate['ScanRange'] == Nr].copy(deep=True)
            pivRng1 = Range.pivot_table(index = 'Mass', columns = 'RetentionTime', values = 'Intensity', aggfunc='sum')
            pivRng1.fillna(0, inplace = True)
            df = pivRng1.transpose()
            # make Chromatogram plot to compare later
            makegraph(df.sum(axis = 1).index, df.sum(axis = 1))
            #operate the noise filter on each single mass
            Counter = 0
            for mass in df.columns:
                data=df[mass]
                #if (Counter % 10 ==0):
                #    print(str(int(Counter/len(df.columns)*100)) + ' %')
                Counter += 1 
                # Filter requirements.
                nyq = 0.5 * fs  # Nyquist Frequency
                order = 2       # sin wave can be approx represented as quadratic
                normal_cutoff = cutoff / nyq
                # Get the filter coefficients 
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                # filter the noise of mass scan
                df[mass] = filtfilt(b, a, data).astype(int)
                #rearrange noise filtered data
                dfnew=df[mass].reset_index()
                dfnew.columns = ['RetentionTime', 'Intensity']
                dfnew.insert(loc=1, column = 'Mass', value = mass)
                dfnew.insert(loc=0, column = 'Scannumber', value = range(1, len(dfnew['RetentionTime'])+1))
                dfnew.insert(loc=0, column = 'ScanRange', value = Nr)
                #collect all noise filtered mass data in new dataframes
                #to save memory Retention Time is put into extra list as float numbers, while other list contains integers
                rettime = dfnew.pop('RetentionTime')
                RetTimelist.extend(rettime.values.tolist())
                datalist.extend(dfnew.values.tolist()) 
            #make graph of noisefiltered Chromatogram to compare with unfiltered data
            makegraph(df.sum(axis = 1).index, df.sum(axis = 1))
        #rearrange datalist so it fits into other programs
        df = pd.DataFrame(datalist, columns = ['ScanRange', 'Scannumber', 'Mass', 'Intensity'])
        # reinsert Retention times into dataframe
        df.insert(loc=2, column = 'RetentionTime', value = pd.DataFrame(RetTimelist))
        # sort dataframe so it matches input MSdf
        df = df.sort_values(by=['Scannumber', 'ScanRange'])
        df = df[df['Intensity'] > 0]
        filter_ok = 'y' #input('Filter factor level ok ? (y/n): ') 
    return df

def getsinglemassspectra(MSdf):
    scannumberlist = pd.unique(MSdf['ScanRange'])
    if len(scannumberlist) >= 1:
        scannumberlist = [str(x) for x in scannumberlist]
        print("'b' for back to main menu")
        scannumber = input('On which scan do you want to operate? (' + str(scannumberlist)+ ') :')
        while scannumber not in scannumberlist:
            if scannumber == 'b':
                main(MSdf)
            else:
                print('wrong input')
                print("'b' for back to main menu")
                scannumber = input('On which scan do you want to operate? (' + str(scannumberlist)+ ') :')
    else:
        scannumber = 1
    scannumber = int(scannumber)
    Range = MSdf[MSdf['ScanRange'] == scannumber].drop('ScanRange', axis = 1)
    MSChromatogramm1 = Range.groupby(['RetentionTime'], sort = False).sum('Intensity') 
    #generating Scannumber for operation   
    MSChromatogramm1['Scannumber'] = range(len(MSChromatogramm1))
    MSChromatogramm1.reset_index(inplace = True)
    #starting loop to limit data range to relevant region
    range_ok = 'n'
    while range_ok != 'y':
        #making graph so operator sees what data he is working on
        makegraph(MSChromatogramm1.index, MSChromatogramm1['Intensity'])
        #asking for left and right data limit
        xstartB, xendB = getlimits(MSdf, 'Range you want to have a closer look (0 - ' + str(MSChromatogramm1['RetentionTime'].max()) + ') (min max): ') 
        MSChromatogrammcut = MSChromatogramm1[MSChromatogramm1['RetentionTime'] >= float(xstartB)]
        MSChromatogrammcut = MSChromatogrammcut[MSChromatogrammcut['RetentionTime'] <= float(xendB)]
        #show graph of data range maxima finding routine works on
        makegraph(MSChromatogrammcut['RetentionTime'], MSChromatogrammcut['Intensity'])
        #checking if range is correct           
        range_ok = getyninput(MSdf, 'Is the range of data points correct?(y/n): ')
    Rettime = getvalue(MSdf, 'What Retention time do you want the mass spectra extracted? (X.XX): ', float)
    #cutting dataframe to limits    
    MSChromatogrammcut = MSChromatogramm1[MSChromatogramm1['RetentionTime'] >= (Rettime -0.1)]
    MSChromatogrammcut = MSChromatogrammcut[MSChromatogrammcut['RetentionTime'] <= (Rettime +0.1)]
    Rettimedet = MSChromatogrammcut['RetentionTime'].iloc[int(len(MSChromatogrammcut)/2)]
    #just an empty list for operation
    massspeclist = []
    #asking for range of mass spectra summation 
    integrationrange = getvalue(MSdf, 'Datapoints to left and right of maximum at retention time of ' + str(Rettimedet) + ' min to integrate mass spectra: ', int)
    #integrationrange = int(integrationrange)
    #starting loop for collecting all single massspectra in integrationrange in positive and negative directionto sum up 
    for item in range(-integrationrange, integrationrange, 1):
        #finding mass spectra in inputfile with scannumber equals to maxima(entry) + integrationrange  
        singlemassspec = MSdf[MSdf['ScanRange'] == scannumber]
        Massspectrum_at_Rettime = MSdf[MSdf['RetentionTime'] == Rettimedet]
        entry = Massspectrum_at_Rettime['Scannumber'].iat[1]
        msscan = entry + item
        singlemassspec = singlemassspec[singlemassspec['Scannumber'] == msscan].drop(['Scannumber', 'ScanRange', 'RetentionTime'], axis = 1)#.reset_index(drop = True)
        #round the masses to integer values
        singlemassspec['Mass'] = np.round(singlemassspec['Mass'], decimals = 0)
        #putting mass spectrum to a list and adding to dataframe end
        massspeclist.append(singlemassspec)
        massspec = pd.concat(massspeclist, ignore_index = True)
    #sorting mass sigmnals by entries of mass integer values
    massspec.sort_values(by=['Mass'])
    #summing up the Intensity values in massspectra having the same integer mass value
    massspecsum = massspec.groupby(['Mass']).sum()
    #normalizing mass spectra
    massspecsum = normalizeIntensity(massspecsum)
    #exporting the mass spectra as .xy-file
    timestamp = str(str(Rettimedet).split('.')[0]) + '_' + str(str(Rettimedet).split('.')[1])
    massspecsum.to_csv('ScanRange_'+ str(scannumber) + '_Massspectrum at RetTime ' + str(timestamp) + ' min.xy', sep = ' ')
    #Chromatogramm plot
    massspecplot(massspecsum.index, massspecsum['Intensity'], 'Mass Spectrum at Retention Time of ' +str(Rettimedet) + ' min', 'ScanRange_'+ str(scannumber) + '_Mass spectrum at RetTime ' + str(timestamp) + ' min.png', 0)          
    return

def maximamassscan(MSdf):
    for Nr in pd.unique(MSdf['ScanRange']):
        Range = MSdf[MSdf['ScanRange'] == Nr].drop('ScanRange', axis = 1)
        Range['Mass'] = Range['Mass'].round(0).astype(int)
        Range['RetentionTime'] = Range['RetentionTime']
        pivRng1 = Range.pivot_table(index = 'Mass', columns = 'RetentionTime', values = 'Intensity', aggfunc='sum')
        pivRng1.fillna(0, inplace = True)
        df = pivRng1.reset_index()
        massmin = Range['Mass'].min()
        massmax = Range['Mass'].max()
        masses = getlistofint(MSdf, 'What Masses ('+str(massmin)+'-' + str(massmax) + ') do you want extracted? (seperated by spaces): ') #input('What Masses ('+str(massmin)+'-' + str(massmax) + ') do you want extracted? (seperated by spaces): ').split()
        if len(masses) != 0:
            figures = getyninput(MSdf, 'Do you want later a graph of each mass with marked maxima?(y/n): ') #input('Do you want later a graph of each mass with marked maxima?(y/n): ')
        else:
            figures = 'n'
        counter = 0
        for mass in masses:
            SIR = df[df['Mass'] == int(mass)]
            SIR = SIR.transpose().reset_index()
            SIR.columns = ['Time', 'Mass '+ str(mass)]
            SIR = SIR.iloc[2:].set_index('Time')
            SIR.to_csv('SIR_Mass_'+ str(mass)+ '.xy') 
            range_ok = 'n'             
            while range_ok == 'n':
                #making graph so operator sees what data he is working on
                if counter == 0:
                    makegraph(SIR.index, SIR['Mass '+ str(mass)])
                #asking for left and right data limit
                if counter == 0:
                    xstart, xend = getlimits(MSdf, 'Retention time to start and end at (min max): ') 
                SIRcut = SIR.loc[float(xstart):float(xend)]
                #make graph
                makegraph(SIRcut.index, SIRcut['Mass '+ str(mass)])
                #checking if range is correct only first loop
                if counter == 0:    
                    range_ok = getyninput(MSdf, 'Is the range of data correct?(y/n): ') #input('Is the range of data correct?(y/n): ')
                else:
                    range_ok = 'y'
                if range_ok == 'y':
                    counter += 1
                #starting loop of maxima detection
            maxima_ok = 'n'
            while maxima_ok == 'n':
                #noise level and peakwidth input
                noise = getvalue(MSdf,'Noise level in % of Maximum (typically = 3): ',int) #int(input('Noise level in % of Maximum (typically = 3): ')) /100
                delta = getvalue(MSdf, 'Peakwidth in Number of Datapoints (typically = 6): ', int) #int(input('Peakwidth in Number of Datapoints (typically = 6): '))     
                #search for local maxima
                maxima_numbers1 = find_peaks(SIRcut['Mass '+ str(mass)], height = (SIRcut['Mass '+ str(mass)].max()*noise, SIRcut['Mass '+ str(mass)].max()), distance = delta)   
                #rearrange maxima data found by find_peaks
                maximadict = maxima_numbers1[-1]
                maxarray = maximadict['peak_heights']
                maxnum = pd.DataFrame(maxima_numbers1[0], columns = ['Scan'])
                #numbers only as integers
                [int(num) for num in maxarray]
                maxdf = pd.DataFrame(maxarray)
                maxnum['Intensity']= maxdf
                RSlist = list()
                #loop to find retention time of each maxima
                for entry in maxnum['Intensity']:
                    RSlist.append(SIRcut.loc[SIRcut['Mass '+ str(mass)] == entry])
                    RSdf = pd.concat(RSlist) 
                makegraphplusscattertofile(SIRcut.index, SIRcut['Mass '+ str(mass)], 'Mass '+ str(mass), RSdf.index, RSdf['Mass '+ str(mass)], 'Mass '+ str(mass), 'Time /min', 'Intensity /counts', False, False, False, False, 0, 1)
                maxima_ok = getyninput(MSdf, 'Are the maxima found correct?(y/n): ')#input('Are the maxima found correct?(y/n): ')
                if maxima_ok == 'y':
                    RSdf.to_csv('Maxima in Scanrange ' + str(Nr) + 'of mass ' + str(mass)+ '.xy')
                    if figures == 'y':
                        makegraphplusscattertofile(SIRcut.index, SIRcut['Mass '+ str(mass)], 'Mass '+ str(mass), RSdf.index, RSdf['Mass '+ str(mass)], 'Mass '+ str(mass), 'Time /min', 'Intensity /counts', SIRcut.index.min(), SIRcut.index.max(), SIRcut['Mass '+ str(mass)].min(), SIRcut['Mass '+ str(mass)].max(), 'Maxima in Scanrange ' + str(Nr) + 'of mass ' + str(mass)+ '.png', 1)
        massatret = getyninput(MSdf, 'Do you need the mass spectra at each maxima? (y/n): ')#input('Do you need the mass spectra at each maxima? (y/n): ')
        if massatret == 'y':
            getmassspectrabyRettime(RSdf, Range, Nr) 
    return

def getmassspectrabyRettime(RSdf, Range, Nr):
    integrationrange = getvalue(MSdf, 'Datapoints to left and right of each maxima to integrate mass spectra?: ', int) #int(input('Datapoints to left and right of each maxima to integrate mass spectra?: '))    
    #for every entry in list of found maxima....
    for time in RSdf.index:
        #get scannumber of each maxima
        Scan = Range.loc[Range['RetentionTime'] == time]['Scannumber'].iloc[1]
        #starting loop for collecting all single massspectra in integrationrange in positive and negative directionto sum up 
        massspectra = []
        # get massspectra left end right of range and make a list
        for item in range(-integrationrange, integrationrange, 1):
            entry = Scan + item
            massspec = Range.loc[Range['Scannumber'] == entry].drop(['Scannumber', 'RetentionTime'], axis = 1)
            massspectra.extend(massspec.values.astype(int).tolist())
        #take list of collected mass spectra and group by mass and sum up intensity
        massspectra = pd.DataFrame(massspectra, columns = ['Mass', 'Intensity']).groupby(['Mass']).sum() 
        #normalize intensity and limit to 999 for Massbank.eu
        massspectra['Intensity'] = massspectra['Intensity'] / massspectra['Intensity'].max() * 999
        #sort data so highes intensity is at beginning of list
        massspectra = massspectra.sort_values(by = ['Intensity'], ascending = False).round(0).astype(int)
        #get timestamp and create string
        timestamp = str(str(time).split('.')[0]) + '_' + str(str(time).split('.')[1])
        #output of integrated massspectra
        massspectra.to_csv('Massspectrum in Scanrange'+ str(Nr) +' at '+ str(timestamp) + '.xy', sep = ' ')
    return

def extractmassscan(MSdf):
    for Nr in pd.unique(MSdf['ScanRange']):
        Range = MSdf[MSdf['ScanRange'] == Nr].drop('ScanRange', axis = 1)
        Range['Mass'] = Range['Mass'].round(0).astype(int)
        Range['RetentionTime'] = Range['RetentionTime'].round(2)
        pivRng1 = Range.pivot_table(index = 'Mass', columns = 'RetentionTime', values = 'Intensity', aggfunc='sum')
        pivRng1.fillna(0, inplace = True)
        df = pivRng1.reset_index()
        # get min and max mass in scanrange
        massmin = Range['Mass'].min()
        massmax = Range['Mass'].max()
        #ask for masses oyu want to look at
        masses = input('What Masses ('+str(massmin)+'-' + str(massmax) + ') do you want extracted? (seperated by spaces): ').split()
        # ask if png files are needed
        if len(masses) != 0:
            figures = getyninput(MSdf, 'Do you want a graph of each mass?(y/n): ') #input('Do you want a graph of each mass?(y/n): ')
        else:
            figures = 'n'
        counter = 0
        for mass in masses:
            SIR = df[df['Mass'] == int(mass)]
            SIR = SIR.transpose().reset_index()
            SIR.columns = ['Time', 'Mass '+ str(mass)]
            SIR = SIR.iloc[2:].set_index('Time')
            SIR.to_csv('SIR_Mass_'+ str(mass)+ '.xy') 
            if str(figures) == 'y':
                range_ok = 'n'             
                while range_ok == 'n':
                    #making graph so operator sees what data he is working on
                    if counter == 0:
                        makegraph(SIR.index, SIR['Mass '+ str(mass)])
                    #asking for left and right data limit
                    if counter == 0:
                        xstart, xend = getlimits(MSdf, 'Retention time to start and end at (min max): ') #input('Retention time to start and end at (min max): ').split()
                    SIRcut = SIR.loc[float(xstart):float(xend)]
                    #make graph                        
                    makegraph(SIRcut.index, SIRcut['Mass '+ str(mass)])
                    #checking if range is correct only first loop
                    if counter == 0:    
                        range_ok = getyninput(MSdf, 'Is the range of data correct?(y/n): ') #input('Is the range of data correct?(y/n): ')
                    else:
                        range_ok = 'y'
                    if range_ok == 'y':
                        counter += 1
                        makegraphtofile(SIRcut.index, SIRcut['Mass '+ str(mass)], 'Mass '+ str(mass), 'Time /min', 'Intensity /counts', SIRcut.index.min(), SIRcut.index.max(), False, False, 'Chromatogram of Mass '+str(mass)+'.png', counter)                        
    return

def getchromandspectraB(MSdf):
    for Nr in pd.unique(MSdf['ScanRange']):
        Range = MSdf[MSdf['ScanRange'] == Nr].drop('ScanRange', axis = 1)              
        MSChromatogramm1 = Range.groupby(['RetentionTime'], sort = False).sum() 
        #removing Mass information
        MSChromatogramm1.pop('Mass')
        #generating Scannumber for operation 
        MSChromatogramm1['Scannumber'] = range(len(MSChromatogramm1))
        MSChromatogramm1.reset_index(inplace = True)
        #starting loop to limit data range to relevant region
        #export chromatogramm to .xy file
        ChrExp1 = MSChromatogramm1.drop('Scannumber', axis = 1).set_index('RetentionTime')
        ChrExp1.to_csv('Chromatogramm_ScanRange_'+ str(Nr) + '.xy')
        range_ok = 'n'
        while range_ok == 'n':
            #making graph so operator sees what data he is working on
            makegraph(MSChromatogramm1.index, MSChromatogramm1['Intensity'])
            #asking for left and right data limit
            xstartB, xendB  = getlimits(MSdf, 'Datapoints to start and end at in ScanRange '+str(Nr) + ' for determination of maxima (0 - ' + str(len(MSChromatogramm1)) + ')(min max): ')#input('Datapoints to start and end at for determination of maxima (0 - ' + str(len(MSChromatogramm1)) + ')(min max): ').split()
            xstartB = int(xstartB)
            xendB = int(xendB)
            #show graph of data range maxima finding routine works on
            makegraph(MSChromatogramm1['RetentionTime'].iloc[xstartB:xendB],MSChromatogramm1['Intensity'].iloc[xstartB:xendB])
            #checking if range is correct
            range_ok = input('Is the range of data points correct?(y/n): ')
        #cutting dataframe to limits
        MSChromatogramm1x = MSChromatogramm1.iloc[xstartB:xendB]    
        #starting loop to find maxima
        maxima_ok = 'n'
        while maxima_ok == 'n':
            #asking for noise level parameter given by operator
            noise = getvalue(MSdf, 'Noise level (typically = 3): ', int)* 1000 #int(input('Noise level (typically = 3): ')) *1000       
            #search for local maxima
            maxima_numbers1 = MSChromatogramm1x.iloc[(argrelextrema(MSChromatogramm1x.Intensity.values, np.greater_equal, order=noise, mode='clip'))]
            #plot Chromatogramm
            makegraphplusscattertofile(MSChromatogramm1x['RetentionTime'], MSChromatogramm1x['Intensity'], 'Intensity', maxima_numbers1['RetentionTime'], maxima_numbers1['Intensity'], 'Local Maxima', 'Retention Time /min', 'Intensity /counts', False, False, False, False, 'Chromatogram of Scan Range_'+ str(Nr) + '.png', 1)
            #output of found maxima data to file
            MaxExp = maxima_numbers1.loc[:, ('RetentionTime', 'Intensity')].set_index('RetentionTime')
            MaxExp.to_csv('Maxima in Chromatogramm Scanrange '+ str(Nr) + '.xy')
            maxima_ok = getyninput(MSdf, 'Are the maxima found correct?(y/n): ') #input('Are the maxima found correct?(y/n): ')
        #starting loop to find mass spectra of maxima in chromatogramm           
        for entry in maxima_numbers1['Scannumber']:        
            #just an empty list for operation
            massspeclist = []
            #looking up retentiontime of maximum
            RT = round(maxima_numbers1.at[entry,'RetentionTime'],3)  
            #asking for range of mass spectra summation 
            integrationrange = getvalue(MSdf, 'Datapoints to left and right of maximum at retention time of ' + str(RT) + ' min to integrate mass spectra: ', int) #int(input('Datapoints to left and right of maximum at retention time of ' + str(RT) + ' min to integrate mass spectra: '))
            #starting loop for collecting all single massspectra in integrationrange in positive and negative directionto sum up 
            for item in range(-integrationrange, integrationrange, 1):
                msscan = entry + item
                #finding mass spectra in inputfile with scannumber equals to maxima(entry) + integrationrange  
                singlemassspec = MSdf[MSdf['ScanRange'] == Nr]
                singlemassspec = singlemassspec[singlemassspec['Scannumber'] == msscan].drop(['Scannumber', 'ScanRange', 'RetentionTime'], axis = 1)#.reset_index(drop = True)
                #round the masses to integer values
                singlemassspec['Mass'] = np.round(singlemassspec['Mass'], decimals = 0)
                #putting mass spectrum to a list and adding to dataframe end
                massspeclist.append(singlemassspec)
                massspec = pd.concat(massspeclist, ignore_index = True)
            #sorting mass sigmnals by entries of mass integer values
            massspec.sort_values(by=['Mass'])
            #summing up the Intensity values in massspectra having the same integer mass value
            massspecsum = massspec.groupby(['Mass']).sum()
            #normalizing mass spectra
            massspecsum = normalizeIntensity(massspecsum)
            #exporting the mass spectra as .xy-file
            massspecsum.to_csv('ScanRange_'+ str(Nr) + '_Massspectrum at RetTime ' + str(int(RT)) + ' min.xy', sep = ' ')
            #Chromatogramm plot
            massspecplot(massspecsum.index, massspecsum['Intensity'], 'Mass Spectrum at Retention Time of ' +str(RT) + ' min', 'ScanRange_'+ str(Nr) + '_Mass spectrum at RetTime ' + str(int(RT)) + ' min.png', 0)          
    return

def getchromandspectraA(MSdf):
    for Nr in pd.unique(MSdf['ScanRange']):
        Range = MSdf[MSdf['ScanRange'] == Nr].drop('ScanRange', axis = 1)              
        MSChromatogramm1 = Range.groupby(['RetentionTime'], sort = False).sum() 
        #removing Mass information
        MSChromatogramm1.pop('Mass')
        #generating Scannumber for operation 
        MSChromatogramm1['Scannumber'] = range(len(MSChromatogramm1))
        MSChromatogramm1.reset_index(inplace = True)
        range_ok = 'n'
        while range_ok == 'n':
            #making graph so operator sees what data he is working on
            makegraph(MSChromatogramm1.index, MSChromatogramm1['Intensity'])
            #asking for left and right data limit
            xstartA, xendA  = getlimits(MSdf, 'Datapoints to start and end at in ScanRange '+str(Nr) + ' for determination of maxima (0 - ' + str(len(MSChromatogramm1)) + ')(min max): ') #input('Datapoints to start and end at for determination of maxima (0 - ' + str(len(MSChromatogramm1)) + ')(min max): ').split()
            xstartA = int(xstartA)
            xendA = int(xendA)
            #show graph of data range maxima finding routine works on
            makegraph(MSChromatogramm1['RetentionTime'].iloc[xstartA:xendA],MSChromatogramm1['Intensity'].iloc[xstartA:xendA])
            #checking if range is correct
            range_ok = getyninput(MSdf, 'Is the range of data points correct?(y/n): ') #input('Is the range of data points correct?(y/n): ')
            
        #cutting dataframe to limits
        MSChromatogramm1 = MSChromatogramm1.iloc[xstartA:xendA]   
        #finding minima for linear background subtraction
        minima = MSChromatogramm1.iloc[(argrelmin(MSChromatogramm1['Intensity'].to_numpy(), order=10, mode='clip'))]
        #linear fit of minima
        linear_model = np.poly1d(np.polyfit(minima['RetentionTime'],minima['Intensity'],1))
        #background subtraction loop
        for entry in MSChromatogramm1['RetentionTime']:
            MSChromatogramm1['IntensityBLC'] = MSChromatogramm1['Intensity'] - linear_model(MSChromatogramm1['RetentionTime'])
        #plot subtracted data 
        makegraph(MSChromatogramm1['RetentionTime'], MSChromatogramm1['IntensityBLC'])
        #starting loop of maxima detection
        maxima_ok = 'n'
        while maxima_ok == 'n':
            #noise level and peakwidth input
            noise = getvalue(MSdf, 'Noise level in % of Maximum (typically = 3): ', int)/100 #float(input('Noise level in % of Maximum (typically = 3): ')) /100
            delta = getvalue(MSdf, 'Peakwidth in Number of Datapoints (typically = 6): ', int) #int(input('Peakwidth in Number of Datapoints (typically = 6): '))     
            #search for local maxima
            maxima_numbers1 = find_peaks(MSChromatogramm1['IntensityBLC'].to_numpy(), height = (MSChromatogramm1['IntensityBLC'].max() *noise, MSChromatogramm1['IntensityBLC'].max()), distance = delta)   
            maximadict = maxima_numbers1[-1]
            maxarray = maximadict['peak_heights']
            maxnum = pd.DataFrame(maxima_numbers1[0], columns = ['Scan'])
            [int(num) for num in maxarray]
            maxdf = pd.DataFrame(maxarray)
            maxnum['Intensity']= maxdf
            RSlist = list()
            #loop to find retention time of each maxima
            for entry in maxnum['Scan']:
                RSlist.append(MSChromatogramm1['RetentionTime'].loc[MSChromatogramm1['Scannumber'] == entry + xstartA])
                RSdf = pd.concat(RSlist) 
            maxdf = MSChromatogramm1.merge(RSdf, left_on = 'Scannumber', right_index = True,  how = 'right', suffixes = ['', 'maximum'])            
            #start Chromatogramm plot
            makegraphplusscattertofile(MSChromatogramm1['RetentionTime'], MSChromatogramm1['Intensity'], 'Intensity', maxdf['RetentionTime'], maxdf['Intensity'], 'Local Maxima', 'Retention Time /min', 'Intensity /counts', MSChromatogramm1['RetentionTime'].min(), MSChromatogramm1['RetentionTime'].max(), MSChromatogramm1['Intensity'].min(), MSChromatogramm1['Intensity'].max(), 0, 1)                
            maxima_ok = getyninput(MSdf, 'Are the maxima found correct?(y/n): ') #input('Are the maxima found correct?(y/n): ')

        #start Chromatogramm plot
        makegraphplusscattertofile(MSChromatogramm1['RetentionTime'], MSChromatogramm1['Intensity'], 'Intensity', maxdf['RetentionTime'], maxdf['Intensity'], 'Local Maxima', 'Retention Time /min', 'Intensity /counts', MSChromatogramm1['RetentionTime'].min(), MSChromatogramm1['RetentionTime'].max(), MSChromatogramm1['Intensity'].min(), MSChromatogramm1['Intensity'].max(), 'Chromatogram_of_Scan_Range_'+ str(Nr) + '.png', 0)
        #output of found maxima data to file
        MaxExp = maxdf.loc[:, ('RetentionTime', 'Intensity')].set_index('RetentionTime')
        MaxExp.to_csv('Maxima in Chromatogramm Scanrange 1.xy')
        #output of chromatogramm to file
        ChrExp = MSChromatogramm1.loc[:, ('RetentionTime', 'Intensity')].set_index('RetentionTime')
        ChrExp.to_csv('Chromatogramm_ScanRange_'+ str(Nr) + '.xy')
        #loop to get massspectra of each maxima in chromatogramm
        maxcount = 1     
        for entry in maxdf['Scannumber']:
            massspeclist = []
            RT = maxdf[maxdf['Scannumber'] == entry].iat[0,0]
            timestamp = str(str(RT).split('.')[0]) + '_' + str(str(RT).split('.')[1])
            #asking for range of mass spectra summation 
            if maxcount ==1:
                integrationrange = getvalue(MSdf, 'Datapoints to left and right of each maximum to integrate mass spectra: ', int) #int(input('Datapoints to left and right of each maximum to integrate mass spectra: '))
            for item in range(-integrationrange, integrationrange, 1):
                msscan = entry + item
                #finding mass spectra in inputfile with scannumber equals to maxima(entry) + integrationrange  
                singlemassspec = MSdf[MSdf['ScanRange'] == Nr]
                singlemassspec = singlemassspec[singlemassspec['Scannumber'] == msscan].drop(['Scannumber', 'ScanRange', 'RetentionTime'], axis = 1)#.reset_index(drop = True)
                #round the masses to integer values
                singlemassspec['Mass'] = np.round(singlemassspec['Mass'], decimals = 0)
                #putting mass spectrum to a list and adding to dataframe end
                massspeclist.append(singlemassspec)                
                massspec = pd.concat(massspeclist, ignore_index = True)
            #sorting mass sigmnals by entries of mass integer values
            massspec.sort_values(by=['Mass'])
            #summing up the Intensity values in massspectra having the same integer mass value
            massspecsum = massspec.groupby(['Mass']).sum()
            #normalizing mass spectra
            massspecsum = normalizeIntensity(massspecsum)
            massspecsum.to_csv('ScanRange_'+ str(Nr) + '_maxima_'+str(maxcount)+'_Mass spectrum at RetTime ' + str(timestamp) + ' min.xy', sep = ' ')
            #start plot
            massspecplot(massspecsum.index, massspecsum['Intensity'], 'Mass Spectrum at Retention Time of ' +str(RT) + ' min', 'ScanRange_'+ str(Nr) + '_Mass spectrum at RetTime ' + str(timestamp) + ' min.png', 0)
            maxcount += 1
            plt.close()
    return

def makingheatmapandfile(MSdf):
    #splitting data into different Mass scan ranges 
    for Nr in pd.unique(MSdf['ScanRange']):
        Range = MSdf[MSdf['ScanRange'] == Nr].drop('ScanRange', axis = 1)
        #reduce data depth
        Range['Mass'] = Range['Mass'].round(0).astype(int)
        Range['RetentionTime'] = Range['RetentionTime'].round(2)
        pivRng1 = Range.pivot_table(index = 'Mass', columns = 'RetentionTime', values = 'Intensity', aggfunc='sum')
        pivRng1.fillna(0, inplace = True)
        #show Heatmap
        plt.figure(figsize = (16, 9))
        sns.heatmap(pivRng1,cmap="Greens", vmin = 5000,  vmax = 50000, fmt = 'd')
        plt.show()
        range_ok = 'n'
        while range_ok == 'n':
            #ask for limitations of graph
            xmin, xmax = getlimits(MSdf, 'Retention time minimum and maximum (min max): ') #input('Retention time minimum and maximum (min max): ').split()
            ymin, ymax = getlimits(MSdf, 'Mass range from minimum to maximum (min max): ') #input('Mass range from minimum to maximum (min max): ').split()
            Imin, Imax = getlimits(MSdf, 'Intensity minimum and maximum value (min max): ') #input('Intensity minimum and maximum value (min max): ').split()
            #cut the data to limitation
            wRange = Range[Range['Mass'] <= float(ymax)]
            wRange = wRange[wRange['Mass'] >= float(ymin)]
            wRange = wRange[wRange['RetentionTime'] <= float(xmax)]
            wRange = wRange[wRange['RetentionTime'] >= float(xmin)]
            #arrange data
            pivRng = wRange.pivot_table(index = 'Mass', columns = 'RetentionTime', values = 'Intensity', aggfunc='sum')
            pivRng.fillna(0, inplace = True)
            #export datatable
            tabexport1 = pivRng
            tabexport1.to_csv('heatmaptable_Scanrange_'+ str(Nr) + '.csv')
            #make new graph with limitations
            plt.close()
            plt.figure(figsize = (16, 9))
            heatmap = sns.heatmap(pivRng,cmap="Greens", yticklabels = int((float(ymax) - float(ymin))/ 10), xticklabels = 100, vmin = float(Imin), vmax = float(Imax), fmt = 'd', cbar_kws={'label': 'Intensity /counts'})
            plt.xlabel('Retention Time /min')
            plt.ylabel('Mass m/z')
            plt.show()
            #save heatmap to png
            fig = heatmap.get_figure()
            fig.savefig('heatmap_Scanrange_' + str(Nr) +  '.png', dpi = 600)
            range_ok = getyninput(MSdf, 'Graphic limits ok?(y/n): ') #input('Graphic limits ok?(y/n): ')
    return



#switch on following line when working in group TGGCMS directory otherwise place file into directory with .TXT-file
os.chdir('W:\#TG-GC-MS')
#asking for subfolders where the measurementfile is placed
subdir = input('Path of the dataset: ' + os.getcwd() + '\ ' )
if subdir != '':
    os.chdir(subdir)
#looking for .TXT-file in working directory
file = glob.glob('*.TXT')    
#prints out working directory path   
print('Used Path is: ' + os.getcwd() + '\ ' + str(file[0]))   
          
#function to remove emptylines in datafile
def nonblank_lines(f):
    for l in f:
            line = l.rstrip()
            if line:
                    yield line
#defining some variables for running the process of datafile import                    
RetTime = 0
ISDT = 0
Scan = 0
MSlist = []
#starting file opening routine
with open(str(file[0])) as MSdata:
    for line in nonblank_lines(MSdata):
        if 'FUNCTION' in line:
            Function = int(line.split(' ')[-1])
            RetTime = 0
            continue
        if 'CycleTime' in line:
            Cycletime = float(line.split(' ')[-1])
            continue
        if 'InterScanDelayTime' in line:
            ISDT = float(line.split(' ')[-1])
            continue
        if 'StartRetentionTime' in line:
            count =0
            continue
        if 'EndRetentionTime' in line:
            count =0
            continue
        if 'NumberofScans' in line:
            count =0
            continue
        if 'AcquisitionDataType' in line:
            count =0
            continue
        if 'Scan' in line:
            Scan = float(line.split('\t\t')[-1])
            continue
        if 'RetentionTime\t' in line:   
            RetTime = float(line.split('\t')[-1])
            continue
        if '$' in line:
            continue
        else:
            row = line.strip()
            Mass = float(row.split('\t')[0])
            Intensity = float(row.split('\t')[-1])
            MSline = [Function, Scan, RetTime, Mass, Intensity]
            MSlist.append(MSline)

#Making dataframe out of imported data
MSdf = pd.DataFrame(MSlist, columns = ['ScanRange', 'Scannumber', 'RetentionTime', 'Mass', 'Intensity'])
denoise = getyninput(MSdf, 'Do you want to work on denoised data (Memory Consuming)? (y/n): ') #input('Do you want to work on denoised data (Memory Consuming)? (y/n): ')
print(MSdf)
if denoise == 'y':
    MSdf = noisefilter(MSdf)
print(MSdf)


def main(MSdf):
    valid = 'n'
    while valid == 'n':
        print('Get Maxima, Chromatogramm and Massspectra of each peack in GC-MS measurement (narrow peaks) = 1')
        print('Get Maxima, Chromatogramm and  Masspectra of each peak in Online-MS measurement (braod peaks) = 2')
        print('Create Heatmap and table of GC-MS run = 3')
        print('Extract Single Mass Scans from TIC = 4')
        print('Extract Maxima of Single Mass Scans from TIC in GC-MS = 5')
        print('Have a Closer Look on Chromatogramm and Extract Single Mass Scans from TIC = 6')
        print('Generate List with Masses having high Signal/Noise ratio = 7')
        print('Exit = 8')
        prgchoice = getvalue(MSdf, 'What do you want to do with the data: ', int) #input('What do you want to do with the data: ')
        if prgchoice == 1:
            getchromandspectraA(MSdf)
            valid = getyninput(MSdf, 'done with the file? (y/n): ')#input('done with the file? (y/n): ')
        elif prgchoice == 2:   
            getchromandspectraB(MSdf)
            valid = getyninput(MSdf, 'done with the file? (y/n): ')#input('done with the file? (y/n): ')
        elif prgchoice == 3: 
            makingheatmapandfile(MSdf)
            valid = getyninput(MSdf, 'done with the file? (y/n): ')#input('done with the file? (y/n): ')
        elif prgchoice == 4:
            extractmassscan(MSdf)
            valid = getyninput(MSdf, 'done with the file? (y/n): ')#input('done with the file? (y/n): ')
        elif prgchoice == 5:
            maximamassscan(MSdf)
            valid = getyninput(MSdf, 'done with the file? (y/n): ')#input('done with the file? (y/n): ')
        elif prgchoice == 6:
            getsinglemassspectra(MSdf)
            valid = getyninput(MSdf, 'done with the file? (y/n): ')#input('done with the file? (y/n): ')
        elif prgchoice == 7:
             massestolookat(MSdf)
             valid = getyninput(MSdf, 'done with the file? (y/n): ')#input('done with the file? (y/n): ')
        elif prgchoice == 8:
             sys.exit()
        else:
            print('number not in program list')
            valid == 'n'
main(MSdf)