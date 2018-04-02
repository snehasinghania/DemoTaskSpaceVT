'''
GSE - Geocentric Solar Ecliptic
    X = Earth-Sun Line
    Z = Ecliptic North Pole
GSM - Geocentric Solar Magnetospheric
    X = Earth-Sun Line
    Z = Projection of dipole axis on GSE YZ plane
'''

import pickle
import numpy as np
import os 
import sys
tables = pickle.load(open("dst_tables.pkl", "rb"))
tables = np.array(tables)
#print (tables[0])

#-------------------------------------------------------------------------------------------
#contains the dst curves in the list for all the year from 1957 to 2013
dst_curves = []

for table in tables:
    #removing the day attribute from the string
    table = [unit[2:] for unit in table]
    curve = ' '.join(table)
    curve = curve.replace('-', ' -').split()
    dst_curves.append(curve)

dst_yearly = []
this_year = []
i = 0
for month in dst_curves:
    i += 1
    this_year += month
    if (i % 12 == 0):
        dst_yearly.append(this_year)
        this_year = []

#to get the data from 1963 (same year as that of imf data)
dst_yearly = dst_yearly[6:]


#-------------------------------------------------------------------------------------------

#The omni data is from 1963 till 2018 feb 28th (based on last update)
B = []
Bz_GSE = []
Bz_GSM = []

i = 0
for f in range(1963, 2014):
    fin = "omni_data/omni2_"+ str(f) + ".dat"
    fline = open(fin, 'r')
    #print (fin)
    #print (len(dst_yearly[i]))
    i += 1 
    data = fline.read().split("\n")
    b_val = []
    gse_val = []
    gsm_val = []
    #from each year extract only the required features
    for omni in data:
        if(omni != ''):
            omni_values = omni.split()
            b_val.append(omni_values[9])
            gse_val.append(omni_values[14])
            gsm_val.append(omni_values[16])   
    #store the feature curves in the global feature list containing all the curves
    B.append(b_val)
    Bz_GSE.append(gse_val)
    Bz_GSM.append(gsm_val)
    
#length of B[0] is 8784 = 24*366
#length of B is 56 = (20180-1963)+1

'''
print (len(B))
print (len(Bz_GSE))
print (len(Bz_GSM))

print (len(B[0]))
print (len(Bz_GSE[0]))
print (len(Bz_GSM[0])) 
'''       

#-------------------------------------------------------------------------------------------
#flatterned the data into a single list
dst_data = sum(dst_yearly, [])
imf_data = sum(B, [])
bz_gsm = sum(Bz_GSM, [])

#storing flattern data
fw = open("dst_data_complete.pkl", "wb")
pickle.dump(dst_data, fw)
fw.close() 

fw = open("imf_data_complete.pkl", "wb")
pickle.dump(imf_data, fw)
fw.close() 

fw = open("bz_gsm_complete.pkl", "wb")
pickle.dump(bz_gsm, fw)
fw.close()

#447072
print (len(dst_data))
print (len(imf_data))
print (len(bz_gsm))

#creating the input output pairs - the input is of length 5 and output is 1

dst_inp = []
imf_inp = []
bz_inp = []
dst_label = []
imf_label = []
bz_label = []

#creating the data points with input sequence length 10 and output is the next dst value
for i in range(0, len(dst_data)-11):
    dst_inp.append(dst_data[i:i+10])  
    imf_inp.append(imf_data[i:i+10]) 
    bz_inp.append(bz_gsm[i:i+10]) 
    
    dst_label.append(dst_data[i+11])
    #here the label is the dst_output - for model2
    imf_label.append(dst_data[i+11]) 
    bz_label.append(dst_data[i+11]) 
     
dst_inp =  np.array(dst_inp)
imf_inp =  np.array(imf_inp)
bz_inp =  np.array(bz_inp)

dst_label =  np.array(dst_label)
imf_label =  np.array(imf_label)
bz_label =  np.array(bz_label)

dst_inp = dst_inp.astype(np.float32)
imf_inp = imf_inp.astype(np.float32)
bz_inp = bz_inp.astype(np.float32)

dst_label = dst_label.astype(np.float32)
imf_label = imf_label.astype(np.float32)
bz_label = bz_label.astype(np.float32)

#print (dst_inp[1:10])
#print (imf_inp.shape)

#------------------------------------------------------------------------------------------
#saving the data and labels

fw = open("dst_inp.pkl", "wb")
pickle.dump(dst_inp, fw)
fw.close() 

fw = open("imf_inp.pkl", "wb")
pickle.dump(imf_inp, fw)
fw.close() 

fw = open("bz_inp.pkl", "wb")
pickle.dump(bz_inp, fw)
fw.close() 

fw = open("dst_label.pkl", "wb")
pickle.dump(dst_label, fw)
fw.close() 

fw = open("imf_label.pkl", "wb")
pickle.dump(imf_label, fw)
fw.close() 

fw = open("bz_label.pkl", "wb")
pickle.dump(bz_label, fw)
fw.close() 
