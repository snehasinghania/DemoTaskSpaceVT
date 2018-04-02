from collections import defaultdict
import datetime as dt
import numpy as np
import davitpy.pydarn.sdio as sdio
import davitpy.pydarn.proc.fov.update_backscatter as ub
import pickle


def get_data(start_time, end_time, radar_name):
    #establish a data pipeline to the VT server and get pointers to beamData class objects 
    data_ptr = sdio.radDataOpen(start_time,radar_name,end_time)   
    #process beam Data using the update_backscatter function
    #updates the propagation path, elevation, backscatter type, 
    #and origin field-of-view (FoV) for all backscatter observations in each beam
    processed_beams = ub.update_backscatter(data_ptr)
    return processed_beams


def save_data(start_time, end_time, radar_name):  
    processed_beams = get_data(start_time, end_time, radar_name)        
    data = defaultdict(list)
    #getting attributes: range gate, power, doppler velocity, spectral width, elevation for future clusterig
    for key in processed_beams.keys():
        for i in range(len(processed_beams[key])):
            beam = processed_beams[key][i]
            #discard beam indices which have elevation angle equal to nan
            good_indices = np.where(np.isfinite(np.array(beam.fit.fovelv)))
            #since we have used a defaultdict list data structure, we can use += instead of append
            #which will directly give a flat list of attribute values for clustering 
            data["gate"] += (np.array(beam.fit.slist)[good_indices]).tolist() 
            data["power"] += (np.array(beam.fit.p_l)[good_indices]).tolist()           
            data["velocity"] += (np.array(beam.fit.v)[good_indices]).tolist()            
            data["width"] += (np.array(beam.fit.w_l)[good_indices]).tolist()            
            data["elevation"] += (np.array(beam.fit.fovelv)[good_indices]).tolist()                        
    
    '''
    #note: all attribute lists should be of the same length
    #in case of future errors, do a data consistency check    
    #data consistency check
    l = len(data[data.keys()[0]])
    for key in data.keys():
        if (len(data[key]) != l):
            print "data size error"
            break
    '''
    with open("radar_data_" + str(str(start_time).split(' ')[0]) + "_" + radar_name + ".pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()    
    
    
if __name__ == "__main__":
    #chosen the same dates used in radDataRead.ipynb in Git repository since assuming reliable data on this date
    start_time = dt.datetime(2011,1,1,0)
    end_time = dt.datetime(2011,1,1,1)
    '''
    Collect data from the Saskatoon radar since A large proportion of the 
    backscatter measured by the Saskatoon radar have reliable elevation angle  
    measurements  with  a  good  coverage in  range,  frequency,  
    and  magnetic  local  time  (MLT). Source: Chisham et. al., 2008
    https://www.ann-geophys.net/26/823/2008/angeo-26-823-2008.pdf
    '''
    radar_name = 'sas'
    #save the data for future use - we don't want to increase the load on the server
    #by asking for data for each analysis run    
    save_data(start_time, end_time, radar_name)
