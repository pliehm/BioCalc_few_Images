#cython: boundscheck=False
#cython: profile=False
#cython: cdivision = True

import numpy as np
cimport numpy as np
cimport cython


# define types for cython, not sure if this is really necessary, but I found it in one of the examples I used to improve the speed
DTYPE1 = np.uint16
ctypedef np.uint16_t DTYPE_t
DTYPE2 = np.float64
ctypedef np.float64_t DTYPE_t2
DTYPE3 = np.uint32
ctypedef np.uint32_t DTYPE_t3

# fit simulated and experimental minima to calculate thickness

# define a faster function to calculate the absolute value of a number
cdef float _abs(float a): return a if a>=0 else -a

###################################################
# Function to fit thickness with thickness limits #
###################################################

# almost the same function as above, the only difference is that only thicknesses within the thickness_limit are considered for the fitting, this improves speed and avoids jumps between cavities
@cython.boundscheck(False)
cdef Fit_limit(np.ndarray[DTYPE_t, ndim=1] thickness,np.ndarray[DTYPE_t3, ndim=1] array_thickness_len_pos,np.ndarray[DTYPE_t, ndim=1] array_length_block, np.ndarray[double,ndim=1] exp_waves, float tolerance,np.ndarray[double,ndim=1] sim_minima_blocks, last_index, thickness_list, thickness_limit):

# cython type definition of the used objects
 
    cdef list sim_min_waves = [[1000],[1000]] # dummy values to have two lists
    cdef unsigned short i, k, min_thickness_i, max_thickness_i
    cdef unsigned short L_exp_waves = len(exp_waves) # too not use the len() function too often
    cdef float summe=0
    cdef unsigned int counter = 0
    cdef unsigned int position, len_block 
    cdef unsigned int breaker = 0

    if L_exp_waves > 0: # first test if there is a minimum
       # for waves in s_waves_arrays:
        min_thickness_i = last_index-thickness_limit #
        max_thickness_i = last_index+thickness_limit #
        for i in range(min_thickness_i,max_thickness_i): # do the following calculations for every simulated thickness
            position = array_thickness_len_pos[i]
            len_block = array_length_block[i]
            if array_length_block[i] == L_exp_waves: # case for equal minimas (exp, sim)
                breaker = 0
                summe=0
                # perform something like least-square with every exp-wavelength 
                for k in xrange(L_exp_waves): 
                    summe+=_abs(sim_minima_blocks[position+k]-exp_waves[k])
                    if summe/L_exp_waves > tolerance:
                        breaker = 1
                        break
                # append the thickness and error to a list
                if breaker == 1:
                    continue
                sim_min_waves[0].append(thickness[i])
                sim_min_waves[1].append(summe/float(L_exp_waves))
                continue

            # do the same if number of exp and sim is not  equal
            if array_length_block[i] == (L_exp_waves + 1):
                breaker=0
                summe=0
                # check if the first elements (exp and sim) or the last tow are not matching
                #if _abs(sim_minima_blocks[position] - exp_waves[0]) > _abs(sim_minima_blocks[position+thickness_len_pos[i][1]-1]-exp_waves[-1]):
                if _abs(sim_minima_blocks[position] - exp_waves[0]) > _abs(sim_minima_blocks[position+len_block-1]-exp_waves[-1]):
                    for k in xrange(L_exp_waves):
                        summe+= _abs(sim_minima_blocks[position+k+1]-exp_waves[k])
                        if summe/L_exp_waves > tolerance:
                            breaker = 1
                            break
                    if breaker == 1:
                        continue
                    sim_min_waves[0].append(thickness[i])
                    sim_min_waves[1].append(summe/float(L_exp_waves))
                    continue
                else:
                    for k in xrange(L_exp_waves):
                        summe+= _abs(sim_minima_blocks[position+k]-exp_waves[k])
                        if summe/L_exp_waves > tolerance:
                            breaker = 1
                            break
                    if breaker == 1:
                        continue
                    sim_min_waves[0].append(thickness[i])
                    sim_min_waves[1].append(summe/float(L_exp_waves))
                    continue

            if (array_length_block[i] == (L_exp_waves - 1)) and ((L_exp_waves-1)>0):
                breaker = 0
                summe=0
                #sim_waves_part_part = sim_waves_part[2]
                if _abs(sim_minima_blocks[position] - exp_waves[0]) > _abs(sim_minima_blocks[position+len_block-1]-exp_waves[-1]):
                    for k in xrange(array_length_block[i]):
                        summe+= _abs(sim_minima_blocks[position+k]-exp_waves[k+1])
                        if summe/(L_exp_waves-1) > tolerance:
                            breaker = 0
                            break
                    if breaker == 1:
                        continue
                    sim_min_waves[0].append(thickness[i])
                    sim_min_waves[1].append(summe/float(L_exp_waves))
                    continue
                else:
                    for k in xrange(array_length_block[i]):
                        summe+= _abs(sim_minima_blocks[position+k]-exp_waves[k])
                        if summe/(L_exp_waves-1) > tolerance:
                            breaker = 1
                            break
                    if breaker == 1:
                        continue
                    sim_min_waves[0].append(thickness[i])
                    sim_min_waves[1].append(summe/float(L_exp_waves))
                    continue

# return the thickness with minimum value
        if  len(sim_min_waves[0])>1 and (min(sim_min_waves[1]) < tolerance):
            return sim_min_waves[0][sim_min_waves[1].index(min(sim_min_waves[1]))], thickness_list.index(sim_min_waves[0][sim_min_waves[1].index(min(sim_min_waves[1]))]), sim_min_waves
        else: 
            return 0, 0,0

    else:
        return 0, 0,0

#########################################################
# function to get minima of the intensity profile array #
#########################################################

cdef list peakdetect(y_axis, x_axis = None, unsigned short lookahead_min=5, unsigned short lookahead_max=3, unsigned short delta = 0):
    
    # define output container
    #cdef list max_peaks=[]
    cdef list min_peaks = []
    cdef list dump = [] # used to pop the first hit which almost always is false
    cdef list y_axis_list = y_axis.tolist() # convert array to list, min() is faster for list

    # check input data --> this makes the algorithm 5 times slower
    #x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis) 
    
    # store data length for later use
    cdef unsigned int length = len(y_axis)

    #perform some checks
    #if lookahead < 1:
    #    raise ValueError, "Lookahead must be '1' or above in value"
    #if not (np.isscalar(delta) and delta >= 0):
    #    raise ValueError, "delta must be a positive number"

    #maxima and minima candidates are temporarily stored in 
    #mx and mn respectively
    cdef int mn = 100000
    cdef int mx = -100000
    cdef unsigned short x,y, index

    for index, (x,y) in enumerate(zip(x_axis[:-lookahead_min], y_axis[:-lookahead_min])):
        
        if y > mx:
            mx = y
            mxpos = x

        if y < mn:
            mn = y
            mnpos = x

        #### look for max ####
        
        if y < mx-delta and mx != 100000:
            #Maxima peak candidate found
            # lool ahead in signal to ensure that this is a peak and not jitter
            if max(y_axis_list[index:index+lookahead_max]) < mx:
                #max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                
                mx = 100000
                mn = 100000
                if index+lookahead_min >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue

        #### look for min ####    
        
        if y > mn+delta and mn != -100000:
            #Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if min(y_axis_list[index:index+lookahead_min]) > mn:
                min_peaks.append(mnpos)
                dump.append(False)
                #set algorithm to only find maximum now
                mn = -100000
                mx = -100000
                if index+lookahead_min >= length:
                    #end is within lookahead no more peaks can be found
                    break


    #Remove the false hit on the first value of the y_axis
    if len(dump)>0:
        if not dump[0]:
            min_peaks.pop(0)
        else:
            pass    
    
        #no peaks were found, should the function return empty lists?
    
    return min_peaks

    

####################################################################################
# Main function which calls the minima finding algorithm and the thickness fitting #
####################################################################################


# the following parameters are passed to the function:
# start wavelength, end wavelength, all images, list of thickness/blocklength/position,list of waves, tolerance, lookahead_min, lookahead_max, delta, delta variations, minima blocks, thickness limits in use?, thickness limit, 
def c_Fit_Pixel(unsigned int start,unsigned int ende, np.ndarray[DTYPE_t, ndim=3] data, list thickness_len_pos, list waves, float tolerance, unsigned short lookahead_min,unsigned short lookahead_max, unsigned short delta, unsigned short delta_vary, list list_minima_blocks, use_thickness_limits, unsigned int thickness_limit, unsigned short area_avrg, np.ndarray[DTYPE_t2, ndim=2] init_map):

    ########################################
    # definition of all the variable types #
    ########################################

    cdef unsigned int Image_width = len(data[0][0])
    cdef np.ndarray[DTYPE_t2, ndim=2] thickness_ready = np.zeros((ende-start,Image_width),np.float64 )
    cdef unsigned short column, row, column_c, row_c
    cdef np.ndarray[DTYPE_t, ndim=1] intensity
    cdef np.ndarray[double,ndim=1] minima_exp
    cdef unsigned int counter=start, 
    cdef np.ndarray[double,ndim=1] sim_minima_blocks
    cdef np.ndarray[DTYPE_t,ndim=1] array_length_block, thickness
    cdef np.ndarray[DTYPE_t3,ndim=1] array_thickness_len_pos
    cdef unsigned int  last_index = 0, limit_counter = 0
    cdef float last_thickness = 0, current_thickness = 0
    cdef list a = [] # dummy list
    cdef list thickness_list = []

    # make a list which contains all the thicknesses (maybe one could also use the array below?)
    for i in range(len(thickness_len_pos)):
        thickness_list.append(int(thickness_len_pos[i][0]))

    # make an array which contains all minima, but not seperated as before
    for block in list_minima_blocks:
        for i in range(len(block)):
            a.append(block[i])
    sim_minima_blocks = np.array(a,dtype=np.float)

    # make arrays of positions, lengths, thickness
    array_thickness_len_pos = np.array(zip(*thickness_len_pos)[2],dtype=np.uint32)
    array_length_block = np.array(zip(*thickness_len_pos)[1],dtype=np.uint16)
    thickness = np.array(zip(*thickness_len_pos)[0],dtype=np.uint16)

    print 'x ', len(data) 
    print 'y ', len(data[0])
    print 'z ', len(data[0][0])


    ######################################### 
    # do calculations with thickness limits #
    #########################################

    if use_thickness_limits:
        #print "using thickness limit: ", thickness_limit

        for row in range(len(data[0])):
            print counter
            counter+=1
            for column in xrange(Image_width):
                #last_thickness = 0
                last_index = 0
                limit_counter = 0
                last_thickness = init_map[row][column]
                # write line to find last_index 
                if (last_thickness != 0) and (not np.isnan(last_thickness)):

                    last_index = thickness_list.index(last_thickness) # could raise an error if 0
                # get array with the intensity profile for the current pixel
                intensity = data[:,row, column]
                # find the minima in the profile
                minima_exp = np.array(peakdetect(intensity, waves, lookahead_min,lookahead_max, delta),dtype=np.float)


                # start calculations with limits
                if (last_thickness != 0) and (not np.isnan(last_thickness)) and (last_index > 0):
                    
                    last_thickness, last_index, sim_min_waves = (Fit_limit(thickness,array_thickness_len_pos, array_length_block, minima_exp,tolerance,sim_minima_blocks,last_index,thickness_list, thickness_limit))

                thickness_ready[row][column]=last_thickness

    # feed the fitted thicknesses back to the main program
    return thickness_ready, sim_min_waves