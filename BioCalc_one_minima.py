##############################################################################
### Script to calculate the cavity thicknes of a Biosensor for every pixel ###
##############################################################################

#############
### INPUT ###
#############


# enter all the folders which contain subfolders with the image files, 
# the folder with the image files MUST be in another folder (its name has to be entered here)
# e.g. you have 5 folders with images: 1,2,3,4,5 --> all these 5 folders have to be e.g. in a folder
# named "data". "data" is what you would enter in the list below. You can enter more then one folder 
# in this list (e.g. for 3 different long-time measurementes)

data = ['test']


# chose wavelength range and step-width

wave_start = 550    # [nm]
wave_end = 750      # [nm]


# enter a value to apply binning to run the calculation faster
binning = 2

# enter average deviation of experiment to simulation in nanometer, "1" is a good value to start

tolerance = 1

# define parameters for minima detection  

lookahead_min = 5 # something like peak width for the minima, 5 is a good value


# enter name of simulation_file, copy and paste the file name of the
# simulation file corresponding to your layer structure

sim_file = 'Sim_0.5Cr_15Au_50SiO2_Elastomer_RT601_15Au_500_760nm.txt'

# chose elastomer thickness range , the smaller the range the faster the program. If you are not sure, just take d_min = 1000, d_max = 19000

d_min= 2000 # [nm]
d_max= 11000 # [nm]


use_thickness_limits = True # Enter "True" if you want to do calculation with thickness limits and "False" if not. I recommend starting with "True" if you see sharpe edges you might need to switch to "Fals"

thickness_limit = 50 # [nm] enter the thickness limit (if thickness was found, next on will be: last_thickness +- thickness_limit)

area_avrg = 2 # this number defines how many pixel are considerd for an average to guess the new thickness, e.g.: 2 means that all pixels in a range of 2 rows above, two columns to the left and the right are averaged, that makes 12px, 1 --> 4px, 2 --> 12px, 3--> 24px

#############
# Smoothing #
#############

# Smoothing the wavelength images does not improve the all_imags that much, not really needed for the moment

# enter True if you want to enable this smoothing
x_y_smooth = False
# enter sigma for the gaussian smoothing
x_y_sigma = 0.5


############
# Plotting #
############

# parameters for printing
# color map is calculated like (mean_thickness - color_min, mean_thickness + color_max) 

color_min = 500
color_max = 500


############################
### END of INPUT SECTION ###
############################



#############################
#### start of the program ###
#############################

# load all the python modules needed
import cython_all_fit as Fit # import self-written cython code
import numpy as np
import time
import os 
import Image as im
from scipy import ndimage
import multiprocessing as mp
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian_filter
from skimage import transform

# change version of the release here which will be included in the results files
version = 'BioCalc 2.1.1'

t_a_start = time.time() # start timer for runtime measurement


# make sure that this part only runs as main process, not subprocess, thats important for the multiprocessing
if __name__ == '__main__':

    # for every folder with images in the data folder do:
    for data_folder in data:

        # make a list of the wavelength images in the folder
        folder_list = os.listdir(data_folder)

        # sort the list of folders
        folder_list.sort()

        # for all folders in the list do
        for folder in folder_list:
            # enter number of cpu cores, this has to be an integer number!
            # number of physical cores is a good start, but you can try with a larger as well

            # True for multiprocessing, False for single core (Windows), should work for linux and osx, but not for windows --> this cuts your images in blocks which are treated seperately
            multi_p = False   
            # number of processes you want to use, should not exeed the number of physical cores
            cores = 4

            lookahead_max = lookahead_min-1 # for the maxima --> should not be larger than lookahead_min

            # make wavelength list

            wave_step = 1       # [nm], this can be adjusted if you don't have an image evrey nanometer
            
            # create an empty list which will later contain all wavelengths, e.g. 550,551,552,...,750
            waves=[]

            # write wavelengths into the list
            waves=[wave_start + i*wave_step for i in xrange((wave_end-wave_start)/wave_step + 1)]

            ###################
            # read image data #
            ###################

            # make and sort list of file in the folder
            files = os.listdir(data_folder+'/'+folder)
            files.sort()
            
            # get size, bit-depth of the images
            for i in range(len(files)):
                # only consider files with the ending tiff, tif
                if files[i][-5:]=='.tiff' or files[i][-4:]=='.tif': 
                    Img=im.open(data_folder + '/'+folder + '/' + files[i])
                    break # stop the loop if an image was found


            Image_width = Img.size[0]/binning
            Image_height = Img.size[1]/binning

            # get colour and bit depth of image, this is important to know the range of the values (e.g. 8-bit is 0-255, 16-bit is 0-65535)
            Image_mode = Img.mode 
            if Image_mode == 'RGB' or Image_mode == 'P' or Image_mode == 'L':
                Image_bit = '8'
            elif Image_mode == 'I;16' or Image_mode == 'I;12' or Image_mode=='I;16B':
                Image_bit = '16'
            else:
                print 'Image mode is:', Img.mode
                Image_bit = '16B'
  
            # create an empty array which will contain all image data 
            all_images = np.zeros(((wave_end-wave_start)/wave_step + 1,Image_height,Image_width),np.uint16)
            # create another empty array to hold the data for the smoothing
            all_images_x_y_smooth = np.zeros(((wave_end-wave_start)/wave_step + 1,Image_height,Image_width),np.uint16)


            # read every image in folder and check if it is in the wavelength range --> write grey values into array
            
            # set a counter to check how many images have been processed
            counter=0
            # print some information for the user what is done
            print 'reading images from folder: ', folder
            if x_y_smooth == True:
                print 'smoothing of every wavelength image with sigma = ', x_y_sigma, ' before thickness calculation'
            else:
                print 'no smoothing of the wavelength images'    

            # start iterating over the files
            for i in xrange(len(files)):
                # only consider files with the ending tiff, tif
                if files[i][-5:]=='.tiff' or files[i][-4:]=='.tif':
                    # check if the current file is in the wavelength range the user specified at the beginning
                    if int(files[i][:3]) >= wave_start and int(files[i][:3]) <= wave_end:
                        #print files[i]
                        #print counter

                        # assign the current image name to a variable
                        Img = data_folder + '/'+folder + '/' + files[i]
                        # check if its 8-bit, convert, load
                        if Image_bit == '8':
                            Img = im.open(data_folder + '/'+folder + '/' + files[i])
                            Img = Img.convert('L')
                            all_images[counter]=transform.rescale(np.asarray(Img),1.0/binning,preserve_range=True).round().astype(np.uint16)

                        # not sure, but I think I used "imread" because it is more platform independent
                        else:
                            all_images[counter]=transform.rescale(imread(Img, as_grey=True),1.0/binning,preserve_range=True).round().astype(np.uint16)

                        # smoothing x-y direction
                        if x_y_smooth == True:
                            if Image_bit == '8':
                                Img_s = ndimage.gaussian_filter(Img, sigma=x_y_sigma)
                            else:
                                all_images_x_y_smooth[counter] = transform.rescale(gaussian_filter(imread(Img_s),sigma=x_y_sigma),1.0/binning,preserve_range=True)
                        counter+= 1

    ##################################
    ##### Section for smoothing ######
    ##################################

            # do you want to smooth in the direction of the wavelength? "z-direction" of the stack    
            lambda_smooth = False
            lambda_sigma = 1

            # if x_y smoothing was applied, use the smoothed array
            if x_y_smooth == True:
                all_images = all_images_x_y_smooth.copy()

            # if you want to smooth in wavelength direction, do the following:
            if lambda_smooth == True:
                print 'smoothing over wavelength with sigma = ', lambda_sigma
                for zeile in range(Image_height):
                    #print zeile
                    for spalte in range(Image_width):
                        if Image_bit == '8':
                            # smooth in wavelength direction
                            all_images[:,zeile,spalte] = ndimage.gaussian_filter1d(all_images[:,zeile,spalte],lambda_sigma)
                        else:
                            # smooth in wavelength direction
                            all_images[:,zeile,spalte] = gaussian_filter(all_images[:,zeile,spalte],sigma = lambda_sigma)
                        


    ########################################################################
    # Read the simulation file and get the minima for the thickness range #
    ########################################################################


            # read simulation file 
            print 'read simulated data'

            # open file
            p= open('Simulated_minima/' + sim_file,'r')

            # read the whole content of the file into a string
            string=p.read()

            # close the file
            p.close()

            # list which will contain, thickness, length of a set of wavelengths, starting position of the block --> this is all useful to have to improve speed
            thickness_len_pos=[]

            # current set of minima for one thickness
            minima_block=[]

            # list which contains all minima_blocks arrays one after another
            list_minima_blocks = []

            # variable which defines at which position a set of minima starts
            position = 0

            # read every line of the string
            for thisline in string.split('\n'):
                # check if the current line contains a thickness
                if ('\t' in thisline) == False and len(thisline) != 0:
                    thickness = thisline
                # check if this is a line with no thickness, but one is still in the thickness range
                if ('\t' in thisline) == True and int(thickness) >= d_min and int(thickness) <= d_max:
                    # split line into words
                    for word in thisline.split('\t'):
                        # check word if it is a wavelengt, only append it if its in the wavelength range +- lookahead
                        if len(word)<6 and float(word) >= wave_start + lookahead_min and float(word)<= wave_end - lookahead_min: # use only minima which are in the wave-range + lookahead_min
                            # append the wavelength to the current block of wavelengths
                            minima_block.append(float(word))

                # check if the line is empty and inside the thickness range
                if len(thisline) == 0 and int(thickness) >= d_min and int(thickness) <= d_max:

                    # append thickness, length of waveblock, position of block to a list
                    thickness_len_pos.append([np.uint16(thickness),np.uint16(len(minima_block)),np.uint32(position)]) # calculate length of the waveblock since it will be needed later

                    # append waveblock to an array
                    list_minima_blocks.append(np.array(minima_block,dtype=np.float))

                    # update the current starting position of the next waveblock
                    position += len(minima_block)
                    # clear the waveblock to write new wavelengths into it
                    minima_block=[]

            print 'perform the calculations'
            # get start time for simulation to check later how long it took
            t1 = time.time()

####################################################
## Find new delta to deal with different dynamics ##
####################################################
               
            # use different delta scaling for 8,12,16 bit, delta something like the peak height of the minima, it differs of course significantly for 8-bit images and 16-bit images, just because of the different range

            # for 16 or 12 bit images do the following
            if Image_bit == '16':
                # the proper delta shall be a 10th of the mean value of all images
                new_delta = int(all_images.mean()/10)
                # in case delta is larger than 7, take it
                if new_delta > 7:
                    delta = new_delta

            # for 8-bit images just take a delta of 7 as this is sufficient
            if Image_bit == '8':
                delta = 7    

            # calculate how much the delta should be varied in case no value is found
            delta_vary = int(delta/5-delta/20) # this has be found empirically to lead to a good fit of the minima
            print 'delta = ', delta

##############################################################
## Start calculations either with multiprocessing or single ##
##############################################################

            #########################
            # Multi-Core Processing #
            #########################

            if multi_p == True:

                def put_into_queue(start,ende,que,all_images, thickness_len_pos, waves, tolerance, lookahead_min, lookahead_max, delta,delta_vary, use_thickness_limits, thickness_limit,area_avrg):

                    # it is weird that the arguments of the function are not the same as the arguments which are used --> list_minima_blocks is missing. But it still works

                    que.put(Fit.c_Fit_Pixel(start,ende,all_images, thickness_len_pos, waves, tolerance, lookahead_min, lookahead_max, delta, delta_vary,list_minima_blocks, use_thickness_limits, thickness_limit,area_avrg)) # calls the C-Fit-function
                    #print 'Schlange ist satt'

                
                
                # devide the rows by the core-number --> to split it equally, assing the rest to the last process
                Zeile_Teil = Image_height/cores
                Zeile_Rest = Image_height%cores

                # start multiprocessing with queues

                Prozesse = []
                Queues = []

                for i in range(cores):
                    Queues.append(mp.Queue())

                # assign the data properly to the Processes
                for i in range(cores):
                    if i < cores-1:
                        Prozesse.append(mp.Process(target=put_into_queue,args=(i*Zeile_Teil,(i+1)*Zeile_Teil,Queues[i],all_images[:,(i*Zeile_Teil):((i+1)*Zeile_Teil),:], thickness_len_pos, waves, tolerance, lookahead_min, lookahead_max, delta,delta_vary, use_thickness_limits, thickness_limit,area_avrg)))
                    if i == cores-1:
                        Prozesse.append(mp.Process(target=put_into_queue,args=(i*Zeile_Teil,(i+1)*Zeile_Teil+Zeile_Rest,Queues[i],all_images[:,(i*Zeile_Teil):((i+1)*Zeile_Teil),:], thickness_len_pos, waves, tolerance, lookahead_min, lookahead_max, delta, delta_vary,use_thickness_limits, thickness_limit,area_avrg)))
                for i in range(cores):
                    Prozesse[i].start()
                    

                # initialise array for thicknesses
                result = np.ndarray((0,Image_width),dtype=np.uint16)

                for i in range(cores):
                    #print 'queuet', i
                    result = np.append(result,Queues[i].get(),axis=0)

                for i in range(cores):
                    #print 'joint', i
                    Prozesse[i].join()


            ##########################
            # Single-Core Processing #
            ##########################

            if multi_p == False:
                # define size of array
                # start row
                start = 0
                # last row
                ende = Image_height

                # call the external cython/c++ function with all the parameters
                result = Fit.c_Fit_Pixel(start,ende,all_images, thickness_len_pos, waves, tolerance, lookahead_min, lookahead_max, delta,delta_vary,list_minima_blocks, use_thickness_limits, thickness_limit,area_avrg)
            t2 = time.time()



            print t2-t1, 'seconds just for the calculation'



            ###########################
            # count not fitted values #
            ###########################

            not_fitted = Image_height*Image_width - np.count_nonzero(result)
            not_fitted_percent = 100.0/(Image_height*Image_width)*not_fitted
            print 'not fitted values',not_fitted
            print 'in percent:', not_fitted_percent

            ######################
            # Write data to file #
            ######################

            print 'write data to file'
            
            # generate a header with all parameters
            HEADER = time.strftime('Version = ' + version + '\n' + "%d.%m.%Y at %H:%M:%S")+'\n' + 'folder with data = ' + folder + '\n' + 'simulation file = ' + sim_file + '\n' + 'wave_start = '+str(wave_start) + '\n' + 'wave_end = ' + str(wave_end) + '\n' + 'lookahead_min = ' + str(lookahead_min) + '\n'  + 'lookahead_max = ' + str(lookahead_max) + '\n' + 'delta = ' + str(delta) + ' delta was varied +-'+str(delta_vary*5)+ '\n' + 'tolerance = ' + str(tolerance) + '\n' + 'thickness limits used: ' + str(use_thickness_limits) + '\n' + 'thickness limits: ' + str(thickness_limit) + '\n' +  'not fitted values: ' + str(not_fitted) + ', percentage of whole image: ' + str(not_fitted_percent)  + '\n'
            if x_y_smooth == True:
                HEADER+= 'x_y_smoothing done with sigma = ' + str(x_y_sigma) + '\n'
            if lambda_smooth == True:
                HEADER+= 'lambda smoothing done with sigma = ' + str(lambda_sigma) + '\n'

            HEADER+= '\n'

            # If filename should be different for calculations with smoothing
            # if x_y_smooth == True and lambda_smooth == False:
            #     file_name = data_folder + '_' +folder + time.strftime("_%Y%m%d_%H%M%S")+'_x_y_sigma_' + str(x_y_sigma) + '_smoothed.txt'
                
            # if lambda_smooth == True and x_y_smooth == False:
            #     file_name = data_folder + '_' + folder + time.strftime("_%Y%m%d_%H%M%S")+'_lambda_sigma_' + str(lambda_sigma) + '_smoothed.txt'

            # if lambda_smooth == True and x_y_smooth == True:
            #     file_name = data_folder + '_' + folder +  time.strftime("_%Y%m%d_%H%M%S")+'_x_y_sigma_' + str(x_y_sigma) + '_lambda_sigma_' +str(lambda_sigma) + '_smoothed.txt'

            # if lambda_smooth == False and x_y_smooth == False: 
            #     file_name = data_folder + '_' + folder + time.strftime("_%Y%m%d_%H%M%S")+'.txt'

            # generate a useful filename
            file_name = data_folder + '_' + folder + time.strftime("_%Y%m%d_%H%M%S")+'.txt'

            # use numpy function to save array to file, '0' and not '-' are used for missing values
            np.savetxt(data_folder + '/' + file_name,result,fmt='%d')#,header=HEADER )

            ######################################################################
            # script to replace a certain string or string combination in a file #
            ######################################################################

            #t_replace_1 = time.time()

            # do it twice because 0 0 would not be - - but - 0 since the empty character before the second 0 has already been read
            for i in range(2):
                p = open(data_folder + '/' + file_name,'r')

                string = p.read()

                p.close()

                string = string.replace(' 0 ', ' - ')
                string = string.replace('\n'+'0 ', '\n'+'- ')
                string = string.replace(' 0'+'\n', ' -'+'\n')

                p = open(data_folder + '/' + file_name,'w')

                p.write(string)

                p.close()
            #print (time.time()-t_replace_1), ' seconds for replaceing 0 with -'
            print (time.time()-t_a_start), ' seconds for the whole program'


            ##############
            # plot datta #
            ##############

            # make a new figure
            plt.figure(folder)
            # create plot of the results
            plt.imshow(result)
            # set the color scale to the limits provided
            plt.clim(result.mean()-color_min,result.mean()+color_max)
            # plot a color bar
            plt.colorbar()

            # remove "#" to show the plot after the calculation
            #plt.show()

