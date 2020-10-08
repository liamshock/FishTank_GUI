''' This file contains functions for
    (1) Loading the sLEAP, idTracker and segmentation information
    (2) Tracking
'''
import numpy as np
import glob
import h5py
import os
import sys



# ----- Section 1: loading data ---- #

class Filepaths(object):
    ''' A simple class for holding all of our filepaths for a specific splitdata

    -- Sample usage --
    # instantiate the object (see __init__ for examples of the args)
    filePaths = Filepaths(main_path, splitdata_name)
    # to get the oneBlob.h5 filepaths for each camera
    filePaths.oneBlobPaths
    '''

    def __init__(self, main_folder_path, splitdata_name):
        ''' Instantiate the object

        -- Args --
        main_folder_path: the 'top-level' folder in an experiment
                          e.g. '/home/liam/Data/FishTank20191211_115245'
        splitdata_name: the name of the splitdata folder that we want to work with
                          e.g. 'splitdata0020'
        '''
        self.main_path = main_folder_path
        self.splitdata_name = splitdata_name

        # the names of the folders for each camera
        self.cam_folder_names = ['D_xz', 'E_xy', 'F_yz']

        # get the paths for each folder containing the information
        self.splitdata_paths_for_cams = [os.path.join(self.main_path, cam_folder, self.splitdata_name)
                                         for cam_folder in self.cam_folder_names]

        # instantiate the lists for each quantity
        self.oneBlobPaths = []
        self.twoBlob1Paths = []
        self.twoBlob2Paths = []
        self.threeBlob1Paths = []
        self.threeBlob2Paths = []
        self.threeBlob3Paths = []
        self.infoPaths = []
        self.backPaths = []
        self.labelsPaths = []

        # for the matched instances
        self.oneBlob_MI_Paths = []
        self.twoBlob1_MI_Paths = []
        self.twoBlob2_MI_Paths = []
        self.threeBlob1_MI_Paths = []
        self.threeBlob2_MI_Paths = []
        self.threeBlob3_MI_Paths = []

        # loop over camera indices in the standard order recording the filepaths
        for i in range(3):
            self.oneBlobPaths.append(os.path.join(self.splitdata_paths_for_cams[i], 'oneBlob.h5'))
            self.twoBlob1Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'twoBlob1.h5'))
            self.twoBlob2Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'twoBlob2.h5'))
            self.threeBlob1Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'threeBlob1.h5'))
            self.threeBlob2Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'threeBlob2.h5'))
            self.threeBlob3Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'threeBlob3.h5'))
            self.infoPaths.append(os.path.join(self.splitdata_paths_for_cams[i], 'info.h5'))
            self.backPaths.append(os.path.join(self.splitdata_paths_for_cams[i], 'averaged_background.h5'))
            self.labelsPaths.append(os.path.join(self.splitdata_paths_for_cams[i], 'labels.h5'))

            self.oneBlob_MI_Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'oneBlob_matched_instances.h5'))
            self.twoBlob1_MI_Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'twoBlob1_matched_instances.h5'))
            self.twoBlob2_MI_Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'twoBlob2_matched_instances.h5'))
            self.threeBlob1_MI_Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'threeBlob1_matched_instances.h5'))
            self.threeBlob2_MI_Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'threeBlob2_matched_instances.h5'))
            self.threeBlob3_MI_Paths.append(os.path.join(self.splitdata_paths_for_cams[i], 'threeBlob3_matched_instances.h5'))

        return




class SplitdataManager(object):
    ''' A simple class for dealing with all of the splitdata folders for an
        experiment.

    -- Use cases --
    (a) find the filepaths for all camera views for a splitdata
    (b) find the global frame idxs of each splitdata
    (c) Given a global idx, find the splitdata folder and local index

    -- Attributes --
    main_path                      : the directory of the main experimental folder
    splitdata_paths                : a list over splitdatas, where each element is a
                                     list to the splitdata folder for each of the 3 cams
    start_stop_frames_for_splitdata: an array of shape (numSplitdatas, 2),
                                     where the 2nd dim gives the global frame idx for the
                                     first and last frame of each splitdata

    -- Public Methods --
    return_splitdata_folder_and_local_idx_for_global_frameIdx

    '''

    def __init__(self, main_folder_path):
        ''' Instantiate the object

        -- Args --
        main_folder_path: the 'top-level' folder in an experiment
                          e.g. '/path/to/Data/FishTank20191211_115245'
        '''
        self.main_path = main_folder_path

        # get the filepaths of all splitdata folders in experiment via the .mp4 names
        folderPaths = []
        for filepath in glob.glob(self.main_path + '**/*.mp4'):
            if '3panel' in filepath:
                continue
            folderPath = filepath.split('.')[0]
            folderPaths.append(folderPath)
        folderPaths.sort()
        self._folderPaths = folderPaths

        # find the 3 filepaths (one for each cam) for each splitdata
        first_splitdata_num = int(self._folderPaths[0].split(sep='/')[-1][-4:])
        last_splitdata_num = int(self._folderPaths[-1].split(sep='/')[-1][-4:])
        splitdata_paths = []
        # plus 1 to include last folder
        for idx in range(first_splitdata_num, last_splitdata_num+1):
            splitdata_idx_paths = []
            splitdata_idx_paths.append(self.main_path + 'D_xz' + '/' + 'splitdata' + str(idx).zfill(4))
            splitdata_idx_paths.append(self.main_path + 'E_xy' + '/' + 'splitdata' + str(idx).zfill(4))
            splitdata_idx_paths.append(self.main_path + 'F_yz' + '/' + 'splitdata' + str(idx).zfill(4))
            splitdata_paths.append(splitdata_idx_paths)
        self.splitdata_paths = splitdata_paths
        self._num_splitdatas = len(self.splitdata_paths)

        # get the global start and stop frame idx for each splitdata
        self.start_stop_frames_for_splitdata = self._get_start_stop_frames_for_splitdata_folders()


    def return_splitdata_folder_and_local_idx_for_global_frameIdx(self, global_frameIdx,
                                                                  return_splitdataIdx=False):
        ''' Given the index of a frame, return the name of the splitdata folder that it came from

        -- see also --
        get_start_stop_frames_for_splitdata_folders

        -- EX1 --
        If we started on splitdata0000, and record 6000 frames in each,

        splitdata_folder,local_idx = return_splitdata_folder_for_global_frameIdx(splitdata_paths,
                                                                                start_stop_frames_for_splitdata,
                                                                                10000)
        print(splitdata_folder)
        >> ['/work/StephensU/liam/experiments/1_male/D_xz/splitdata0001',
            '/work/StephensU/liam/experiments/1_male/E_xy/splitdata0001',
            '/work/StephensU/liam/experiments/1_male/F_yz/splitdata0001']
        print(local_idx)
        >> 4000
        '''
        # find the folder index
        for fld_idx in range(self._num_splitdatas):
            if global_frameIdx >= self.start_stop_frames_for_splitdata[fld_idx, 1]:
                continue
            elif global_frameIdx >= (self.start_stop_frames_for_splitdata[fld_idx, 0] and
                 global_frameIdx < self.start_stop_frames_for_splitdata[fld_idx, 1]):
                splitdata_folder_idx = fld_idx
                break
            else:
                continue

        # find the local frame number
        local_idx = global_frameIdx - self.start_stop_frames_for_splitdata[splitdata_folder_idx, 0]

        if return_splitdataIdx == True:
            return splitdata_folder_idx, local_idx
        else:
            return self.splitdata_paths[splitdata_folder_idx], local_idx


    def _get_start_stop_frames_for_splitdata_folders(self):
        ''' Return an array containing the global start and stop frame index for each
            splitdata folder.
        '''
        # initialize an array for global start and stop idxs for each splitdata folder
        start_stop_frames_for_splitdata = np.zeros((self._num_splitdatas, 2), dtype=int)

        # set a counter for global frame index, updated as we move through the folders
        running_total_idx = 0

        # loop over splitdata folders getting the indices
        #total_numFrames = []
        for splitdata_idx, splitdata_folder in enumerate(self.splitdata_paths):

            # first get the number of frames in this folder by examining frameCropType array shape
            folder = splitdata_folder[0] # XZ, XY or YZ - doesn't matter, we just want to find the number of frames
            info_path = os.path.join(folder, 'info.h5')
            with h5py.File(info_path, 'r') as hf:
                splitdata_frameCropType = hf['frameCropType'][:]
                splitdata_numFrames = splitdata_frameCropType.shape[0]

            # now get the frame indices
            splitdata_start = running_total_idx
            splitdata_stop = splitdata_start + splitdata_numFrames
            start_stop_frames_for_splitdata[splitdata_idx, 0] = splitdata_start
            start_stop_frames_for_splitdata[splitdata_idx, 1] = splitdata_stop

            # update the counter
            running_total_idx = splitdata_stop

        return start_stop_frames_for_splitdata

