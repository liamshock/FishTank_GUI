''' This file contains functions for
    (1) Loading the sLEAP, idTracker and segmentation information
    (2) Tracking
'''
import numpy as np
import glob
from joblib import load
import h5py
import os
import sys

# thomas was here
import cv2



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

            # thomas was here
            video_capture = cv2.VideoCapture(folder + '.mp4')
            splitdata_numFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()
            # info_path = os.path.join(folder + '.mp4')
            # with h5py.File(info_path, 'r') as hf:
            #     splitdata_frameCropType = hf['frameCropType'][:]
            #     splitdata_numFrames = splitdata_frameCropType.shape[0]

            # now get the frame indices
            splitdata_start = running_total_idx
            splitdata_stop = splitdata_start + splitdata_numFrames
            start_stop_frames_for_splitdata[splitdata_idx, 0] = splitdata_start
            start_stop_frames_for_splitdata[splitdata_idx, 1] = splitdata_stop

            # update the counter
            running_total_idx = splitdata_stop

        return start_stop_frames_for_splitdata









class Calibration(object):
    ''' A class for loading and using the calibrations for the cameras

    -- Methods --
    compute_imageCoord_triplet_from_XYZ
    compute_XYZ_from_imageCoord_triplet
    compute_XZ_imcoords_from_XY_YZ
    compute_XY_imcoords_from_XZ_YZ
    compute_YZ_imcoords_from_XZ_XY
    compute_point_correspondence_error

    '''

    def __init__(self, calibration_folder_path):
        ''' Instantiate the object

        -- args --
        calibration_folder_path: the path to a calibration folder, where the regressed functions
                                 have already been computed and saved

        '''
        # record the folder paths
        self.calibration_folder_path = calibration_folder_path
        self.python_calibration_folderPath = os.path.join(self.calibration_folder_path,
                                                          'python_calibration_models')

        # load the models and assign as attributes
        self._load_models()


    def _load_models(self):
        ''' Instantiate the regression object attributes:
            xyz_getter, imCoord_getter, xz_getter, xy_getter, yz_getter
        '''
        imCoords_to_XYZ_path = os.path.join(self.python_calibration_folderPath, 'imCoords_to_XYZ.joblib')
        XYZ_to_imCoords_path = os.path.join(self.python_calibration_folderPath, 'XYZ_to_imCoords.joblib')
        xy_yz_to_xz_path = os.path.join(self.python_calibration_folderPath, 'xy_yz_to_xz.joblib')
        xz_yz_to_xy_path = os.path.join(self.python_calibration_folderPath, 'xz_yz_to_xy.joblib')
        xz_xy_to_yz_path = os.path.join(self.python_calibration_folderPath, 'xz_xy_to_yz.joblib')

        self.xyz_getter = load(imCoords_to_XYZ_path)
        self.imCoord_getter = load(XYZ_to_imCoords_path)
        self.xz_getter = load(xy_yz_to_xz_path)
        self.xy_getter = load(xz_yz_to_xy_path)
        self.yz_getter = load(xz_xy_to_yz_path)
        return


    # ---- Main Methods ---- #

    def compute_imageCoord_triplet_from_XYZ(self, XYZ):
        ''' Predict the image coordinates in all 3 camera views of the
            3D point XYZ

        -- inputs --
        XYZ: array (3,), the position of a point in 3D

        -- returns --
        imCoords: array (3,2) of image coordinates in standard camera
                  order of XZ,XY,YZ
        '''
        imCoords = self.imCoord_getter.predict(XYZ.reshape(1,-1))
        imCoords = imCoords.reshape(3,2)
        return imCoords


    def compute_XYZ_from_imageCoord_triplet(self, imCoords):
        ''' Predict the XYZ position of the point given by the image
            coordinates from all 3 cameras

        -- Inputs --
        imCoords: array of shape (3,2)

        -- Outputs --
        XYZ: array of shape (3)

        '''
        XYZ = self.xyz_getter.predict(imCoords.reshape(-1,6))
        return XYZ


    def compute_XZ_imcoords_from_XY_YZ(self, xy_imCoord, yz_imCoord):
        ''' Given an image coordinate from both the XY and YZ views,
            compute the corresponding image coordinate from the XZ view

        -- args --
        xy_imCoord: image coordinate of shape (2,)
        yz_imCoord: image coordinate of shape (2,)

        -- returns --
        xz_imCoord: image coordinate of shape (2,)

        '''
        input_data = np.hstack((xy_imCoord, yz_imCoord)).reshape(1,4)
        xz_imCoord = self.xz_getter.predict(input_data)
        return xz_imCoord

    def compute_XY_imcoords_from_XZ_YZ(self, xz_imCoord, yz_imCoord):
        ''' Given an image coordinate from both the XZ and YZ views,
            compute the corresponding image coordinate from the XY view

        -- args --
        xz_imCoord: image coordinate of shape (2,)
        yz_imCoord: image coordinate of shape (2,)

        -- returns --
        xy_imCoord: image coordinate of shape (2,)
        '''
        # prepare the input for predictor, and predict the imcoord
        input_data = np.hstack((xz_imCoord, yz_imCoord)).reshape(1,4)
        xy_imCoord = self.xy_getter.predict(input_data)
        return xy_imCoord

    def compute_YZ_imcoords_from_XZ_XY(self, xz_imCoord, xy_imCoord):
        ''' Given an image coordinate from both the XY and YZ views,
            compute the corresponding image coordinate from the XZ view

        -- args --
        xz_imCoord: image coordinate of shape (2,)
        xy_imCoord: image coordinate of shape (2,)

        -- returns --
        yz_imCoord: image coordinate of shape (2,)

        '''
        # prepare the input for predictor, and predict the imcoord
        input_data = np.hstack((xz_imCoord, xy_imCoord)).reshape(1,4)
        yz_imCoord = self.yz_getter.predict(input_data)
        return yz_imCoord


    def compute_point_correspondence_error(self, camIdxs, imCoords_cam1, imCoords_cam2):
        ''' Compute the error of making a cross-camera association between these points

        -- args --
        camIdxs: a list denoting the cameras the imCoords args are coming from.
                 Has to be [0,1], [1,2], or [0, 2]
        imCoords_cam1: image coordinates from a camera
        imCoords_cam2: image coordinates from a different camera


        -- returns --
        error: a scalar error value for making this association
        '''
        # STEP 0: The error is NaN if either point is NaN
        if np.all(np.isnan(imCoords_cam1)) or np.all(np.isnan(imCoords_cam2)):
            return np.NaN

        # STEP 1: Compute the proposed image coordinate triplet
        if camIdxs == [0,1]:
            # derive YZ
            imCoords_cam3 = self.compute_YZ_imcoords_from_XZ_XY(imCoords_cam1, imCoords_cam2)
            proposed_imCoords = np.vstack((imCoords_cam1, imCoords_cam2, imCoords_cam3))
        elif camIdxs == [0, 2]:
            # derive XY
            imCoords_cam3 = self.compute_XY_imcoords_from_XZ_YZ(imCoords_cam1, imCoords_cam2)
            proposed_imCoords = np.vstack((imCoords_cam1, imCoords_cam3, imCoords_cam2))
        elif camIdxs == [1, 2]:
            # derive XZ
            imCoords_cam3 = self.compute_XZ_imcoords_from_XY_YZ(imCoords_cam1, imCoords_cam2)
            proposed_imCoords = np.vstack((imCoords_cam3, imCoords_cam1, imCoords_cam2))


        # STEP 2: Compute the errors

        # For each pairing of cameras, compute the 3rd cam image coordinate,
        # then compare this triplet to the proposed_imCoords, which act as truth
        # Note1: If this is a good pairing, then proposed_imCoords represent the same point in 3D
        # Note2: for one of these camera pairings test, we will get back an error of 0,
        #        since we did the same computation to compute proposed_coordinates.
        # Note3: to deal with note2, we define the error as the maximum of the 3 errors
        derived_xz = self.compute_XZ_imcoords_from_XY_YZ(proposed_imCoords[1], proposed_imCoords[2])
        image_coords_derXZ = np.vstack((derived_xz, proposed_imCoords[1], proposed_imCoords[2]))
        error_derXZ = np.linalg.norm(proposed_imCoords - image_coords_derXZ)

        derived_xy = self.compute_XY_imcoords_from_XZ_YZ(proposed_imCoords[0], proposed_imCoords[2])
        image_coords_derXY = np.vstack((proposed_imCoords[0], derived_xy, proposed_imCoords[2]))
        error_derXY = np.linalg.norm(proposed_imCoords - image_coords_derXY)

        derived_yz = self.compute_YZ_imcoords_from_XZ_XY(proposed_imCoords[0], proposed_imCoords[1])
        image_coords_derYZ = np.vstack((proposed_imCoords[0], proposed_imCoords[1], derived_yz))
        error_derYZ = np.linalg.norm(proposed_imCoords - image_coords_derYZ)

        errors = np.vstack((error_derXZ, error_derXY, error_derYZ))
        error = np.sum(errors)

        return error
