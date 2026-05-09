#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import logging
import datetime
import numpy as np

import isceobj
import isceobj.Sensor.MultiMode as MultiMode
from isceobj.Planet.Planet import Planet
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxRdr
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxGeo
from isceobj.Alos2Proc.Alos2ProcPublic import modeProcParDict

logger = logging.getLogger('isce.alos2insar.runPreprocessor')


# ---- ALOS-2 / ALOS-4 LED/IMG filename parsing helpers -----------------------
#
# ALOS-2 LED:  LED-ALOS2{orbit9d}-{YYMMDD}-{mode3c}...
#   e.g.  LED-ALOS2211083630-180420-FBDR1.1__D
# ALOS-4 LED:  LED-ALOS4{orbit7d}{YYMMDD}{mode3c}-{passdir2+frame4}-1.1__-
#   e.g.  LED-ALOS41370060250228FWD-RA0107-1.1__-
# ALOS-4 IMG:  IMG-{pol}-ALOS4{orbit7d}{YYMMDD}{mode3c}-{passdir2+frame4}-1.1__-
#   e.g.  IMG-HH-ALOS41370060250228FWD-RA0107-1.1__-

def _get_mode_from_led(led_base):
    """Return the 3-character operation-mode code from a LED filename.

    ALOS-4 Scene ID: ALOS4{path3}{frame4}{YYMMDD6}{mode3}
      'ALOS4'=5 + path=3 + frame=4 + date=6 = 18 → mode at [18:21]
    ALOS-2: last component of dash-separated name, first 3 chars.
    """
    if led_base.startswith('LED-ALOS4'):
        return led_base.split('-')[1][18:21]
    else:
        return led_base.split('-')[-1][0:3]


def _get_date_from_led(led_base):
    """Return the YYMMDD date string from a LED filename.

    ALOS-4 Scene ID: ALOS4{path3}{frame4}{YYMMDD6}{mode3}
      date at [5+3+4 : 5+3+4+6] = [12:18]
    ALOS-2: third dash-separated field.
    """
    if led_base.startswith('LED-ALOS4'):
        return led_base.split('-')[1][12:18]
    else:
        return led_base.split('-')[2]


def _get_frame_from_led(led_base):
    """Return the 4-digit frame number string from a LED filename.

    ALOS-4 Scene ID: ALOS4{path3}{frame4}{YYMMDD6}{mode3}
      frame at [5+3 : 5+3+4] = [8:12]  e.g. '2900'
      NOTE: split('-')[2] is the Product ID (e.g. 'RD0107') which encodes
      look/pass/beam — NOT the frame number.
    ALOS-2: last 4 chars of the orbit+frame sensor-id field.
    """
    if led_base.startswith('LED-ALOS4'):
        return led_base.split('-')[1][8:12]
    else:
        return led_base.split('-')[1][-4:]


def _get_led_files(data_dir, frame):
    """Return sorted list of LED files for the given frame in data_dir.

    ALOS-4: frame is embedded in the Scene ID as ALOS4{path3}{frame4}...
      Use '???' wildcard for the 3-char path field, then literal frame.
    """
    alos2 = glob.glob(os.path.join(data_dir, 'LED-ALOS2*{}-*-*'.format(frame)))
    alos4 = glob.glob(os.path.join(data_dir, 'LED-ALOS4???{}*'.format(frame)))
    return sorted(alos2 + alos4)


def _get_img_files(data_dir, pol, frame, swath=None):
    """Return sorted list of IMG files for pol+frame (and optionally swath) in data_dir.

    ALOS-4: frame embedded in Scene ID (ALOS4{path3}{frame4}...); use '???' for path.
    For ScanSAR swath, ALOS-4 appends '-F{N}' or '-B{N}' at end of filename.
    """
    if swath is not None:
        alos2 = glob.glob(os.path.join(data_dir, 'IMG-{}-ALOS2*{}-*-*-F{}'.format(pol, frame, swath)))
        alos4 = glob.glob(os.path.join(data_dir, 'IMG-{}-ALOS4???{}*-F{}'.format(pol, frame, swath)))
    else:
        alos2 = glob.glob(os.path.join(data_dir, 'IMG-{}-ALOS2*{}-*-*'.format(pol, frame)))
        alos4 = glob.glob(os.path.join(data_dir, 'IMG-{}-ALOS4???{}*'.format(pol, frame)))
    return sorted(alos2 + alos4)

# ------------------------------------------------------------------------------


def runPreprocessor(self):
    '''Extract images.
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)


    #find files
    #actually no need to use absolute path any longer, since we are able to find file from vrt now. 27-JAN-2020, CRL.
    #denseoffset may still need absolute path when making links
    self.referenceDir = os.path.abspath(self.referenceDir)
    self.secondaryDir = os.path.abspath(self.secondaryDir)

    ledFilesReference = sorted(glob.glob(os.path.join(self.referenceDir, 'LED-ALOS2*-*-*')) +
                               glob.glob(os.path.join(self.referenceDir, 'LED-ALOS4*-*-*')))
    imgFilesReference = sorted(glob.glob(os.path.join(self.referenceDir, 'IMG-{}-ALOS2*-*-*'.format(self.referencePolarization.upper()))) +
                               glob.glob(os.path.join(self.referenceDir, 'IMG-{}-ALOS4*-*-*'.format(self.referencePolarization.upper()))))

    ledFilesSecondary = sorted(glob.glob(os.path.join(self.secondaryDir, 'LED-ALOS2*-*-*')) +
                               glob.glob(os.path.join(self.secondaryDir, 'LED-ALOS4*-*-*')))
    imgFilesSecondary = sorted(glob.glob(os.path.join(self.secondaryDir, 'IMG-{}-ALOS2*-*-*'.format(self.secondaryPolarization.upper()))) +
                               glob.glob(os.path.join(self.secondaryDir, 'IMG-{}-ALOS4*-*-*'.format(self.secondaryPolarization.upper()))))

    firstFrameReference = _get_frame_from_led(os.path.basename(ledFilesReference[0]))
    firstFrameSecondary = _get_frame_from_led(os.path.basename(ledFilesSecondary[0]))
    firstFrameImagesReference = _get_img_files(self.referenceDir, self.referencePolarization.upper(), firstFrameReference)
    firstFrameImagesSecondary = _get_img_files(self.secondaryDir, self.secondaryPolarization.upper(), firstFrameSecondary)


    #determin operation mode
    referenceMode = _get_mode_from_led(os.path.basename(ledFilesReference[0]))
    secondaryMode = _get_mode_from_led(os.path.basename(ledFilesSecondary[0]))
    spotlightModes = ['SBS']
    stripmapModes = ['UBS', 'UBD', 'HBS', 'HBD', 'HBQ', 'FBS', 'FBD', 'FBQ',
                     # ALOS-4 stripmap equivalents (U=SM1/3m, H=SM2/6m, F=SM3/10m, W=wide swath)
                     'UWS', 'UWD', 'UWQ', 'HWS', 'HWD', 'HWQ', 'FWS', 'FWD', 'FWQ']
    scansarNominalModes = ['WBS', 'WBD', 'WWS', 'WWD',
                           # ALOS-4 ScanSAR (X=ScanSAR, W=wide)
                           'XWS', 'XWD']
    scansarWideModes = ['VBS', 'VBD']
    scansarModes = ['WBS', 'WBD', 'WWS', 'WWD', 'VBS', 'VBD',
                    'XWS', 'XWD']

    #usable combinations
    if (referenceMode in spotlightModes) and (secondaryMode in spotlightModes):
        self._insar.modeCombination = 0
    elif (referenceMode in stripmapModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 1
    elif (referenceMode in scansarNominalModes) and (secondaryMode in scansarNominalModes):
        self._insar.modeCombination = 21
    elif (referenceMode in scansarWideModes) and (secondaryMode in scansarWideModes):
        self._insar.modeCombination = 22
    elif (referenceMode in scansarNominalModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 31
    elif (referenceMode in scansarWideModes) and (secondaryMode in stripmapModes):
        self._insar.modeCombination = 32
    else:
        print('\n\nthis mode combination is not possible')
        print('note that for ScanSAR-stripmap, ScanSAR must be reference\n\n')
        raise Exception('mode combination not supported')

# pixel size from real data processing. azimuth pixel size may change a bit as
# the antenna points to a different swath and therefore uses a different PRF.

#   MODE  RANGE PIXEL SIZE (LOOKS)       AZIMUTH PIXEL SIZE (LOOKS)
# -------------------------------------------------------------------
#   SPT    [SBS]
#          1.4304222392897463 (2)         0.9351804642158579 (4)
#   SM1    [UBS,UBD]
#          1.4304222392897463 (2)         1.8291988125114438 (2)
#   SM2    [HBS,HBD,HBQ]
#          2.8608444785794984 (2)         3.0672373839847196 (2)
#   SM3    [FBS,FBD,FBQ]
#          4.291266717869248  (2)         3.2462615913656667 (4)

#   WD1    [WBS,WBD] [WWS,WWD]
#          8.582533435738496  (1)         2.6053935830031887 (14)
#          8.582533435738496  (1)         2.092362043327227  (14)
#          8.582533435738496  (1)         2.8817632034495717 (14)
#          8.582533435738496  (1)         3.054362492601842  (14)
#          8.582533435738496  (1)         2.4582084463356977 (14)

#   WD2    [VBS,VBD]
#          8.582533435738496  (1)         2.9215796012950728 (14)
#          8.582533435738496  (1)         3.088859074497863  (14)
#          8.582533435738496  (1)         2.8792293071133073 (14)
#          8.582533435738496  (1)         3.0592146044234854 (14)
#          8.582533435738496  (1)         2.8818767752199137 (14)
#          8.582533435738496  (1)         3.047038521027477  (14)
#          8.582533435738496  (1)         2.898816222039108  (14)

    #determine default number of looks:
    self._insar.numberRangeLooks1 = self.numberRangeLooks1
    self._insar.numberAzimuthLooks1 = self.numberAzimuthLooks1
    self._insar.numberRangeLooks2 = self.numberRangeLooks2
    self._insar.numberAzimuthLooks2 = self.numberAzimuthLooks2
    #the following two will be automatically determined by runRdrDemOffset.py
    self._insar.numberRangeLooksSim = self.numberRangeLooksSim
    self._insar.numberAzimuthLooksSim = self.numberAzimuthLooksSim
    self._insar.numberRangeLooksIon = self.numberRangeLooksIon
    self._insar.numberAzimuthLooksIon = self.numberAzimuthLooksIon

    if self._insar.numberRangeLooks1 is None:
        self._insar.numberRangeLooks1 = modeProcParDict['ALOS-2'][referenceMode]['numberRangeLooks1']
    if self._insar.numberAzimuthLooks1 is None:
        self._insar.numberAzimuthLooks1 = modeProcParDict['ALOS-2'][referenceMode]['numberAzimuthLooks1']

    if self._insar.numberRangeLooks2 is None:
        self._insar.numberRangeLooks2 = modeProcParDict['ALOS-2'][referenceMode]['numberRangeLooks2']
    if self._insar.numberAzimuthLooks2 is None:
        self._insar.numberAzimuthLooks2 = modeProcParDict['ALOS-2'][referenceMode]['numberAzimuthLooks2']

    if self._insar.numberRangeLooksIon is None:
        self._insar.numberRangeLooksIon = modeProcParDict['ALOS-2'][referenceMode]['numberRangeLooksIon']
    if self._insar.numberAzimuthLooksIon is None:
        self._insar.numberAzimuthLooksIon = modeProcParDict['ALOS-2'][referenceMode]['numberAzimuthLooksIon']


    #define processing file names
    self._insar.referenceDate = _get_date_from_led(os.path.basename(ledFilesReference[0]))
    self._insar.secondaryDate = _get_date_from_led(os.path.basename(ledFilesSecondary[0]))
    self._insar.setFilename(referenceDate=self._insar.referenceDate, secondaryDate=self._insar.secondaryDate, nrlks1=self._insar.numberRangeLooks1, nalks1=self._insar.numberAzimuthLooks1, nrlks2=self._insar.numberRangeLooks2, nalks2=self._insar.numberAzimuthLooks2)


    #find frame numbers
    if (self._insar.modeCombination == 31) or (self._insar.modeCombination == 32):
        if (self.referenceFrames == None) or (self.secondaryFrames == None):
            raise Exception('for ScanSAR-stripmap inteferometry, you must set reference and secondary frame numbers')
    #if not set, find frames automatically
    if self.referenceFrames == None:
        self.referenceFrames = []
        for led in ledFilesReference:
            frameNumber = _get_frame_from_led(os.path.basename(led))
            if frameNumber not in self.referenceFrames:
                self.referenceFrames.append(frameNumber)
    if self.secondaryFrames == None:
        self.secondaryFrames = []
        for led in ledFilesSecondary:
            frameNumber = _get_frame_from_led(os.path.basename(led))
            if frameNumber not in self.secondaryFrames:
                self.secondaryFrames.append(frameNumber)
    #sort frames
    self.referenceFrames = sorted(self.referenceFrames)
    self.secondaryFrames = sorted(self.secondaryFrames)
    #check number of frames
    if len(self.referenceFrames) != len(self.secondaryFrames):
        raise Exception('number of frames in reference dir is not equal to number of frames \
            in secondary dir. please set frame number manually')


    #find swath numbers (if not ScanSAR-ScanSAR, compute valid swaths)
    if (self._insar.modeCombination == 0) or (self._insar.modeCombination == 1):
        self.startingSwath = 1
        self.endingSwath = 1

    if self._insar.modeCombination == 21:
        if self.startingSwath == None:
            self.startingSwath = 1
        if self.endingSwath == None:
            self.endingSwath = 5

    if self._insar.modeCombination == 22:
        if self.startingSwath == None:
            self.startingSwath = 1
        if self.endingSwath == None:
            self.endingSwath = 7

    #determine starting and ending swaths for ScanSAR-stripmap, user's settings are overwritten
    #use first frame to check overlap
    if (self._insar.modeCombination == 31) or (self._insar.modeCombination == 32):
        if self._insar.modeCombination == 31:
            numberOfSwaths = 5
        else:
            numberOfSwaths = 7
        overlapSubswaths = []
        for i in range(numberOfSwaths):
            overlapRatio = check_overlap(ledFilesReference[0], firstFrameImagesReference[i], ledFilesSecondary[0], firstFrameImagesSecondary[0])
            if overlapRatio > 1.0 / 4.0:
                overlapSubswaths.append(i+1)
        if overlapSubswaths == []:
            raise Exception('There is no overlap area between the ScanSAR-stripmap pair')
        self.startingSwath = int(overlapSubswaths[0])
        self.endingSwath = int(overlapSubswaths[-1])

    #save the valid frames and swaths for future processing
    self._insar.referenceFrames = self.referenceFrames
    self._insar.secondaryFrames = self.secondaryFrames
    self._insar.startingSwath = self.startingSwath
    self._insar.endingSwath = self.endingSwath


    ##################################################
    #1. create directories and read data
    ##################################################
    self.reference.configure()
    self.secondary.configure()
    self.reference.track.configure()
    self.secondary.track.configure()
    for i, (referenceFrame, secondaryFrame) in enumerate(zip(self._insar.referenceFrames, self._insar.secondaryFrames)):
        #frame number starts with 1
        frameDir = 'f{}_{}'.format(i+1, referenceFrame)
        os.makedirs(frameDir, exist_ok=True)
        os.chdir(frameDir)

        #attach a frame to reference and secondary
        frameObjReference = MultiMode.createFrame()
        frameObjSecondary = MultiMode.createFrame()
        frameObjReference.configure()
        frameObjSecondary.configure()
        self.reference.track.frames.append(frameObjReference)
        self.secondary.track.frames.append(frameObjSecondary)

        #swath number starts with 1
        for j in range(self._insar.startingSwath, self._insar.endingSwath+1):
            print('processing frame {} swath {}'.format(referenceFrame, j))

            swathDir = 's{}'.format(j)
            os.makedirs(swathDir, exist_ok=True)
            os.chdir(swathDir)

            #attach a swath to reference and secondary
            swathObjReference = MultiMode.createSwath()
            swathObjSecondary = MultiMode.createSwath()
            swathObjReference.configure()
            swathObjSecondary.configure()
            self.reference.track.frames[-1].swaths.append(swathObjReference)
            self.secondary.track.frames[-1].swaths.append(swathObjSecondary)

            #setup reference
            self.reference.leaderFile = _get_led_files(self.referenceDir, referenceFrame)[0]
            if referenceMode in scansarModes:
                self.reference.imageFile = _get_img_files(self.referenceDir, self.referencePolarization.upper(), referenceFrame, swath=j)[0]
            else:
                self.reference.imageFile = _get_img_files(self.referenceDir, self.referencePolarization.upper(), referenceFrame)[0]
            self.reference.outputFile = self._insar.referenceSlc
            self.reference.useVirtualFile = self.useVirtualFile
            #read reference
            (imageFDR, imageData)=self.reference.readImage()
            (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)=self.reference.readLeader()
            self.reference.setSwath(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.reference.setFrame(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.reference.setTrack(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)

            #setup secondary
            self.secondary.leaderFile = _get_led_files(self.secondaryDir, secondaryFrame)[0]
            if secondaryMode in scansarModes:
                self.secondary.imageFile = _get_img_files(self.secondaryDir, self.secondaryPolarization.upper(), secondaryFrame, swath=j)[0]
            else:
                self.secondary.imageFile = _get_img_files(self.secondaryDir, self.secondaryPolarization.upper(), secondaryFrame)[0]
            self.secondary.outputFile = self._insar.secondarySlc
            self.secondary.useVirtualFile = self.useVirtualFile
            #read secondary
            (imageFDR, imageData)=self.secondary.readImage()
            (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)=self.secondary.readLeader()
            self.secondary.setSwath(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.secondary.setFrame(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)
            self.secondary.setTrack(leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData)

            os.chdir('../')
        self._insar.saveProduct(self.reference.track.frames[-1], self._insar.referenceFrameParameter)
        self._insar.saveProduct(self.secondary.track.frames[-1], self._insar.secondaryFrameParameter)
        os.chdir('../')
    self._insar.saveProduct(self.reference.track, self._insar.referenceTrackParameter)
    self._insar.saveProduct(self.secondary.track, self._insar.secondaryTrackParameter)


    catalog.printToLog(logger, "runPreprocessor")
    self._insar.procDoc.addAllFromCatalog(catalog)



def check_overlap(ldr_m, img_m, ldr_s, img_s):
    from isceobj.Constants import SPEED_OF_LIGHT

    rangeSamplingRateReference, widthReference, nearRangeReference = read_param_for_checking_overlap(ldr_m, img_m)
    rangeSamplingRateSecondary, widthSecondary, nearRangeSecondary = read_param_for_checking_overlap(ldr_s, img_s)

    farRangeReference = nearRangeReference + (widthReference-1) * 0.5 * SPEED_OF_LIGHT / rangeSamplingRateReference
    farRangeSecondary = nearRangeSecondary + (widthSecondary-1) * 0.5 * SPEED_OF_LIGHT / rangeSamplingRateSecondary

    #This should be good enough, although precise image offsets are not used.
    if farRangeReference <= nearRangeSecondary:
        overlapRatio = 0.0
    elif farRangeSecondary <= nearRangeReference:
        overlapRatio = 0.0
    else:
        #                     0                  1               2               3
        ranges = np.array([nearRangeReference, farRangeReference, nearRangeSecondary, farRangeSecondary])
        rangesIndex = np.argsort(ranges)
        overlapRatio = ranges[rangesIndex[2]]-ranges[rangesIndex[1]] / (farRangeReference-nearRangeReference)

    return overlapRatio


def read_param_for_checking_overlap(leader_file, image_file):
    from isceobj.Sensor import xmlPrefix
    import isceobj.Sensor.CEOS as CEOS

    #read from leader file
    # ALOS-4/PALSAR-3: from FTR-240031A Table 4.7-1; CEOS stores ~98.24/49.12/32.75/16.37 MHz
    #   → int() truncation gives keys 98/49/32/16 (NOT the ADC rates 166/83/55/28 MHz)
    fsampConst = { 104: 1.047915957140240E+08,          # ALOS-2 SM1 (UB*)
                   52:  5.239579785701190E+07,          # ALOS-2 SM2 (HB*)
                   34:  3.493053190467460E+07,          # ALOS-2 SM3 (FB*)
                   17:  1.746526595233730E+07,          # ALOS-2 ScanSAR (WB*)
                   # ALOS-4 (PALSAR-3) — exact values per FTR-240031A Table 4.7-1
                   98:  9.824218687500000E+07,          # SM1/UW* (98.2422 MHz stored)
                   49:  4.912109343750000E+07,          # SM2/HW* (49.1211 MHz stored)
                   32:  3.274739562500000E+07,          # SM3/FW* (32.7474 MHz stored)
                   16:  1.637369781250000E+07 }         # ScanSAR/XW* (16.3737 MHz stored)

    fp = open(leader_file,'rb')
    leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/leader_file.xml'),dataFile=fp)
    leaderFDR.parse()
    fp.seek(leaderFDR.getEndOfRecordPosition())
    sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/scene_record.xml'),dataFile=fp)
    sceneHeaderRecord.parse()
    fp.seek(sceneHeaderRecord.getEndOfRecordPosition())

    fsamplookup = int(sceneHeaderRecord.metadata['Range sampling rate in MHz'])
    rangeSamplingRate = fsampConst[fsamplookup]
    fp.close()
    #print('{}'.format(rangeSamplingRate))

    #read from image file
    fp = open(image_file, 'rb')
    imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_file.xml'), dataFile=fp)
    imageFDR.parse()
    fp.seek(imageFDR.getEndOfRecordPosition())
    imageData = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_record.xml'), dataFile=fp)
    imageData.parseFast()

    width = imageFDR.metadata['Number of pixels per line per SAR channel']
    near_range = imageData.metadata['Slant range to 1st data sample']
    fp.close()
    #print('{}'.format(width))
    #print('{}'.format(near_range))

    return (rangeSamplingRate, width, near_range)


