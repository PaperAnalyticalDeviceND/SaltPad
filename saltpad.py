#!/usr/bin/env python
import sys
import subprocess
import cv
import cv2
import os
import math
import numpy as np
import getopt

####################################################################################################
# Sandipan Banergee. Computer Science and Engineering. Notre Dame.
# 03/21/2014
# Routine to popup image and select QR and alignment points.
####################################################################################################
def pointPick( org_image ):

    #points to return
    pointspicked = []

    #scale image for popup
    (h, w, p) = orig_im.shape
    scale = 1000.0 / h
    print "Scale", scale
    org_image_s = cv2.resize(org_image, (int(scale * w), int(scale * h)) )

    #mouse control routine
    def mouseStuff(event, x, y, flags, param):
        #select points
        if event == cv2.EVENT_LBUTTONDOWN:
            #is it a near point
            wasNear = False
            plen = len(pointspicked)
            if plen > 0:
                for i in range(0,plen):
                    if np.linalg.norm(np.array([int(x/scale) - pointspicked[i][0], int(y/scale) - pointspicked[i][1]])) < 100:
                        #near point
                        pointspicked[i] = [int(x/scale), int(y/scale)]
                        print 'Moved ',i,' to ', pointspicked[i][0], ',', pointspicked[i][1]
                        #crazy hack for python localizing variables if assigned!
                        org_image_s[0:int((1000.0 * w)), 0:1000] = cv2.resize(org_image, (int(scale * w), int(scale * h)) )
                        for i in range(0,plen):
                            x = int(pointspicked[i][0] * scale + 0.5)
                            y = int(pointspicked[i][1] * scale + 0.5)
                            cv2.line(org_image_s, (x - 10, y), (x + 10, y), (0, 255, 0), 1)
                            cv2.line(org_image_s, (x, y - 10), (x, y + 10), (0, 255, 0), 1)
                        wasNear = True
                        break

            #or just add point
            if (not wasNear) and len(pointspicked) < 6:
                #print 'clicked at ',x,',',y
                cv2.line(org_image_s, (x - 10, y), (x + 10, y), (0, 255, 0), 1)
                cv2.line(org_image_s, (x, y - 10), (x, y + 10), (0, 255, 0), 1)
                #resize and store
                x /= scale
                y /= scale
                x = int(x)
                y = int(y)
                pointspicked.append([x, y])
                print 'clicked at ', x, ',', y
            else:
                if not wasNear:
                    print 'All six points selected'

        #right click to remove point
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(pointspicked) > 0:
                #print 'Cancelling points'
                plen = len(pointspicked)
                if plen > 0:
                    for i in range(0,plen):
                        if np.linalg.norm(np.array([int(x/scale) - pointspicked[i][0], int(y/scale) - pointspicked[i][1]])) < 100:
                            print 'Cancelling point at',i
                            del pointspicked[i]#[plen - 1]#[:]
                            #crazy hack for python localizing variables if assigned!
                            org_image_s[0:int((1000.0 * w)), 0:1000] = cv2.resize(org_image, (int(scale * w), int(scale * h)) )
                            for i in range(0,plen-1):
                                x = int(pointspicked[i][0] * scale + 0.5)
                                y = int(pointspicked[i][1] * scale + 0.5)
                                cv2.line(org_image_s, (x - 10, y), (x + 10, y), (0, 255, 0), 1)
                                cv2.line(org_image_s, (x, y - 10), (x, y + 10), (0, 255, 0), 1)
                            break

    #Setup window
    cv2.namedWindow('PAD Image', cv.CV_WINDOW_NORMAL)
    cv2.setMouseCallback('PAD Image', mouseStuff)

    #loop while selecting
    current_length = -1
    while (1):
        #print image if changed
        #if current_length != len(pointspicked):
        #    current_length = len(pointspicked)
        cv2.imshow('PAD Image', org_image_s)
        #wait for user input to exit
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == 120:
            break
    cv2.destroyAllWindows()

    #return data
    return pointspicked

####################################################################################################
# James Sweet. Computer Science and Engineering. Notre Dame.
# 03/12/2014
# Single Value Decomposition code. This takes the over defined problem and solves for a
# rotaton and translation (Affine) matrix which is then applied to the image.
# Based on the DHARMA java code for mapping point-cloud views.
####################################################################################################
def SVD(srcpoints, dstpoints):
    # Calculate Centroids
    centroid_a = (0, 0)
    centroid_b = (0, 0)
    for i in range(0, len(srcpoints)):
        centroid_a = (centroid_a[0] + srcpoints[i][0], centroid_a[1] + srcpoints[i][1])
        centroid_b = (centroid_b[0] + dstpoints[i][0], centroid_b[1] + dstpoints[i][1])
    centroid_a = (centroid_a[0] / len(srcpoints), centroid_a[1] / len(srcpoints))
    centroid_b = (centroid_b[0] / len(srcpoints), centroid_b[1] / len(srcpoints))

    # Remove Centroids
    new_src = np.copy(srcpoints)
    new_dst = np.copy(dstpoints)
    for i in range(0, len(srcpoints)):
        new_src[i] = (new_src[i][0] - centroid_a[0], new_src[i][1] - centroid_a[1])
        new_dst[i] = (new_dst[i][0] - centroid_b[0], new_dst[i][1] - centroid_b[1])

    # Calculate outer product
    oproduct = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    for i in range(0, len(srcpoints)):
        oproduct.A[0][0] += new_src[i][0] * new_dst[i][0];
        oproduct.A[1][0] += new_src[i][1] * new_dst[i][0];

        oproduct.A[0][1] += new_src[i][0] * new_dst[i][1];
        oproduct.A[1][1] += new_src[i][1] * new_dst[i][1];

    U, s, V = np.linalg.svd(oproduct)

    # Create Rotation Matrix
    R = V * U.T

    # Check for Reflection
    if np.linalg.det(R) < 0.0:
        #print "Reflection"
        V.A[0][1] = -V.A[0][1]
        V.A[1][1] = -V.A[1][1]
        R = V * U.T

    # Calculate Scaling
    Source = R * new_src.T

    sum_ss = 0
    sum_tt = 0
    for i in range(0, len(srcpoints)):
        sum_ss += new_src[i][0] * new_src[i][0]
        sum_ss += new_src[i][1] * new_src[i][1]

        sum_tt += new_dst[i][0] * Source.A[0][i];
        sum_tt += new_dst[i][1] * Source.A[1][i];

    # Scale Matrix
    #RI = (sum_ss / sum_tt) * R
    R = (sum_tt / sum_ss) * R

    # Calculate Translation
    C_A = np.matrix([[-centroid_a[0], -centroid_a[1]]])
    C_B = np.matrix([[centroid_b[0], centroid_b[1]]])

    TL = (C_B.T + (R * C_A.T))

    # Combine Results
    # version for image transformation
    T = np.matrix([
        [R.A[0][0], R.A[0][1], TL.A[0][0]],
        [R.A[1][0], R.A[1][1], TL.A[1][0]]
    ])

    #return partial matrix
    return T


if len(sys.argv) < 2:
    print 'Usage: ' + sys.argv[
        0] + '[-i value] [-s] [-w] [-b] [-g] [-d] [-m] [-t templatefile] [-o] imagefile [guess1 [guess2]]'
    print '-i is Interactive: use mouse to select QR code rectangle.'
    print '      0 no interaction, 1 interact if fails automatic, 2 force interactive.'
    print '-s is Smooth: blur the image a little (can be used multiple times.)'
    print '-g is graphics: show partial results in windows, press a key to continue.'
    print '-w is white balance: make average color in white color square pure white.'
    print '-b is "black balance": make average color in black color square pure black.'
    print '-m is Matrix: print mapping matrix.'
    print '-l is tempLate method: Use to force template matching, not line search.'
    sys.exit(-1)

optlist, args = getopt.getopt(sys.argv[1:], 'wbgdsi:mlt:o:c:a:')

save_correlation = False
debug_images = False
whitebalance = False
blackbalance = False
graphics = False
smoo = 0
interactive = 0
mouseflag = False
mappingmatrix = False
templatefile = 'template2.png'
templatemethod = False
resultsfile = ""
concentrationfile = ""
file_rights = 'a'
selected_lab = ""

calibrationFile = ""

for o, a in optlist:
    if o == '-w':
        whitebalance = True
    elif o == '-b':
        blackbalance = True
    elif o == '-g':
        graphics = True
    elif o == "-d":
        debug_images = True
    elif o == '-s':
        smoo = smoo + 1
    elif o == '-i':
        interactive = int(float(a))
    elif o == '-t':
        templatefile = a
    elif o == "-o":
        resultsfile = a
    elif o == "-m":
        mappingmatrix = True
    elif o == "-l":
        templatemethod = True
    elif o == "-c":
        calibrationFile = a
    elif o == "-a":
        selected_lab = a
    else:
        print 'Unhandled option: ', o
        sys.exit(-2)

print 'args: ', args

#flag for SVD well transformation
doSVDForWells = True

#get filenames
filename = args[0]
filenameroot = '.'.join(filename.split('.')[:-1])
if resultsfile == "auto":
    resultsfile = filenameroot+'.csv'
    file_rights = 'w'

#add specific concentration csv
if resultsfile != "":
    concentrationfile = '.'.join(resultsfile.split('.')[:-1])+'_conc.csv'

if len(args) > 2:
    guess1 = args[1]
    if len(args) > 3:
        guess2 = args[2]
    else:
        guess2 = None
else:
    guess1 = None
    guess2 = None

#get calibration if available
wellcal = []
callab = ""
calset = selected_lab
if calibrationFile != "":
    f = open(calibrationFile, 'rU')
    if f:
        validCal = False
        for line in f:
            #comment line?
            if "#" in line:
                continue
            #get selected cal set
            elif "selected" in line:
                if selected_lab == "":
                    split = line.split(",")
                    calset = split[1]
            #calibration identifier?
            elif "calibration" in line:
                if not validCal:
                    split = line.split(",")
                    if split[1] == calset or calset == "":
                        callab = split[1]
                        validCal = True
                        print "Calibration:",split[1],
                else:
                    validCal = False
                    break
            #get calibration data line if valid calibration name
            elif validCal:
                    split = line.split(",")
                    if len(split) < 5:
                        continue
                    if len(split[0]) == 0:
                        continue
                    numwells = int(split[0])

                    #get string of cal data
                    if len(split) >= numwells * 2 + 3:
                        for i in range(0,numwells):
                            data = []
                            #Standard data
                            try:
                                data = [int(split[i*2+1]), int(split[i*2+2]), float(split[numwells * 2 + 1]), float(split[numwells * 2 + 2])]
                                #extended data
                                try:
                                    minppm = float(split[numwells * 2 + 3])
                                    maxppm = float(split[numwells * 2 + 4])
                                    data.append(minppm)
                                    data.append(maxppm)
                                except:
                                    e = sys.exc_info()[0]
                                    #print "No extended calibration available!"
                                #message?
                                try:
                                    mess = split[numwells * 2 + 5]
                                    if len(mess) > 1: #puts line feed in so remove
                                        data.append(mess)
                                except:
                                    e = sys.exc_info()[0]
                                    #print "No message!"
                            except:
                                e = sys.exc_info()[0]
                                #print "Error in calibration parse!"
                            #add to array
                            if len(data) > 0:
                                wellcal.append(data)

                    else:
                        #print "Calibration file error!"
                        wellcal = []
                        break
    if len(wellcal) > 0:
        print "Calibration data", wellcal
    else:
        print "Calibration file error!"

# OK load image
print 'filename is :', filename

orig_im = cv2.imread(filename)
(h, w, p) = orig_im.shape

#dont print if LS as scale invariant
#print "Original Size:", w, h, p

####################################################################################################
#
# Chris Sweet. Center for Research Computing. Notre Dame.
# 05/09/2014
# Additional code to wrap James Sweet's Line search code for detection of QR code and edge markers.
#
####################################################################################################
#try calling James' Line Search method as scale and rotation invariant
dataLines = []

#new flag to force template method
if not templatemethod and interactive != 2:
    try:
        p = subprocess.Popen(["./ComputerVision2", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        dataLines = stdout.split("\n")
    except:
        print "Unexpected error executing ComputerVision2:", sys.exc_info()[0], ". Reverting to Template method."

points = []
sumsize = 0

#separated points
qrpoints = []
outerpoints = []

#points for transformation
src_points = []
dst_points = []

#enough points? Need 4+
if len(dataLines) >= 5:
    #get point pairs
    for j in range(0, len(dataLines)):
        if "Point" in dataLines[j]:
            dim = dataLines[j].split(":")[1].split(",")
            x = int(dim[1])
            y = int(dim[0])
            sz = int(dim[2])
            #check point unique
            uniquep = True
            for prepoint in points:
                if abs(prepoint[0] - x) < 5 and abs(prepoint[1] - y) < 5:
                    uniquep = False
            if uniquep:
                points = points + [[x, y, sz]]
                sumsize += sz

    print "Points", len(points), "using Line Search method"
    if len(points) >= 4:
        #print them
        print "Points", points

        #get mean size
        meansize = sumsize / len(points)

        #separate points
        for point in points:
            if point[2] > meansize:
                qrpoints = qrpoints + [[point[0], point[1]]]
            else:
                outerpoints = outerpoints + [[point[0], point[1]]]

#need at least 2 of each points
if len(qrpoints) >= 2 and len(outerpoints) >= 2:
    print "Method: LS. Points found by LS,", len(points)
    #print "QR",len(qrpoints),"Outer",len(outerpoints)
    #order QR points
    if len(qrpoints) == 3:
        qr_top_left = (9999, 9999)
        qr_top_right = (0, 0)
        qr_bot_left = (0, 0)

        for point in qrpoints:
            if point[0] > qr_top_right[0]:
                qr_top_right = point

            if point[1] > qr_bot_left[1]:
                qr_bot_left = point

        for point in qrpoints:
            if point != qr_top_right and point != qr_bot_left:
                qr_top_left = point
        qrpoints = [qr_top_left, qr_bot_left, qr_top_right]
    else:
        #get distance between points and take root 2 in case diag the take half to find bounds of 'near' points
        print "QR ", qrpoints
        dist = math.sqrt((qrpoints[0][0] - qrpoints[1][0]) * (qrpoints[0][0] - qrpoints[1][0]) + (
        qrpoints[0][1] - qrpoints[1][1]) * (qrpoints[0][1] - qrpoints[1][1])) / 3.0
        print "Dist ", dist
        #x coords the same? then
        if abs(qrpoints[0][0] - qrpoints[1][0]) < dist:
            if qrpoints[0][1] < qrpoints[1][1]:
                qrpoints = [qrpoints[0], qrpoints[1], [-1, -1]]
            else:
                qrpoints = [qrpoints[1], qrpoints[0], [-1, -1]]
        else:
            #maybe y coord same?
            if abs(qrpoints[0][1] - qrpoints[1][1]) < dist:
                if qrpoints[0][0] < qrpoints[1][0]:
                    qrpoints = [qrpoints[0], [-1, -1], qrpoints[1]]
                else:
                    qrpoints = [qrpoints[1], [-1, -1], qrpoints[0]]
            #else diagonal
            else:
                if qrpoints[0][0] < qrpoints[1][0]:
                    qrpoints = [[-1, -1], qrpoints[0], qrpoints[1]]
                else:
                    qrpoints = [[-1, -1], qrpoints[1], qrpoints[0]]

    print "QR points", qrpoints

    #order outer points
    if len(outerpoints) == 3:
        top_right = [0, 9999]
        bottom_right = [0, 0]
        bottom_left = [9999, 0]

        for point in outerpoints:
            if point[0] > top_right[0]:
                if point[1] < top_right[1]:
                    top_right = point

            if point[0] < bottom_left[0]:
                if point[1] > bottom_left[1]:
                    bottom_left = point

        for point in outerpoints:
            if point != top_right and point != bottom_left:
                bottom_right = point
        outerpoints = [bottom_left, bottom_right, top_right]
    else:
        #get distance between points and take root 2 in case diag the take half to find bounds of 'near' points
        dist = math.sqrt((outerpoints[0][0] - outerpoints[1][0]) * (outerpoints[0][0] - outerpoints[1][0]) + (
        outerpoints[0][1] - outerpoints[1][1]) * (outerpoints[0][1] - outerpoints[1][1])) / 3.0
        #x coords the same? then
        if abs(outerpoints[0][0] - outerpoints[1][0]) < dist:
            if outerpoints[0][1] < outerpoints[1][1]:
                outerpoints = [[-1, -1], outerpoints[1], outerpoints[0]]
            else:
                outerpoints = [[-1, -1], outerpoints[0], outerpoints[1]]
        else:
            #maybe y cood same?
            if abs(outerpoints[0][1] - outerpoints[1][1]) < dist:
                if outerpoints[0][0] < outerpoints[1][0]:
                    outerpoints = [outerpoints[0], outerpoints[1], [-1, -1]]
                else:
                    outerpoints = [outerpoints[1], outerpoints[0], [-1, -1]]
            #then opposing corvers
            else:
                if outerpoints[0][0] < outerpoints[1][0]:
                    outerpoints = [outerpoints[0], [-1, -1], outerpoints[1]]
                else:
                    outerpoints = [outerpoints[1], [-1, -1], outerpoints[0]]

    print "Outer points", outerpoints

    #add qr points and their transform
    transqrpoints = [[10, 10], [10, 146], [146, 10]]

    for i in range(0, 3):
        if qrpoints[i][0] >= 0:
            src_points.append(qrpoints[i])
            dst_points.append(transqrpoints[i])

    #and outerpoints
    transpoints = [[0, 1086], [601, 1086], [601, 0]]

    for i in range(0, 3):
        if outerpoints[i][0] >= 0:
            src_points.append(outerpoints[i])
            dst_points.append(transpoints[i])

    print "Source points", src_points
    print "Destination points", dst_points

####################################################################################################
#
# James Sweet. Computer Science and Engineering. Notre Dame.
# 03/12/2014
# Replacement code to find the markers by using templates.
#
####################################################################################################
###only do this if no points, Template method###############################################
else:
    if interactive != 2:
        print "Method: Template. Points found by LS,", len(points)
        print "Original Size:", w, h, p
        #sys.exit(0)
        # if image is landscape, rotate it.
        if w > h:
            print 'transposing image'
            im2 = cv2.transpose(orig_im)
            im3 = cv2.flip(im2, 1)
            orig_im = im3
            (h, w, p) = orig_im.shape
            print 'transposed size ', w, 'w x ', h, 'h'


        #resize? Template matching is size sensitive
        if h / w < 1.3388:
            orig_im = cv2.resize(orig_im, (1936, 1936 * h / w))
        else:
            orig_im = cv2.resize(orig_im, (2592 * w / h, 2592))

        (h, w, p) = orig_im.shape

        #orig_im = cv2.resize(orig_im, (1936, 2592))

        orig_g = cv2.cvtColor(orig_im, cv.CV_BGR2GRAY)
        orig_gf = orig_g.astype(np.float32)
        if debug_images:
            cv2.imwrite(filenameroot + '.fgorig.png', orig_gf, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

        # QR Square Template Matching
        qrSquare = cv2.imread("QRSquare.png", cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32) / 255.0
        qw, qh = qrSquare.shape[::-1]

        qrResult = cv2.matchTemplate(orig_gf, qrSquare, cv.CV_TM_CCOEFF_NORMED)
        if save_correlation:
            np.savetxt("matchresult.txt", qrResult)

        if debug_images:
            cv2.imwrite(filenameroot + '.qrsquare.png', (((qrResult + 1.0) * 0.5) * 255).astype(np.uint8),
                        [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

        qrPoints = []

        ####################################################################################################
        # Chris Sweet. Center for Research Computing. Notre Dame.
        # 05/04/2014
        # Replacement code for finding points whose correlation exceeds 75%. New method finds the global
        # maximum then masks out an area around this point equal to the template size. The routine then finds
        # the next maximum etc. until the required number is found or the correlation falls below a set level.
        ####################################################################################################
        #get maximum points until we have all three or the certainty is <=0.75
        qrmask = np.ones(qrResult.shape, np.uint8)
        maxVal = 1
        qrtol = 0.55

        while len(qrPoints) < 3 and maxVal > qrtol:
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(qrResult, qrmask);
            if maxVal <= qrtol:
                break
            print "Max QR code location", maxLoc, ",", maxVal
            qrPoints.append((maxLoc[0] + qw / 2, maxLoc[1] + qh / 2))
            rect = [[maxLoc[0] - qw / 2, maxLoc[1] - qh / 2], [maxLoc[0] + qw / 2, maxLoc[1] - qh / 2],
                    [maxLoc[0] + qw / 2, maxLoc[1] + qh / 2], [maxLoc[0] - qw / 2, maxLoc[1] + qh / 2]]
            poly = np.array([rect], dtype=np.int32)
            cv2.fillPoly(qrmask, poly, 0)

        ####################################################################################################

        if len(qrPoints) != 0:
            if debug_images:
                orig_ann = orig_im.copy()

                i = 0
                for point in qrPoints:
                    pt1 = (point[0] - 20, point[1] - 20)
                    pt2 = (point[0] + 20, point[1] + 20)
                    cv2.rectangle(orig_ann, pt1, pt2, (255, 0, 0), 5)
                    cv2.putText(orig_ann, str(i), point, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
                    i += 1
                cv2.imwrite(filenameroot + '.qrsquare.ann.png', orig_ann, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

            print "QR Points Found:", len(qrPoints)
            if len(qrPoints) == 3:
                qr_top_left = (999, 999)
                qr_top_right = (0, 0)
                qr_bot_left = (0, 0)

                print qrPoints

                for point in qrPoints:
                    if point[0] > qr_top_right[0]:
                        qr_top_right = point

                    if point[1] > qr_bot_left[1]:
                        qr_bot_left = point

                for point in qrPoints:
                    if point != qr_top_right and point != qr_bot_left:
                        qr_top_left = point

                src_points.append(qr_top_left)
                dst_points.append((10, 10))

                src_points.append(qr_top_right)
                dst_points.append((146, 10))

                src_points.append(qr_bot_left)
                dst_points.append((10, 146))

        # Alignment template matching
        templateSquare = cv2.imread("AlignmentSquare.png", cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32) / 255.0
        tw, th = templateSquare.shape[::-1]
        #templateSquare = cv2.resize(templateSquare, (int(tw*1.1), int(th*1.1)))
        #tw, th = templateSquare.shape[::-1]

        resultSquare = cv2.matchTemplate(orig_gf, templateSquare, cv.CV_TM_CCOEFF_NORMED)
        if save_correlation:
            np.savetxt("alignresult.txt", resultSquare)

        if debug_images:
            cv2.imwrite(filenameroot + '.square.png', (((resultSquare + 1.0) * 0.5) * 255).astype(np.uint8),
                        [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

        if graphics:
            cv2.imshow("templateSquare", templateSquare)
            cv2.imshow("Correlation coefficient", resultSquare)
            cv2.waitKey(0)

        affinePoints = []

        ####################################################################################################
        # Chris Sweet. Center for Research Computing. Notre Dame.
        # 05/04/2014
        # Replacement code for finding points whose correlation exceeds 75%. New method finds the global
        # maximum then masks out an area around this point equal to the template size. The routine then finds
        # the next maximum etc. until the required number is found or the correlation falls below a set level.
        ####################################################################################################
        #get maximum points until we have all three or the certainty is <=0.75
        affinemask = np.ones(resultSquare.shape, np.uint8)
        affinemaxVal = 1
        affinetol = 0.55

        while len(affinePoints) < 3 and affinemaxVal > affinetol:
            affineminVal, affinemaxVal, affineminLoc, affinemaxLoc = cv2.minMaxLoc(resultSquare, affinemask);
            if affinemaxVal <= affinetol:
                break
            print "Max affine point location", affinemaxLoc, ",", affinemaxVal
            affinePoints.append((affinemaxLoc[0] + tw / 2, affinemaxLoc[1] + th / 2))
            rect = [[affinemaxLoc[0] - tw / 2, affinemaxLoc[1] - th / 2],
                    [affinemaxLoc[0] + tw / 2, affinemaxLoc[1] - th / 2],
                    [affinemaxLoc[0] + tw / 2, affinemaxLoc[1] + th / 2],
                    [affinemaxLoc[0] - tw / 2, affinemaxLoc[1] + th / 2]]
            poly = np.array([rect], dtype=np.int32)
            cv2.fillPoly(affinemask, poly, 0)
        ####################################################################################################

        if debug_images:
            print affinePoints
            orig_ann = orig_im.copy()

            i = 0
            for point in affinePoints:
                pt1 = (point[0] - 20, point[1] - 20)
                pt2 = (point[0] + 20, point[1] + 20)
                cv2.rectangle(orig_ann, pt1, pt2, (255, 0, 0), 5)
                cv2.putText(orig_ann, str(i), point, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
                i += 1
            cv2.imwrite(filenameroot + '.square.ann.png', orig_ann, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

        print "Alignment Points Found:", len(affinePoints)

        #enough points?
        if len(affinePoints) == 3:
            top_right = (0, 999)
            bottom_right = (0, 0)
            bottom_left = (999, 0)

            for point in affinePoints:
                if point[0] > top_right[0]:
                    if point[1] < top_right[1]:
                        top_right = point

                if point[0] <= bottom_left[0]:
                    if point[1] > bottom_left[1]:
                        bottom_left = point

            for point in affinePoints:
                if point != top_right and point != bottom_left:
                    bottom_right = point

            src_points.append(top_right)
            dst_points.append((601, 0))

            src_points.append(bottom_right)
            dst_points.append((601, 1086))

            src_points.append(bottom_left)
            dst_points.append((0, 1086))

#end of affine point acquisition

#### Do we have enough points to find the transformation? First test to call Sandipan Banergee's code if interactive.
if len(src_points) < 3 and interactive > 0:
    print "Call Sandipan Banergee's code..."
    ## Put up image, hold code execution and return 6 values in array points
    #
    points = pointPick(orig_im)
    # then sort the points based on geometric knowledge Chris Sweet 05/14/2014
    if len(points) == 6:
        #print them
        print "Points", points
        #put norms in here
        pointnorms = []
        #get norms
        for i in range(0, 6):
            pointnorms = pointnorms + [math.sqrt(points[i][0] * points[i][0] + points[i][1] * points[i][1])]
        #order points
        #get two extrems
        rhs_bottom_edge = points[0]
        rhs_bottom_edge_norm = pointnorms[0]
        lhs_top_qr = points[0]
        lhs_top_qr_norm = pointnorms[0]
        for i in range(1, 6):
            if pointnorms[i] > rhs_bottom_edge_norm:
                rhs_bottom_edge = points[i]
                rhs_bottom_edge_norm = pointnorms[i]
            if pointnorms[i] < lhs_top_qr_norm:
                lhs_top_qr = points[i]
                lhs_top_qr_norm = pointnorms[i]
        # get other edges
        lhs_bottom_edge = [0, 0]
        rhs_top_edge = [0, 0]
        for i in range(0, 6):
            if points[i] != rhs_bottom_edge and points[i] != lhs_top_qr:
                if points[i][1] > lhs_bottom_edge[1]:
                    lhs_bottom_edge = points[i]
                if points[i][0] > rhs_top_edge[0]:
                    rhs_top_edge = points[i]
        # get other qr
        lhs_bottom_qr = [0, 0]
        rhs_top_qr = [0, 0]
        for i in range(0, 6):
            if points[i] != rhs_bottom_edge and points[i] != lhs_top_qr and points[i] != lhs_bottom_edge and points[
                i] != rhs_top_edge:
                if points[i][1] > lhs_bottom_qr[1]:
                    lhs_bottom_qr = points[i]
                if points[i][0] > rhs_top_qr[0]:
                    rhs_top_qr = points[i]
        #back into array
        points = [lhs_top_qr, rhs_top_qr, lhs_bottom_qr, rhs_top_edge, rhs_bottom_edge, lhs_bottom_edge]
        actualpoints = [(10, 10), (146, 10), (10, 146), (601, 0), (601, 1086), (0, 1086)]
        for i in range(0, 6):
            src_points.append(points[i])
            dst_points.append(actualpoints[i])
print "Re-arranged points", points

#### Do we have enough points to find the transformation? final test and exit if not.###############
if len(src_points) < 3:
    print "Insufficient data for SVD."
    sys.exit(-3)

####################################################################################################
# James Sweet. Computer Science and Engineering. Notre Dame.
# 03/12/2014
# Single Value Decomposition code. This takes the over defined problem and solves for a
# rotaton and translation (Affine) matrix which is then applied to the image.
# Based on the DHARMA java code for mapping point-cloud views.
####################################################################################################
srcpoints = np.array(src_points, np.float32)
dstpoints = np.array(dst_points, np.float32)

np.set_printoptions(precision=4, suppress=True)

#use points to find rotation/translation partial matrix with SVD
T = SVD(srcpoints, dstpoints)

# actual full matrix version (used below and also printed for test suite)
TI = np.matrix([
    [T.A[0][0], T.A[0][1], T.A[0][2]],
    [T.A[1][0], T.A[1][1], T.A[1][2]],
    [0, 0, 1]
])

if mappingmatrix:
    print "Mapping Matrix"
    print TI.tolist()

# calculate errors by transforming points
maxerror = 0
for i in range(0, len(src_points)):
    transformed_point = TI * np.matrix([src_points[i][0], src_points[i][1], 1.0]).T
    error = np.linalg.norm(np.array(transformed_point.A1[:2] - dst_points[i]))
    if error > maxerror:
        maxerror = error

print "Transformation maximum error,",maxerror

# bail if error exceeds 15 pixels (relates to sample circle in relation to sample well)
if maxerror > 15:
    print "Transformation error exceeds threshold of 15 pixels."
    sys.exit(-4)

#if debug_images:
    #im_scaled = cv2.resize(orig_im, (601, 1086))
    #cv2.imwrite(filenameroot + '.scaled.png', im_scaled, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

#eye candy
im_warped = cv2.warpAffine(orig_im, T, (601 + 40, 1086))
gim_warped = cv2.cvtColor(im_warped, cv.CV_BGR2GRAY)
fgim_warped = gim_warped.astype(np.float32)

if graphics:
    cv2.imshow("Warped image", im_warped)
    cv2.waitKey(0)

if debug_images:
    cv2.imwrite(filenameroot + '.warped.png', im_warped, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

mask = np.zeros(im_warped.shape[0:2], np.uint8)
sim_warped = im_warped   # handle the case where neither black nor white balance

#### Find squares #################################################################################
white_square = (0, 0)
template_squares = cv2.imread("padscrs2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32) / 255.0
result_squares = cv2.matchTemplate(fgim_warped, template_squares, cv.CV_TM_CCOEFF_NORMED)
sqminVal, sqmaxVal, sqminLoc, sqmaxLoc = cv2.minMaxLoc(result_squares)
#print "Squares at",sqmaxLoc[0]+120,sqmaxLoc[1]+76,"with threshold",sqmaxVal
if sqmaxVal > 0.80:
    #TODO read in the template offsets (120, 76)
    white_square = (sqmaxLoc[0]+120,sqmaxLoc[1]+76)
    print "Squares at", white_square, "with threshold", sqmaxVal

#### Use template matching to gather evidence for the twelve cells. ###############################
# Load cell template image
# this cell template was chopped out of a normalized image.
template = cv2.imread(templatefile, cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32) / 255.0
(ch, cw) = template.shape

result = cv2.matchTemplate(fgim_warped, template, cv.CV_TM_CCOEFF_NORMED)
if save_correlation:
    np.savetxt("targetresult.txt", result)

cellPoints = []

####################################################################################################
# Chris Sweet. Center for Research Computing. Notre Dame.
# 05/04/2014
# Replacement code for finding points whose correlation exceeds 75%. New method finds the global
# maximum then masks out an area around this point equal to the template size. The routine then finds
# the next maximum etc. until the required number is found or the correlation falls below a set level.
####################################################################################################
#get maximum points until we have all three or the certainty is <=threshold
cellmask = np.ones(result.shape, np.uint8)
cellmaxVal = 1
cellthr = 0.50

while len(cellPoints) < 12 and cellmaxVal > cellthr:
    cellminVal, cellmaxVal, cellminLoc, cellmaxLoc = cv2.minMaxLoc(result, cellmask);
    if cellmaxVal <= cellthr:
        break
    print "Max cell point location", cellmaxLoc, ",", cellmaxVal
    #TODO read in the template offsets (2, 1)
    cellPoints.append((cellmaxLoc[0] + cw / 2.0 - 1.61, cellmaxLoc[1] + ch / 2.0 - 0.50))
    rect = [[cellmaxLoc[0] - cw / 2, cellmaxLoc[1] - ch / 2], [cellmaxLoc[0] + cw / 2, cellmaxLoc[1] - ch / 2],
            [cellmaxLoc[0] + cw / 2, cellmaxLoc[1] + ch / 2], [cellmaxLoc[0] - cw / 2, cellmaxLoc[1] + ch / 2]]
    poly = np.array([rect], dtype=np.int32)
    cv2.fillPoly(cellmask, poly, 0)
####################################################################################################

# sort contours from top to bottom, left to right.
# the contour extractor appears to go bottom-up robustly,
# so the rows are four groups of three.  But left-to-right sorting
# is not as robust.

# OK, ad hoc sorting. Some of the circles will be detected and some won't.
# find the detected circles, in the 4x3 grid, and flag them

# approx centers (cx,cy) in normalized coordinates (danger this may be flaky in
# the lower right corner...)

centerboxes = [
    [[82, 301], [295, 301], [507, 301]],
    [[82, 514], [295, 514], [507, 514]],
    [[82, 727], [295, 727], [507, 727]],
    [[82, 940], [295, 940], [507, 940]]
]

outcircles = []
comparePoints = []
cellpointindex = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
k = 0
for l in range(0, len(cellPoints)):#c in cellPoints:
    cx = cellPoints[l][0]
    cy = cellPoints[l][1]
    #print "Print Point",cx,",",cy
    #mask.fill(0)
    #cv2.circle(mask, (cx, cy), 17, (255, 255, 255, 255), cv.CV_FILLED)
    #cv2.circle(im_warped, (cx, cy), 17, (0, 255, 255, 255), 2)
    #cv2.putText(im_warped, str(k), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255))
    k = k + 1

    # which center am I closest to?
    dmin_x = 10000
    dmin_y = 10000
    dmin = 10000
    ijmin = [-1, -1]
    for i in xrange(4):
        for j in xrange(3):
            cc = centerboxes[i][j]

            d_x = abs(cc[0] - cx)
            dmin_x = min(dmin_x, d_x)

            d_y = abs(cc[1] - cy)
            dmin_y = min(dmin_y, d_y)

            d = abs(d_x) + abs(d_y)
            if d < dmin:
                dmin = d
                ijmin = [i, j]

    #save data
    comparePoints.append(centerboxes[ijmin[0]][ijmin[1]])
    outcircles = outcircles + [[ijmin[0], ijmin[1], cx, cy, dmin, dmin_x, dmin_y]]
    cellpointindex[ijmin[0] + ijmin[1] * 4] = l

print "Contours Matched:", len(outcircles)

#preset offsets
d_x = 0
d_y = 0

#do SVD for rotation/translation?
if doSVDForWells and len(cellPoints) > 1:
    #do SVD for wells
    TCP = SVD(cellPoints, comparePoints)

    #get full matrix
    TICP = np.matrix([
        [TCP.A[0][0], TCP.A[0][1], TCP.A[0][2]],
        [TCP.A[1][0], TCP.A[1][1], TCP.A[1][2]],
        [0, 0, 1]
    ])

    #update centerboxes
    for i in xrange(4):
        for j in xrange(3):
            temp_point = np.matrix([centerboxes[i][j][0], centerboxes[i][j][1], 1.0])
            temp_point = TICP.I * temp_point.T
            centerboxes[i][j] = (temp_point[0], temp_point[1])

else:
    # else Calculate Average X, Y Distance
    for c in outcircles:
        d_x += centerboxes[c[0]][c[1]][0] - c[2]
        d_y += centerboxes[c[0]][c[1]][1] - c[3]
    d_x = d_x / len(outcircles)
    d_y = d_y / len(outcircles)
    print "Well offsets,", d_x, ", ", d_y

# Calculate data values
# Handle Colour Squares
if resultsfile == "":
	fout = sys.stdout
else:
    fout = file(resultsfile, file_rights)
    foutconc = file(concentrationfile, file_rights)


print >>fout,'File name,%s' % (filename)
print >> fout, 'i, j, red, green, blue, A'
if resultsfile != "":
    print 'File name,%s' % (filename)
    print 'i, j, red, green, blue, A'

colour_mask = np.zeros(im_warped.shape[0:2], np.uint8)

colour_square_center = [
    [[507, 132]],
    [[555, 106], [555, 152]],
    [[600, 106], [600, 152]]
]

A = 70
k = 0
#calculate offset if detected
square_offset = (0, 0)
if white_square[0] > 0 and white_square[1] > 0:
    square_offset = (colour_square_center[2][1][0] - white_square[0], colour_square_center[2][1][1] - white_square[1])

for i in range(0, len(colour_square_center)):
    for j in range(0, len(colour_square_center[i])):
        # Offset location by averages
        cx = colour_square_center[i][j][0] - square_offset[0]
        cy = colour_square_center[i][j][1] - square_offset[1]

        pt1 = (cx - 8, cy - 8)
        pt2 = (cx + 8, cy + 8)

        colour_mask.fill(0)
        cv2.rectangle(colour_mask, pt1, pt2, (255, 0, 0), 1)
        s = cv2.mean(sim_warped, colour_mask)

        if i == 2 and j == 1:
            A = 255 - (s[0] + s[1] + s[2]) / 3
            #print "A value", A
            print >> fout, '%d, %d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2], A)
            if resultsfile != "":
                print '%d, %d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2], A)
        else:
            print >> fout, '%d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2])
            if resultsfile != "":
                print '%d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2])

        cv2.rectangle(im_warped, pt1, pt2, (255, 0, 0), 1)
        cv2.putText(im_warped, str(k), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        k = k + 1

print >> fout, ''

# Handle Wax Circles
print >> fout, 'i, j, cx, cy, inv_g_avg_unweighted, compensated, sample diameter, ppm iodine'
if resultsfile != "":
    print 'i, j, cx, cy, inv_g_avg_unweighted, compensated, sample diameter, ppm iodine'

# save results for Nick's high level analysis# Creates a list containing 5 lists initialized to 0
intensity = [[0 for x in range(len(centerboxes[0]))] for x in range(len(centerboxes))]
ppm = [[0 for x in range(len(centerboxes[0]))] for x in range(len(centerboxes))]

k = 0
for i in range(0, len(centerboxes)):
    for j in range(0, len(centerboxes[i])):
        # Offset location by averages
        if cellpointindex[i + j * 4] == -1:
            cx = centerboxes[i][j][0] - d_x
            cy = centerboxes[i][j][1] - d_y
        else:
            cx = int(cellPoints[cellpointindex[i + j * 4]][0])
            cy = int(cellPoints[cellpointindex[i + j * 4]][1])

        mask.fill(0)

        #first do test at 10 pixels
        cv2.circle(mask, (cx, cy), 10, (255, 255, 255, 255), cv.CV_FILLED)
        s = cv2.mean(sim_warped, mask)
        b=255 - (s[0] + s[1] + s[2]) / 3

        #then the general one at 17 pixels
        cv2.circle(mask, (cx, cy), 17, (255, 255, 255, 255), cv.CV_FILLED)
        s = cv2.mean(sim_warped, mask)

        B = 255 - (s[0] + s[1] + s[2]) / 3

        #test to see if we are near the edge
        testradius = 17
        if abs(b - B) > 5:
            B = b
            testradius = 10

        Bdash = B - (A - 70)

        original_point = np.matrix([cx, cy, 1.0])
        original_point = TI.I * original_point.T

        #save intensity. Note i is row, j column
        intensity[i][j] = Bdash
        
        #check for calibration data
        offset = 0
        divisor = 0
        for cal in wellcal:
            if cal[0] == i and cal[1] == j:
                offset = cal[2]
                divisor = cal[3]

        if divisor == 0:
            print >> fout, '%d, %d, %d, %d, %.1f, %.1f, %d' % (i, j, original_point[0], original_point[1], B, Bdash, testradius)
        else:
            print >> fout, '%d, %d, %d, %d, %.1f, %.1f, %d, %.1f' % (i, j, original_point[0], original_point[1], B, Bdash, testradius, ((Bdash - offset) / divisor))
            #save ppm if available. Note i is row, j column
            ppm[i][j] = (Bdash - offset) / divisor
        
        if resultsfile != "":
            if divisor == 0:
                print '%d, %d, %d, %d, %.1f, %.1f, %d' % (i, j, original_point[0], original_point[1], B, Bdash, testradius)
            else:
                print '%d, %d, %d, %d, %.1f, %.1f, %d, %.1f' % (i, j, original_point[0], original_point[1], B, Bdash, testradius, ((Bdash - offset) / divisor))

        cv2.circle(im_warped, (cx, cy), 17, (255, 255, 255, 255), 2)
        cv2.putText(im_warped, str(k), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        k = k + 1

#do Nick's analysis, now using row,column indexing

#Test for message breaks
message_break = False

#loop over all test cells
for i in range(0, len(centerboxes)): #row
    for j in range(0, len(centerboxes[i])): #column

        #test to see if this call has a matching extended calibration
        for cal in wellcal:
            if cal[0] == i and cal[1] == j and len(cal) >= 7: #if length is 7 then message, else average
                #print "Testing cell",i,j
                #do test
                test_value = ppm[i][j]

                #outside range?
                if test_value < cal[4] or test_value > cal[5]:
                    if len(cal) > 7:    #stop in name
                        message_break = True
                    print >> fout, cal[6]
                    print >> foutconc, filename, ', -1, 0'
                    if resultsfile != "":
                        print cal[5]

                break   #only use first entry of clarity!

#OK to continue?
if not message_break:
    #averaging of valid data
    #initialize
    average_count = 0
    averaged_value = 0
    averaged_squared_value = 0

    #loop over all test cells
    for i in range(0, len(centerboxes)): #row
        for j in range(0, len(centerboxes[i])): #column

            #test to see if this call has a matching extended calibration
            for cal in wellcal:
                if cal[0] == i and cal[1] == j and len(cal) == 6: #if length is 7 then message, else average
                    #do test
                    test_value = ppm[i][j]

                    #inside range?
                    if test_value >= cal[4] and test_value <= cal[5]:
                        #print "Averaging cell",i,j,ppm[i][j],cal[4],cal[5]
                        average_count += 1
                        averaged_value += test_value
                        averaged_squared_value += test_value * test_value

                    break   #only use first entry of clarity!

    # Calculate the standard deviation of every result that got averaged in previous steps
    # Multiply the result and the standard deviation (SD) by 5 to get final result.
    # Display "Concentration in dry salt sample is X plus/minus SD ppm iodine from potassium iodate."
    if average_count > 0:
        mean = averaged_value / average_count
        meansq = averaged_squared_value / average_count
        sd = math.sqrt(meansq - mean * mean)
        print >> fout, "Concentration in dry salt sample is","{0:.1f}".format(mean * 5),"plus/minus","{0:.1f}".format(sd * 5),"ppm iodine from potassium iodate."
        print >> foutconc, filename, ',',"{0:.1f}".format(mean * 5),',',"{0:.1f}".format(sd * 5),',',average_count,','
        if resultsfile != "":
            print "Concentration in dry salt sample is","{0:.1f}".format(mean * 5),"plus/minus","{0:.1f}".format(sd * 5),"ppm iodine from potassium iodate."
    else:
        print "No data in average, all cells out of range!"

#new marker for end of data
print "End of analysis."


if graphics:
    cv2.imshow('annotated', im_warped)
    cv2.waitKey(0)

# Print annotated image
cv2.imwrite(filenameroot + '.ann.png', im_warped, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

sys.exit(0)
