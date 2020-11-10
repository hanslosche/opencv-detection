import cv2 as cv
import numpy as np


def findClickPositions(needle_img_path, haystack_img_path, threshold=0.5, debug_mode=None):

    haystack_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
    needle_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # Found best results in this case using TM_SQDIFF_NORMED
    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(haystack_img, needle_img, method)

    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    #print(locations)

    # first we need to create the list of [x, y, w, h] rectangles
    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
        rectangles.append(rect)
        rectangles.append(rect)


    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
    print(rectangles, "\n Lenght of :", len(rectangles))

    points = []
    if len(rectangles):
        print('Found needle.')

        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        # Loop over all the locations and draw their rectangle
        for (x, y, w, h) in rectangles:

            # Determine the center position
            center_x = x + int(w/2)
            center_y = y + int(h/2)

            # save the points
            points.append((center_x, center_y ))
            if debug_mode == 'rectangles':
                # Determine the box positions
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                # Draw the box
                cv.drawMarker(haystack_img, (center_x, center_y), marker_color, marker_type)
            elif debug_mode == 'points':
                cv.drawMarker(haystack_img, (center_x, center_y), marker_color, marker_type)

        if debug_mode:
            cv.imshow('Matches', haystack_img)
            cv.waitKey()
            #cv.imwrite('result.jpg', haystack_img)

    return points

points = findClickPositions('farm.jpg', 'cabbage.jpg',
                            threshold=0.46, debug_mode='points')

print(points)
