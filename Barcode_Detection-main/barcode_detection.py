import cv2
# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png
import os

#Utilize camera on machine (0 is default, 1 for external USB camera device
onboard_camera = cv2.VideoCapture(0)

#Create new window to capture images
cv2.namedWindow("Barcode Detection Capture")

#Track how many images the user has taken
img_counter = 0

#User can continue to take captures until they press ESC
while True:
    #Read camera capture
    ret, capture = onboard_camera.read()
    if not ret:
        print("Could not capture image.")
        break
    cv2.imshow("Barcode Detection Capture", capture)

    k = cv2.waitKey(1)

    #Check if user takes a photo or exits capture tool
    #If they do, the photo is saved to the local directory
    if k%256 == 27:
        # ESC has been pressed so the capture tool will exit
        print("ESC has been pressed, closing image capture.")
        break
    elif k%256 == 32:
        #SPACE has been pressed to capture a photo
        img_name = "extension_input_images/"+"product_{}.png".format(img_counter)
        cv2.imwrite(img_name, capture)
        print("{} has been captured and saved.".format(img_name))
        img_counter += 1

#Close camera
onboard_camera.release()

#Close capture window
cv2.destroyAllWindows()


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


#Converting RGB to Greyscale
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    count = 0
    
    for row in greyscale_pixel_array:
        for index in range(len(row)):
            gs = int(round(pixel_array_r[count][index] * 0.299 +  pixel_array_g[count][index] * 0.587 +  pixel_array_b[count][index] * 0.114,0))
            row[index] = gs
        count+=1
    
    return greyscale_pixel_array

#Scaling pixel value to 0 and 255
def scaleTo0And255(pixel_array, image_width, image_height):
    output_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    min_val = pixel_array[0][0]
    max_val = pixel_array[0][0]
    
    for row in pixel_array:
        for val in row:
            if val < min_val:
                min_val = val
                
    for row in pixel_array:
        for val in row:
            if val > max_val:
                max_val = val
    if (max_val - min_val) == 0:
        scale_factor = 0
    else:               
        scale_factor = 255.0 / (max_val - min_val)
    
    for row in pixel_array:
        for i in range(len(row)):
            scaled = int(round((row[i]- min_val) * scale_factor))
            if scaled > 255:
                row[i] = 255
            elif scaled < 0:
                row[i] = 0
            else:
                row[i] = scaled
            
    return pixel_array

#Compute the vertical sobel edge component
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    horizontal_edges = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for x in range(len(pixel_array)):
        if x != 0 and x != image_height-1:
            for i in range(len(pixel_array[x])):
                if i != 0 and i != image_width -1:
                    pixel_value1 = (pixel_array[x-1][i-1] + (pixel_array[x][i-1] *2) + pixel_array[x+1][i-1]) * -1
                    pixel_value2 = (pixel_array[x-1][i+1] + (pixel_array[x][i+1] *2) + pixel_array[x+1][i+1])
                    final_value = (pixel_value1 + pixel_value2) / 8
                    horizontal_edges[x][i] = abs(final_value)

    

    
    
    
    for j in range(len(horizontal_edges)):
        if j == 0 or j == image_height-1:
            horizontal_edges[j] = [0.0] * image_width
        else:
            for i in range(len(horizontal_edges[j])):
                if i == 0 or i == image_width -1:
                    horizontal_edges[j][i] = 0.0
                
            
    return horizontal_edges

#Compute the horizontal sobel edge component
def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    horizontal_edges = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for x in range(len(pixel_array)):
        if x != 0 and x != image_height-1:
            for i in range(len(pixel_array[x])):
                if i != 0 and i != image_width -1:
                    pixel_value1 = (pixel_array[x-1][i-1] + (pixel_array[x+1][i-1]*-1))
                    pixel_value2 = ((pixel_array[x-1][i]*2) + (pixel_array[x+1][i]*-2))
                    pixel_value3 = (pixel_array[x-1][i+1] + (pixel_array[x+1][i+1]*-1))
                    final_value = (pixel_value1 + pixel_value2 + pixel_value3) / 8
                    horizontal_edges[x][i] = abs(final_value)

    

    
    
    
    for j in range(len(horizontal_edges)):
        if j == 0 or j == image_height-1:
            horizontal_edges[j] = [0.0] * image_width
        else:
            for i in range(len(horizontal_edges[j])):
                if i == 0 or i == image_width -1:
                    horizontal_edges[j][i] = 0.0
                
            
    return horizontal_edges


#Compute Gaussian Averaging for each pixel in 3x3 format
def computeGaussianAveraging3x3(pixel_array, image_width, image_height):
    horizontal_edges = createInitializedGreyscalePixelArray(image_width, image_height)
    median_list = []
    for x in range(image_height):

        for i in range(image_width):


            median_list.append(pixel_array[x][i] * 4)
            if i+1 in range(image_width):
                right_pixels = pixel_array[x][i+1] * 2
            else:
                right_pixels = pixel_array[x][i] * 2
                
            if i-1 in range(image_width):
                left_pixels = pixel_array[x][i-1] * 2
            else:
                left_pixels = pixel_array[x][i] * 2
            
            median_list.append(right_pixels)
            median_list.append(left_pixels)
            
            if x+1 in range(image_height):
                median_list.append(pixel_array[x+1][i] *2)
                if i+1 in range(image_width):
                    right_pixels = pixel_array[x+1][i+1]
                else:
                    right_pixels = pixel_array[x+1][i]
                    
                if i-1 in range(image_width):
                    left_pixels = pixel_array[x+1][i-1]
                else:
                    left_pixels = pixel_array[x+1][i]
                
                median_list.append(right_pixels)
                median_list.append(left_pixels)
                
            else:
                median_list.append(pixel_array[x][i] * 2)
                if i-1 in range(image_width):
                    median_list.append(pixel_array[x][i-1])
                else:
                    median_list.append(pixel_array[x][i])
                if i+1 in range(image_width):
                    median_list.append(pixel_array[x][i+1])
                else:
                    median_list.append(pixel_array[x][i])
            
            if x-1 in range(image_height):
                median_list.append(pixel_array[x-1][i] * 2)
                if i+1 in range(image_width):
                    right_pixels = pixel_array[x-1][i+1]
                else:
                    right_pixels = pixel_array[x-1][i]
                    
                if i-1 in range(image_width):
                    left_pixels = pixel_array[x-1][i-1]
                else:
                    left_pixels = pixel_array[x-1][i]
                
                median_list.append(right_pixels)
                median_list.append(left_pixels)
                
            else:
                median_list.append(pixel_array[x][i] * 2)
                if i-1 in range(image_width):
                    median_list.append(pixel_array[x][i-1])
                else:
                    median_list.append(pixel_array[x][i])
                if i+1 in range(image_width):
                    median_list.append(pixel_array[x][i+1])
                else:
                    median_list.append(pixel_array[x][i])
                

                

            horizontal_edges[x][i] = sum(median_list)/16
            median_list = []
    return horizontal_edges

#Compute pixel thresholding by the determined value
def computeThreshold(pixel_array, threshold_value, image_width, image_height):
    for row in pixel_array:
        for i in range(len(row)):
            if row[i] < threshold_value:
                row[i] = 0
            else:
                row[i] = 255
                
    return pixel_array


#Absolute difference between vertical and horizonal sobel edge operations
def sobelDifference(vsobel, hsobel, image_width, image_height):
    pixel_arr = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(image_height):
        for i in range(image_width):
            pixel_arr[x][i] = abs(vsobel[x][i] - hsobel[x][i])

    return pixel_arr

#Dialiate pixel in 3x3 format
def computeDilation3x3(pixel_array, image_width, image_height):
    return_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for j in range(image_height):
        for i in range(image_width):
            if pixel_array[j][i] > 0:
                return_array[j][i] = 1
                if i-1 in range(image_width):
                    return_array[j][i-1] = 1
                    if j-1 in range(image_height):
                        return_array[j-1][i] = 1
                        return_array[j-1][i-1] = 1
                        if i + 1 in range(image_width):
                            return_array[j-1][i+1] = 1
                            
                if i+1 in range(image_width):
                    return_array[j][i+1] = 1
                    if j+1 in range(image_height):
                        return_array[j+1][i] = 1
                        return_array[j+1][i+1] = 1
                        if i - 1 in range(image_width):
                            return_array[j+1][i-1] = 1
                            
    return return_array

#Erode pixel in 3x3 format
def computeErosion3x3(pixel_array, image_width, image_height):
    return_array = createInitializedGreyscalePixelArray(image_width+2, image_height+2, 1)
    for x in pixel_array:
        x.insert(0,0)
        x.append(0)
    addon = [[0] * (image_width+2)]
    endadd = [0] * (image_width+2)
    pixel_array.append(endadd)
    border = addon+pixel_array

    for j in range(image_height+2):
        for i in range(image_width+2):
            if border[j][i] == 0:
                return_array[j][i] = 0
                if i-1 in range(image_width+2):
                    return_array[j][i-1] = 0
                    if j-1 in range(image_height+2):
                        return_array[j-1][i] = 0
                        return_array[j-1][i-1] = 0
                        if i + 1 in range(image_width+2):
                            return_array[j-1][i+1] = 0
                            
                if i+1 in range(image_width+2):
                    return_array[j][i+1] = 0
                    if j+1 in range(image_height+2):
                        return_array[j+1][i] = 0
                        return_array[j+1][i+1] = 0
                        if i - 1 in range(image_width+2):
                            return_array[j+1][i-1] = 0
    for k in range(len(return_array)):
        return_array[k] = return_array[k][1:-1]
                    
                            
    return return_array[1:-1]

#Defining queue class for calculating connected components
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
#Find connected components
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):

    my_dict = {}
    visited = {}
    
    q = Queue()
    
    current_label = 1

    for j in range(image_height):
        visited[j] = []
    
    for j in range(image_height):
        for i in range(image_width):
            
            if pixel_array[j][i] > 0 and i not in visited[j]:
                value = 0
                q.enqueue([j, i])
                while not q.isEmpty():
                    pixel = q.dequeue() 
                    x = pixel[0]
                    y = pixel[1]   

                    if y not in visited[x]:
                        visited[x].append(y)                           
                        value+=1                                                                      
                        pixel_array[x][y] = current_label
                                  
                        if x-1 in range(image_height):
                            if pixel_array[x-1][y] > 0:
                                q.enqueue([x-1, y])    
    
                        if x+1 in range(image_height):
                            if pixel_array[x+1][y] > 0:
                                q.enqueue([x+1 , y])       
                                
                        if y-1 in range(image_width):
                            if pixel_array[x][y-1] > 0:
                                q.enqueue([x , y-1])
                                
                        if y+1 in range(image_width):
                            if pixel_array[x][y+1] > 0:
                                q.enqueue([x, y+1])
                            
                my_dict[current_label] = value
                current_label += 1

    return pixel_array, my_dict


#Retrieve the pixel boundary coordinates for the largest connected component (or component with that satisfies requirements)
def largest_component_coord(pixel_array, image_height, image_width, key1):
    min_y = image_height - 1
    min_x = image_width - 1
    max_y = 0
    max_x = 0
    
    for j in range(image_height):
        for i in range(image_width):
            if pixel_array[j][i] == key1:
                if j < min_y:
                    min_y = j
                if i < min_x:
                    min_x = i
                if j > max_y:
                    max_y = j
                if i > max_x:
                    max_x = i


    return min_x, min_y, max_x, max_y

#Converge r, g and b arrays back into one RGB array
def separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height):
    new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]
    for y in range(image_height):
        for x in range(image_width):
            new_array[y][x][0] = px_array_r[y][x]
            new_array[y][x][1] = px_array_g[y][x]
            new_array[y][x][2] = px_array_b[y][x]
    return new_array

#Calculate pixel density within bounding box
def calculate_density(pixel_array, image_height, image_width, min_x, min_y, max_x, max_y, key):
    target_pixels = 0
    total_pixels = (max_x - min_x) * (max_y - min_y)

    for j in range(min_y, max_y + 1):
        for i in range(min_x, max_x + 1):
            if pixel_array[j][i] == key:
                target_pixels += 1

    density = (target_pixels/total_pixels) * 100

    return density
            
# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    #Create an array of images that were captured by the onboard camera and process the barcode one at a time
    images = []
    for filename in os.listdir("extension_input_images/"):
        images.append(filename[:-4])
        print(filename[:-4])

    for filename in images:
        


        command_line_arguments = sys.argv[1:]

        SHOW_DEBUG_FIGURES = True

        # this is the default input image filename

        #Import from the local directory
        input_filename = "extension_input_images/"+filename+".png"

        if command_line_arguments != []:
            input_filename = command_line_arguments[0]
            SHOW_DEBUG_FIGURES = False
            
        #Save to a local output directory
        output_path = Path("extension_output_images")
        if not output_path.exists():
            # create output directory
            output_path.mkdir(parents=True, exist_ok=True)

        output_filename = output_path / Path(filename+"_extension_output.png")
        if len(command_line_arguments) == 2:
            output_filename = Path(command_line_arguments[1])

        # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
        # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
        (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

        # setup the plots for intermediate results in a figure
        fig1, axs1 = pyplot.subplots(2, 2)
        axs1[0, 0].set_title('Input red channel of image')
        axs1[0, 0].imshow(px_array_r, cmap='gray')
        axs1[0, 1].set_title('Input green channel of image')
        axs1[0, 1].imshow(px_array_g, cmap='gray')
        axs1[1, 0].set_title('Input blue channel of image')
        axs1[1, 0].imshow(px_array_b, cmap='gray')


        # STUDENT IMPLEMENTATION here
        
        #Convert pixel array to greyscale array
        greyscale_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

        #Scale pixels to 0 and 255
        scaled_array = scaleTo0And255(greyscale_array, image_width, image_height)

        #Carry out vertical and horizontal sobel edge calculations and take the difference
        horizontal_sobel = computeHorizontalEdgesSobelAbsolute(scaled_array, image_width, image_height)
        vertical_sobel = computeVerticalEdgesSobelAbsolute(scaled_array, image_width, image_height)
        sobel_filtered = sobelDifference(vertical_sobel, horizontal_sobel, image_width, image_height)

        #Apply a series of 4 3x3 Gaussian Average filters
        gaussian_filtered = computeGaussianAveraging3x3(sobel_filtered, image_width, image_height)
        gaussian_filtered = computeGaussianAveraging3x3(gaussian_filtered, image_width, image_height)
        gaussian_filtered = computeGaussianAveraging3x3(gaussian_filtered, image_width, image_height)
        gaussian_filtered = computeGaussianAveraging3x3(gaussian_filtered, image_width, image_height)

        #Perfom a thresholding on the image with a value of 6 (I found this worked best)
        thresholded = computeThreshold(gaussian_filtered, 6, image_width, image_height)

        #A series of 2 preliminary erosions to isolate barcode area, I found this worked best for the specific camera I used
        eroded = computeErosion3x3(thresholded, image_width, image_height)
        eroded = computeErosion3x3(eroded, image_width, image_height)

        #2 Dialations to connect areas of the barcode
        dialated = computeDilation3x3(eroded, image_width, image_height)
        dialated = computeDilation3x3(dialated, image_width, image_height)

        #A series of 3 erosions to isolate barcode area from any areas in which the dialations above effected
        eroded = computeErosion3x3(dialated, image_width, image_height)
        eroded = computeErosion3x3(eroded, image_width, image_height)
        eroded = computeErosion3x3(eroded, image_width, image_height)



        #Calculate connected components
        (ccimg,ccsizes) = computeConnectedComponentLabeling(eroded,image_width,image_height)


            

        #Converge seperate rays so RGB image can be outputted
        finalRGBArray = separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)
        px_array = finalRGBArray




        # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
        # Change these values based on the detected barcode region from your algorithm

        #Sort the connected components by largest to smallest by number of pixels
        sorted_ccsizes = sorted(ccsizes.items(), key = lambda x:x[1], reverse = True)
        ccsizes_descending = dict(sorted_ccsizes)

        #Iterate through connected components and find the one that satisfies the barcode aspect ratio
        for keyz in ccsizes_descending:
            (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y) = largest_component_coord(ccimg, image_height, image_width, keyz)
            longest_side = max(bbox_max_x - bbox_min_x,  bbox_max_y - bbox_min_y)
            shortest_side = min(bbox_max_x - bbox_min_x,  bbox_max_y - bbox_min_y)
            if longest_side ==0 or shortest_side == 0:
                print("Cannot identify barcode, make sure the barcode is in view and clear")
                break
            density = calculate_density(ccimg, image_height, image_width, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, keyz)          

            if longest_side/shortest_side <= 2 and density > 40:
                break



        # The following code is used to plot the bounding box and generate an output for marking
        # Draw a bounding box as a rectangle into the input image
        axs1[1, 1].set_title('Final image of detection')
        axs1[1, 1].imshow(px_array, cmap='gray')
        rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                         edgecolor='m', facecolor='none')
        axs1[1, 1].add_patch(rect)

        # write the output image into output_filename, using the matplotlib savefig method
        extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
        pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

if __name__ == "__main__":
    main()

