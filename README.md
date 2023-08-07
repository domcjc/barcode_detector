# barcode_detector
Uses user dekstop camera to capture a image and apply a series of filters and gradients to detect presence of a barcode and draw a boundary box.

Dominic Coleman – dcol279

I wanted to extend my original barcode detection  pipeline to allow users to take as many photos of items
they wish to scan and then immediately run all these photos through the barcode detection
program, and saving the output.

For this I decided to use OpenCV as it allows me to utilize the computers camera or an external USB
camera to take photos of products that wish to be scanned. I created a window that opens when the
program is run and displays the camera picture. From here, the user can then take as many photos of
products that they wish to scan.

As these photos are taken (by the user pressing ‘Space’) they are saved to a local directory,
continuing until the user presses ‘Esc’ key which quits the capture window and then begins the
processing of the captured images.

The barcode detection part of the program then runs for each image captured by the user, and
outputs the detected image to another local directory. This continues until all images have been
processed.

One key issue I had with this implementation was that I only had access to one webcam that
captured the images (which is of particularly poor quality) therefore I had to carry out heavier image
processing to isolate the barcode region. This can be seen by the series of erosions, dilations and
final erosions. I ended up being able to detect majority of barcodes I captured with my webcam,
however this is may vary from machine to machine.


To run install the following libraries and ensure a camera (either internal or external USB is present):
pip install opencv-python
pip install opencv-contrib-python
