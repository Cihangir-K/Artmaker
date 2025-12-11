Art Maker: Python-Powered Paint-by-Number Generator

Art Maker is a desktop application built using Python (specifically Tkinter for the GUI) that leverages computer vision techniques to convert complex raster images into simplified, numbered coloring templates.

Core Functionality
Image Input & Parameter Setting (Tkinter GUI): The user selects an image file and defines the desired final color count (pieces count) via a user-friendly interface.

Color Quantization (OpenCV & K-Means): The application applies the K-Means clustering algorithm to reduce the image's color depth to the specified number of primary colors (N). This creates the foundational simplified image.

Interactive Preview: A separate window displays the quantized image, allowing the user to zoom and pan (using the mouse wheel and scrollbars) to assess the color simplification quality before committing to the output. This window includes options to approve the current settings or retry with a new color count.

Reference Generation: Upon approval, Art Maker generates and saves three essential print-ready files:

Numbered Color Reference: The simplified image with the unique number of its corresponding color placed clearly in each colored area.

Numbered Line-Art Template: A high-contrast, black-and-white outline of the color zones, also marked with the corresponding numbers, ideal for printing onto canvas or paper.

Visual Color Palette: A separate sheet displaying each numbered color in a solid block alongside its official RGB color code, serving as a mixing guide.

Technology Stack: Python 3.10+, Tkinter, OpenCV, NumPy, Pillow.
