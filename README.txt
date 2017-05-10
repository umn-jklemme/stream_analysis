Written in C by Joel Klemme 3765483

Two code sections are available and are individually made.

Both require a compilation of the Free Image library:
http://freeimage.sourceforge.net/
Depending on the compiler, some adjustments to switch single quote (') and
double quote (") may need to be made.


The GPU code does not need FreeImage to run, so if it is troublesome to install,
strip out the references to compile image-analysis.c in the makefile.

However, the GPU section is made to produce outputs that might need to be read
using the program GIMP: https://www.gimp.org/

---convert_to_bmp---
convert_to_bmp can be made using a gcc compiler

It can be run by using

image-converter -i <image_file> -o <output_file.bmp>

or running

image-converter -h

for help

---image_analysis---

Running the containing makefile should produce image-analysis-cpu and
image-analysis-gpu output executables.

The image-analysis-cpu runs the original c code and can be called with

image-analysis-cpu -i <input_file>

or

image-analysis-cpu -h

for help

For convenience, some example inputs are provided in the input folder.

The image-analysis-gpu runs the ported c code as well as the gpu kernel code.

The ported cpu can be run with

image-analysis-gpu -i <input_file> -o <output_file> -m

NOTE: input files for image-analysis-gpu must be in the uncompressed format

P3 <WIDTH> <HEIGHT> 255
<FIRST_PIXEL_RED> <FIRST_PIXEL_BLUE> <FIRST_PIXEL_GREEN>
<SECOND_PIXEL_RED> <SECOND_PIXEL_BLUE> <SECOND_PIXEL_GREEN>

An example file is provided in input_bmp

---

Example:

Run the Original with:

image-analysis-cpu -i input/X-Rite-Photo.jpg

Run the Ported Original (CPU) with:

image-analysis-gpu -i input_bmp/X-Rite-Photo.bmp -o original-output.bmp -m

... and viewing original-output.bmp with GIMP

Run the CUDA Kernelized version with:

image-analysis-gpu -i input_bmp/X-Rite-Photo.bmp -o original-output.bmp

... and viewing the kernelized-output.bmp with GIMP

