#include <fstream>
#include <stdio.h> 
#include <stdlib.h> 
#include <mpi.h>
#include <string>
#include <iostream>
#include "win-gettimeofday.h"

using namespace std;

//Resolution
#define WIDTH 1080 /* Width of Mandelbrot set Matrix Image */			
#define HEIGHT 1080 /* Height of Mandelbrot set Matrix Image */

#define MaxRGB 256 //Max RGB value

//Data type declaration of a RGB
typedef struct {
	unsigned int red;
	unsigned int green;
	unsigned int blue;
} RGB;

//Data type declaration of a Mandelbrot
typedef struct {
	RGB* image;
	unsigned int width;
	unsigned int height;
} Mandelbrot;

void checkFunctions(bool local_check, string fname, string message, MPI_Comm comm);
void splitLengths(int *local_length, int lenght, int comm_size, MPI_Comm comm);
void allocateMemoryC(double** local_c, int local_length, MPI_Comm comm);
void fillCvalues(double* c, double minVal, double maxVal, int rank, int local_length, int length, MPI_Comm comm);
void getCValues(double* c, int index, int state, double beginRange, double endRange, double minVal, double maxVal, int local_length);
void gatherAllCValues(double* c, double* local_c, int length, int local_length, MPI_Comm comm);
void allocateMemoryRGB(RGB** local_rgb, int local_length, MPI_Comm comm);
void parrMandelbrot(RGB* local_rgb, double* ci, double* cr, int local_length, int width, int height, int rank);
RGB colouringMandelbrot(int i, int maxIterations);
void CreatePPMImage(RGB* local_rgb, int local_rgb_length, int width, int height, int dim, int maxRGB, int rank, MPI_Comm comm);

int main(int argc, char **argv) {

	//Set default values to 0
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int maxRGB = 0;

	if (argc < 2) {
		width = WIDTH; //Set Width value
		height = HEIGHT; //Set Height value
		maxRGB = MaxRGB; //Set MaxRGB value
	}
	else {
		width = atoi(argv[1]); //Set Width value
		height = atoi(argv[2]); //Set Height value
		maxRGB = atoi(argv[3]); //Set MaxRGB Value
	}

	//Local length of cr and ci vectors
	int local_cr_length = 0;
	int local_ci_length = 0;

	//Public cr and ci vectors
	double* cr;
	double* ci;

	double* local_cr;//C local real values vector
	double *local_ci; //C local imaginariy values vector

					  //Set the range of the mandelbrot set for c Real number and Imaginari values (Zoom in, Zoom out)
	double minValR = -2.0;
	double maxValR = 1.0;
	double minValI = -1.5;
	double maxValI = 1.5;

	int comm_size; //Number of Processes in Communicator
	int rank; // Rank process


	//Initialize communicator
	MPI_Comm comm;
	MPI_Init(&argc, &argv);
	//Assign the comm_worl communicator to the initialized comm object
	comm = MPI_COMM_WORLD;
	//Get communicator size
	MPI_Comm_size(comm, &comm_size);
	//Get the Rank processs
	MPI_Comm_rank(comm, &rank);

	//Wait for all to execute at the same time
	MPI_Barrier(comm);

	//Start timer Global time for all funcitons
	double startTimeMPI = get_current_time();

	//Wait for all to execute at the same time
	MPI_Barrier(comm);

	//Start timer Global time for all funcitons except the ppm file creation
	double startTimeParr = get_current_time();

	//Get lengths of Ci and Cr values
	splitLengths(&local_cr_length, width, comm_size, comm);
	splitLengths(&local_ci_length, height, comm_size, comm);

	//Dynamic allocate memory
	allocateMemoryC(&local_cr, local_cr_length, comm);
	allocateMemoryC(&local_ci, local_ci_length, comm);

	//Fill local ci and cr values vectors
	fillCvalues(local_cr, minValR, maxValR, rank, local_cr_length, width, comm);
	fillCvalues(local_ci, minValI, maxValI, rank, local_ci_length, height, comm);

	//Dynamic allocate memory for c vector
	allocateMemoryC(&cr, width, comm);
	allocateMemoryC(&ci, height, comm);

	//Gather all ci and cr vectors
	gatherAllCValues(cr, local_cr, width, local_cr_length, comm);
	gatherAllCValues(ci, local_ci, height, local_ci_length, comm);

	
	//Initialize RGB
	RGB* local_rgb;
	//Local rgb length
	int local_rgb_length = 0;
	//Dimensions of the mandelbrot set
	int dim = width * height;

	//Split lengths for mandelbrot Set
	splitLengths(&local_rgb_length, dim, comm_size, comm);

	//Dynamic allocate memory for RGB vector
	allocateMemoryRGB(&local_rgb, local_rgb_length, comm);

	//Mandelbrot set function get RGB 
	parrMandelbrot(local_rgb, ci, cr, local_rgb_length, width, height, rank);

	//Wait for all to execute at the same time
	MPI_Barrier(comm);

	//Start timer Global time for all funcitons except the ppm file creation
	double endTimeParr = get_current_time();

	if (rank == 0) {
		printf("Time for the creation of C values and RGB for image: %lfs\n", endTimeParr - startTimeParr);
	}

	//Create PPM image file
	CreatePPMImage(local_rgb, local_rgb_length, width, height, dim, maxRGB, rank, comm);

	//Wait all to finish
	MPI_Barrier(comm);
	//Start timer Global time
	double endTimeMPI = get_current_time();
	

	if (rank == 0) {
		printf("Time for the hole execution: %lfs\n", endTimeMPI - startTimeMPI);
	}

	MPI_Finalize();
	return 0;
}


void checkFunctions(
	bool local_check /* in */,
	string fname /* in */,
	string message /* in */,
	MPI_Comm comm /* in */)
{
	//Store the final boolean value
	bool check;

	//Check all booleans with logical AND operator At least one false will finalize MPI
	MPI_Allreduce(&local_check, &check, 1, MPI_C_BOOL, MPI_LAND, comm);

	if (local_check) {
		int rank;
		MPI_Comm_rank(comm, &rank);
		if (rank == 0) {
			cout << stderr << "Process: " << rank << " -> in: " << fname << ", " << message << endl;
			fflush(stderr);
		}
		MPI_Finalize();
		exit(-1);
	}
}

void splitLengths(
	int *local_length, //Out
	int length, //in
	int comm_size, // in
	MPI_Comm comm /*in*/)
{

	bool local_check = false;
	const string fname = "Split Lengths";
	string message = "";

	//Check if length is divisible by the size of the communicator
	if (length % comm_size != 0) {
		local_check = true;
		message = "Length (Width,Height) must be divisible by the size of communicator";
	}

	checkFunctions(local_check, fname, message, comm);

	*local_length = length / comm_size;
}

void allocateMemoryC(
	double** local_c /*out*/
	, int local_length /*in*/
	, MPI_Comm comm /*in*/)
{

	const string fname = "Allocate Memory C values";
	bool local_check = false;
	string message = "";
	//Dynamic allocate memory for each process to local c vector
	*local_c = (double*)malloc(local_length * sizeof(double));

	if (*local_c == NULL) {
		local_check = true;
		message = "Cannot allocate memory local vector c";
	}

	checkFunctions(local_check, fname, message, comm);
}

void fillCvalues(
	double* local_c /*out*/
	, double minVal /*in*/
	, double maxVal /*in*/
	, int rank /*in*/
	, int local_length /*in*/
	, int length /*in*/
	, MPI_Comm comm /*in*/)
{
	const string fname = "Fill C Values";
	string message = "";
	bool local_check = false;
	//Get the state where to begin
	int state = rank * local_length;
	getCValues(local_c, 0, state, 0, length, minVal, maxVal, local_length);

	//Check the vallues that have been assign if is correct
	if (local_c[0] < minVal || local_c[local_length - 1] > maxVal) {
		local_check = true;
		message = "C Vallues didn't generated correctly";
	}

	checkFunctions(local_check, fname, message, comm);
}

//Function the fill the c values Recursively
void getCValues(
	double* local_c, /*out*/
	int index, /*in*/
	int state, /*in*/
	double beginRange, /*in*/
	double endRange, /*in*/
	double minVal, /*in*/
	double maxVal, /*in*/
	int local_length /*in*/)
{
	//Check if we are on the last state on end Range
	if (index < local_length) {
		//Fill c values by breaking into equal part the range between minVal and maxVal by the state and the range between Begin and End
		local_c[index] = ((state - beginRange) / (endRange - beginRange))*(maxVal - minVal) + minVal;
		//call the functio itself
		return getCValues(local_c, index + 1, state + 1, beginRange, endRange, minVal, maxVal, local_length);
	}
	else {
		//End function
		return;
	}
}


void gatherAllCValues(
	double* c, /*out*/
	double* local_c, /*in*/
	int length, /*in*/
	int local_length, /*in*/
	MPI_Comm comm /*in*/) {

	//Gather all cr vectors and ci vectors in one vector to all process
	MPI_Allgather(local_c, local_length, MPI_DOUBLE, c, local_length, MPI_DOUBLE, comm);
}


void allocateMemoryRGB(
	RGB** local_rgb /*out*/
	, int local_length /*in*/
	, MPI_Comm comm /*in*/)
{

	const string fname = "Allocate Memory RGB values";
	bool local_check = false;
	string message = "";

	//Dynami allocate memory to each processe to local rgb vector
	*local_rgb = (RGB*)malloc(local_length * sizeof(RGB));

	if (*local_rgb == NULL) {
		local_check = true;
		message = "Cannot allocate memory local vector rgb";
	}

	checkFunctions(local_check, fname, message, comm);
}


void parrMandelbrot(
	RGB* local_rgb,/*out*/
	double* ci, /*in*/
	double* cr, /*in*/
	int local_length, /*in*/
	int width, /*in*/
	int height, /*in*/
	int rank /*in*/)
{
	//Get the startIndex of the vector for the local process
	int indexStart = rank * local_length;
	//Get the start Row and Collumn of the local process
	int rowStart = (int)(indexStart / width);
	int colStart = indexStart % width;
	//Get the last index of the vector for the local process
	int indexEnd = indexStart + local_length;
	//Get the end Row and Collumn of the local process
	int rowEnd = (int)(indexEnd / width);
	int colEnd = indexEnd % width;

	//Create variables for the collumn iteration
	int colStartI = 0;
	int colEndI = width;

	/*Mandelbort Set Function and Calculations
	F(z) = z^2 + c
	Where c is a complex number
	Z start from 0 So f(z)1 = c;
	c = a + bi
	Where a and b are real numbers and i is an imaginari number
	(a + bi)^2 = a^2 + 2*abi + (bi)^2
	i is sqrt(-1) so sqrt(-1) ^ 2 = -1
	So z^2 = (a^2 + 2*bi - b)
	F(z)2 = (a^2 + 2*bi - b) + (a + bi)
	*/
	int index = 0;
	int i = 0;
	//Create variables to store the z Real values and Z imaginari values
	double zr = 0.0;
	double zi = 0.0;
	double fz = 0.0;
	//How many times loop to find if number is increasing to infinite
	const int maxIterations = 500;

	for (int row = rowStart; row < rowEnd; row++) {
		//Check if is the first row to assign the colStart or else assign from begin value
		if (row == rowStart) {
			colStartI = colStart;
		}
		else {
			colStartI = 0;
		}

		//Check if is the lass row to assign the final column
		if (row == rowEnd - 1 && colEnd != 0) {
			colEndI = colEnd;
		}

		for (int col = colStartI; col < colEndI; col++) {
			//Reset values
			i = 0; zr = 0.0; zi = 0.0;
			//Calculate mandelbrot algorithm f(z) function 
			while (i < maxIterations && zr * zr + zi * zi < 4.0) {
				//Calculate fz by using the c real value
				fz = zr * zr - zi * zi + cr[col];
				//Calculate the z imaginari value using the c imaginari value
				zi = 2.0 * zr * zi + ci[row];
				//Store new z real value to zr
				zr = fz;
				//+1 Next itteration
				i++;
			}
			//Assign the colour tou our rgb array
			local_rgb[index] = colouringMandelbrot(i, maxIterations);
			index++;
		}
	}
}


RGB colouringMandelbrot(int i, int maxIterations) {
	//Create an Instance of RGB
	RGB rgb;
	//Create variables to store r g b values
	int r, g, b;

	//Colouring the mandelbrot set image
	int maxRGB = 256;
	int max3 = maxRGB * maxRGB * maxRGB;
	double t = (double)i / (double)maxIterations;
	i = (int)(t * (double)max3);
	g = i / (maxRGB * maxRGB);
	int nn = i - g * maxRGB * maxRGB;
	b = nn / maxRGB;
	r = nn - b * maxRGB;

	//Save Red Green Blue colours to RGB
	rgb.red = r;
	rgb.green = g;
	rgb.blue = b;
	return rgb;
}


void CreatePPMImage(
	RGB* local_rgb, /*in*/
	int local_rgb_length, /*in*/
	int width, /*in*/
	int height, /*in*/
	int dim, /*in*/
	int maxRGB, /*in*/
	int rank, /*in*/
	MPI_Comm comm /*in*/)
{
	//Initialize mandelbrot
	Mandelbrot mandelbrot;

	//Dynamic allocate memory for RGB vector
	allocateMemoryRGB(&mandelbrot.image, dim, comm);

	//Create a Custom Datatype RGB for collecting our rgb values from local vectors
	MPI_Datatype MPI_RGB;
	MPI_Type_contiguous(3, MPI_UNSIGNED, &MPI_RGB);
	MPI_Type_commit(&MPI_RGB);

	//Gather all rgb vector to root proccess 0 and create it to the mandelbrot image
	if (rank == 0) {
		//Gather all local rgb vectors one vector to the 0 root process
		MPI_Gather(local_rgb, local_rgb_length, MPI_RGB, mandelbrot.image, local_rgb_length, MPI_RGB, 0, comm);

		//Start timer PPM FILE
		double startTimePPM = get_current_time();

		printf("Creating PPM image File...\n");

		//Create a PPM image file
		ofstream fout("output_image.ppm");
		//Set it to be a PPM file
		fout << "P3" << endl;
		//Set the Dimensions
		fout << width << " " << height << endl;
		//Max RGB Value
		fout << maxRGB << endl;

		//Fill the image with rgb values
		for (int h = 0; h < height; h++) {
			//Unrolling Loop check less and store tow elements per loop Width must be divisible by 2
			for (int w = 0; w < width; w += 2) {
				//Calculate Index
				int index = h * width + w;
				//Store in image every RGB pixel
				fout << mandelbrot.image[index].red << " " << mandelbrot.image[index].green << " " << mandelbrot.image[index].blue << " ";
				fout << mandelbrot.image[index + 1].red << " " << mandelbrot.image[index + 1].green << " " << mandelbrot.image[index + 1].blue << " ";
			}
			fout << endl;
		}

		fout.close();

		//End timer PPM file Created
		double endTimePPM = get_current_time();
		printf("Time creating PPM file: %lfs\n", endTimePPM - startTimePPM);

		printf("Done!! The PPM image file is ready!\n");
	}
	else {
		//Gather all local rgb vectors one vector to the 0 root process
		MPI_Gather(local_rgb, local_rgb_length, MPI_RGB, mandelbrot.image, local_rgb_length, MPI_RGB, 0, comm);
	}

	//free allocated memory
	free(mandelbrot.image);
}