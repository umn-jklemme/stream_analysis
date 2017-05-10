#include <cstdio>
#include <cstdint>
extern "C"
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "image-analysis.h"
#include "kernel.cu"
#include "support.h"



static void print_usage(char * appname)
{
    printf("Usage: %s [options]\n", appname);
    printf("Available options are\n");
    printf(" -i <image_file>    The image to load\n");
    printf(" -o <output_file>   An output image to print\n");
    printf(" -m                 Specify to run on CPU\n");
    printf(" -h                 Print this help screen and exit\n");
}


__host__ __device__ int pixel_similar(uint32_t pixel_a, uint32_t pixel_b, int max_diff_luma, int max_diff_chroma)
{
    unsigned char a_red, a_gre, a_blu;
    unsigned char b_red, b_gre, b_blu;
    int a_Y_val, a_Cb_val, a_Cr_val;
    int b_Y_val, b_Cb_val, b_Cr_val;
    a_red = (unsigned char)(( pixel_a & RED_MASK) >> 16); /* Red */
    a_gre = (unsigned char)(( pixel_a & GREEN_MASK) >> 8);  /* Green */
    a_blu = (unsigned char)(( pixel_a & BLUE_MASK));       /* Blue */

    b_red = (unsigned char)(( pixel_b & RED_MASK) >> 16); /* Red */
    b_gre = (unsigned char)(( pixel_b & GREEN_MASK) >> 8);  /* Green */
    b_blu = (unsigned char)(( pixel_b & BLUE_MASK));       /* Blue */
    a_Y_val = 16 + (0.257*a_red) + (0.504*a_gre) + (0.098*a_blu);
    if (a_Y_val < 16) a_Y_val = 16;
    else if (a_Y_val > 235) a_Y_val = 235;
    a_Cb_val = 128 - (0.148*a_red) - (0.291*a_gre) + (0.439*a_blu);
    if (a_Cb_val < 16) a_Cb_val = 16;
    else if (a_Cb_val > 240) a_Cb_val = 240;
    a_Cr_val = 128 + (0.439*a_red) - (0.368*a_gre) - (0.071*a_blu);
    if (a_Cr_val < 16) a_Cr_val = 16;
    else if (a_Cr_val > 240) a_Cr_val = 240;

    b_Y_val = 16 + (0.257*b_red) + (0.504*b_gre) + (0.098*b_blu);
    if (b_Y_val < 16) b_Y_val = 16;
    else if (b_Y_val > 235) b_Y_val = 235;
    b_Cb_val = 128 - (0.148*b_red) - (0.291*b_gre) + (0.439*b_blu);
    if (b_Cb_val < 16) b_Cb_val = 16;
    else if (b_Cb_val > 240) b_Cb_val = 240;
    b_Cr_val = 128 + (0.439*b_red) - (0.368*b_gre) - (0.071*b_blu);
    if (b_Cr_val < 16) b_Cr_val = 16;
    else if (b_Cr_val > 240) b_Cr_val = 240;

    if (abs(a_Y_val - b_Y_val) > max_diff_luma) {
        return 0;
    }
    else if (abs(a_Cb_val - b_Cb_val) > max_diff_chroma) {
        return 0;
    }
    else if (abs(a_Cr_val - b_Cr_val) > max_diff_chroma) {
        return 0;
    }

    return 1;
}

__host__ __device__ int pixel_similar_ycbcr(uint32_t pixel_a, uint32_t pixel_b, int max_diff_luma, int max_diff_chroma)
{
    unsigned char a_Y_val, a_Cb_val, a_Cr_val;
    unsigned char b_Y_val, b_Cb_val, b_Cr_val;
    a_Y_val = (unsigned char)(( pixel_a & Y_MASK) >> 16);  /* Y */
    a_Cb_val = (unsigned char)(( pixel_a & CB_MASK) >> 8); /* Cb */
    a_Cr_val = (unsigned char)(( pixel_a & CR_MASK));      /* Cr */

    b_Y_val = (unsigned char)(( pixel_b & Y_MASK) >> 16);  /* Y */
    b_Cb_val = (unsigned char)(( pixel_b & CB_MASK) >> 8); /* Cb */
    b_Cr_val = (unsigned char)(( pixel_b & CR_MASK));      /* Cr */

    if (abs(a_Y_val - b_Y_val) > max_diff_luma) {
        return 0;
    }
    else if (abs(a_Cb_val - b_Cb_val) > max_diff_chroma) {
        return 0;
    }
    else if (abs(a_Cr_val - b_Cr_val) > max_diff_chroma) {
        return 0;
    }

    return 1;
}

__host__ __device__ uint32_t rgb_to_ycbcr(uint32_t pixel)
{
    unsigned char red, gre, blu;
    unsigned char Y_val, Cb_val, Cr_val;
    red = (unsigned char)(( pixel & Y_MASK) >> 16);  /* Y */
    gre = (unsigned char)(( pixel & CB_MASK) >> 8);  /* CB */
    blu = (unsigned char)(( pixel & CR_MASK));       /* CR */

    Y_val = 16 + (0.257*red) + (0.504*gre) + (0.098*blu);
    if (Y_val < 16) Y_val = 16;
    else if (Y_val > 235) Y_val = 235;
    Cb_val = 128 - (0.148*red) - (0.291*gre) + (0.439*blu);
    if (Cb_val < 16) Cb_val = 16;
    else if (Cb_val > 240) Cb_val = 240;
    Cr_val = 128 + (0.439*red) - (0.368*gre) - (0.071*blu);
    if (Cr_val < 16) Cr_val = 16;
    else if (Cr_val > 240) Cr_val = 240;

    uint32_t ycbcr = 0x00 << 24 | Y_val << 16 | Cb_val << 8 | Cr_val;

    return ycbcr;
}



int import_bmp(char* import_name, uint32_t* pixels, uint32_t* tp_pixels) {
    FILE *bmp_fd;
    bmp_fd = fopen(import_name, "r");
    if (bmp_fd < 0){
        printf("Failed to open file for reading\n");
        return -1;
    }
    char input[128];
    fgets(input, 128, bmp_fd);
    char type[5];
    int height, width, bits;
    sscanf(input, "%s  %d %d %d", type, &width, &height, &bits);
    int red, green, blue;
    int x, y;
    for (y=0; y < height; y++){
        for (x=0; x < width; x++){
            fgets(input, 128, bmp_fd);
            sscanf(input, "%d %d %d", &red, &green, &blue);
            pixels[y*width + x] = (0x00 << 24) | (red << 16) | (green << 8) | blue;
            tp_pixels[y + height*x] = (0x00 << 24) | (red << 16) | (green << 8) | blue;
        }
    }
    fclose(bmp_fd);
    return 0;
}

int get_bmp_height(char* import_name) {
    FILE *bmp_fd;
    bmp_fd = fopen(import_name, "r");
    if (bmp_fd < 0){
        printf("Failed to open file for reading\n");
        return -1;
    }
    char input[128];
    fgets(input, 128, bmp_fd);
    char type[5];
    int height, width, bits;
    sscanf(input, "%s  %d %d %d", type, &width, &height, &bits);
    fclose(bmp_fd);
    return height;
}

int get_bmp_width(char* import_name) {
    FILE *bmp_fd;
    bmp_fd = fopen(import_name, "r");
    if (bmp_fd < 0){
        printf("Failed to open file for reading\n");
        return -1;
    }
    char input[128];
    fgets(input, 128, bmp_fd);
    char type[5];
    int height, width, bits;
    sscanf(input, "%s  %d %d %d", type, &width, &height, &bits);
    fclose(bmp_fd);
    return width;
}

int print_bmp(char* export_name, uint32_t* pixels, int height, int width) {
    FILE *bmp_fd;
    bmp_fd = fopen(export_name, "w");
    if (bmp_fd < 0){
        printf("Failed to open file for writing\n");
        return -1;
    }
    unsigned char red, green, blue;
    uint32_t pixel;
    int x,y;
    fprintf(bmp_fd, "P3 %d %d 255\n", width, height);
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            pixel = (uint32_t)pixels[y*width + x];
            red = (unsigned char)(( pixel & RED_MASK) >> 16); /* Red */
            green = (unsigned char)(( pixel & GREEN_MASK) >> 8);  /* Green */
            blue = (unsigned char)(( pixel & BLUE_MASK));       /* Blue */
            fprintf(bmp_fd, "%u %u %u\n", red, green, blue);
        }
    }
    fclose(bmp_fd);
    return 0;
}

int main(int argc, char **argv)
{
    Timer timer;
    cudaError_t cuda_ret;
    int x,y;
    int opt, i,j,k;
    char image_name[100] = "";
    char output_name[100] = "";
    int cpu_only = 0;

    /* pull in any arguments */
    while ((opt = getopt(argc, argv, "i:o:mh")) != -1) {
        switch (opt) {
            case 'i':
                strncpy(image_name, optarg, 100);
                break;

            case 'o':
                strncpy(output_name, optarg, 100);
                break;

            case 'm':
                cpu_only = 1;
                break;

            case 'h':
                print_usage(argv[0]);
                return 0;

            default:
                printf("Invalid option '-%c'\n", opt);
                print_usage(argv[0]);
                return 1;
        }
    }

    printf("\nSet up..."); fflush(stdout);
    startTime(&timer);

    uint32_t *pixels;
    uint32_t *tp_pixels;
    int height = get_bmp_height(image_name);
    int width = get_bmp_width(image_name);
    // printf("\n Height %d, width %d\n", height, width);

    pixels = (uint32_t *)malloc(sizeof(uint32_t)*height*width);
    // transposed pixels...
    tp_pixels = (uint32_t *)malloc(sizeof(uint32_t)*height*width);
    import_bmp(image_name, pixels, tp_pixels);

    /* Need this for the part where we go through the bits */
    pixel_data_s *pixel_data;
    pixel_data = (pixel_data_s* )malloc(sizeof(pixel_data_s)*height*width);
    if (pixel_data == NULL) {
        printf("Error allocating space for parsing!\n");
        // FreeImage_DeInitialise();
        return -1;
    }
    // initialize pixel data
    for (i=0; i < height*width; i++) {
        pixel_data[i].bucketed = 0;
        pixel_data[i].color_type = bucket_type_default;
        pixel_data[i].bucket_row = -1;
        pixel_data[i].bucket_col = -1;
        pixel_data[i].last_x = -1;
    }

    // initialize vote data


    // initialize buckets
    bucket_mat_s bucket_matrix;
    bucket_row_s this_row;
    bucket_data_s this_box;
    bucket_matrix.col_len = 0;
    for (i=0; i < NUM_SAVE_ROWS; i++){
        this_row = bucket_matrix.bucket_row[i];
        this_row.max_height = -1;
        this_row.min_height = height;
        this_row.row_top = -1;
        this_row.max_row_top = -1;
        this_row.valid = -1;
        this_row.col_count = 0;
        for (j=0; j < NUM_SAVE_COLS; j++) {
            this_box = this_row.box[j];
            this_box.bucket_color = bucket_type_default;
            this_box.start_x=-1;
            this_box.start_y=-1;
            this_box.mid_x=-1;
            this_box.mid_y=-1;
            this_box.width=-1;
            this_box.height=-1;
            this_box.red=-1;
            this_box.green=-1;
            this_box.blue=-1;
            this_box.red_percent=-1;
            this_box.green_percent=-1;
            this_box.blue_percent=-1;
            this_box.did_global_add=0;
            this_box.valid=-1;
            this_row.box[j] = this_box;
        }
        bucket_matrix.bucket_row[i] = this_row;
    }

    bucket_values_s bucket_values[bucket_type_invalid];
    for (i=0; i < bucket_type_invalid; i++) {
        // initialize target values
        // bucket_values[i];
        switch ( i ) {
        case bucket_type_dark_skin:
            bucket_values[i].red = 115;
            bucket_values[i].green = 82;
            bucket_values[i].blue = 68;
            strncpy(bucket_values[i].name, "dark_skin",20);
            break;
        case bucket_type_light_skin:
            bucket_values[i].red = 194;
            bucket_values[i].green = 150;
            bucket_values[i].blue = 130;
            strncpy(bucket_values[i].name, "light_skin",20);
            break;
        case bucket_type_blue_sky:
            bucket_values[i].red = 98;
            bucket_values[i].green = 122;
            bucket_values[i].blue = 157;
            strncpy(bucket_values[i].name, "blue_sky",20);
            break;
        case bucket_type_foliage:
            bucket_values[i].red = 87;
            bucket_values[i].green = 108;
            bucket_values[i].blue = 67;
            strncpy(bucket_values[i].name, "foliage",20);
            break;
        case bucket_type_blue_flower:
            bucket_values[i].red = 133;
            bucket_values[i].green = 128;
            bucket_values[i].blue = 177;
            strncpy(bucket_values[i].name, "blue_flower",20);
            break;
        case bucket_type_bluish_green:
            bucket_values[i].red = 103;
            bucket_values[i].green = 189;
            bucket_values[i].blue = 170;
            strncpy(bucket_values[i].name, "bluish_green",20);
            break;
        case bucket_type_orange:
            bucket_values[i].red = 214;
            bucket_values[i].green = 126;
            bucket_values[i].blue = 44;
            strncpy(bucket_values[i].name, "orange",20);
            break;
        case bucket_type_purplish_blue:
            bucket_values[i].red = 80;
            bucket_values[i].green = 91;
            bucket_values[i].blue = 166;
            strncpy(bucket_values[i].name, "purplish_blue",20);
            break;
        case bucket_type_moderate_red:
            bucket_values[i].red = 193;
            bucket_values[i].green = 90;
            bucket_values[i].blue = 99;
            strncpy(bucket_values[i].name, "moderate_red",20);
            break;
        case bucket_type_purple:
            bucket_values[i].red = 94;
            bucket_values[i].green = 60;
            bucket_values[i].blue = 108;
            strncpy(bucket_values[i].name, "purple",20);
            break;
        case bucket_type_yellow_green:
            bucket_values[i].red = 157;
            bucket_values[i].green = 188;
            bucket_values[i].blue = 64;
            strncpy(bucket_values[i].name, "yellow_green",20);
            break;
        case bucket_type_orange_yellow:
            bucket_values[i].red = 224;
            bucket_values[i].green = 163;
            bucket_values[i].blue = 46;
            strncpy(bucket_values[i].name, "orange_yellow",20);
            break;
        case bucket_type_blue:
            bucket_values[i].red = 56;
            bucket_values[i].green = 61;
            bucket_values[i].blue = 150;
            strncpy(bucket_values[i].name, "blue",20);
            break;
        case bucket_type_green:
            bucket_values[i].red = 70;
            bucket_values[i].green = 148;
            bucket_values[i].blue = 73;
            strncpy(bucket_values[i].name, "green",20);
            break;
        case bucket_type_red:
            bucket_values[i].red = 175;
            bucket_values[i].green = 54;
            bucket_values[i].blue = 60;
            strncpy(bucket_values[i].name, "red",20);
            break;
        case bucket_type_yellow:
            bucket_values[i].red = 231;
            bucket_values[i].green = 199;
            bucket_values[i].blue = 31;
            strncpy(bucket_values[i].name, "yellow",20);
            break;
        case bucket_type_magenta:
            bucket_values[i].red = 187;
            bucket_values[i].green = 86;
            bucket_values[i].blue = 149;
            strncpy(bucket_values[i].name, "magenta",20);
            break;
        case bucket_type_cyan:
            bucket_values[i].red = 8;
            bucket_values[i].green = 133;
            bucket_values[i].blue = 161;
            strncpy(bucket_values[i].name, "cyan",20);
            break;
        case bucket_type_white:
            bucket_values[i].red = 243;
            bucket_values[i].green = 243;
            bucket_values[i].blue = 243;
            strncpy(bucket_values[i].name, "white",20);
            break;
        case bucket_type_neutral_8:
            bucket_values[i].red = 200;
            bucket_values[i].green = 200;
            bucket_values[i].blue = 200;
            strncpy(bucket_values[i].name, "neutral_8",20);
            break;
        case bucket_type_neutral_6_5:
            bucket_values[i].red = 160;
            bucket_values[i].green = 160;
            bucket_values[i].blue = 160;
            strncpy(bucket_values[i].name, "neutral_6_5",20);
            break;
        case bucket_type_neutral_5:
            bucket_values[i].red = 122;
            bucket_values[i].green = 122;
            bucket_values[i].blue = 122;
            strncpy(bucket_values[i].name, "neutral_5",20);
            break;
        case bucket_type_neutral_3_5:
            bucket_values[i].red = 85;
            bucket_values[i].green = 85;
            bucket_values[i].blue = 82;
            strncpy(bucket_values[i].name, "neutral_3_5",20);
            break;
        case bucket_type_black:
            bucket_values[i].red = 52;
            bucket_values[i].green = 52;
            bucket_values[i].blue = 52;
            strncpy(bucket_values[i].name, "black",20);
            break;
        default:
            bucket_values[i].red = 0;
            bucket_values[i].green = 0;
            bucket_values[i].blue = 0;
            strncpy(bucket_values[i].name, "default",20);
            break;
        }
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    int check_y;
    uint32_t pixel, check_pixel;

    // largest dimensions of a valid square
    int max_sq_width = width/6;
    // int max_sq_height = height/4;
    // smallest dimensions of a valid square
    int min_sq_width = width/14;
    int min_sq_height = height/14;

    int min_sq_sep = min_sq_height/4;

    int cur_bucket_row = 0;
    int cur_bucket_col = 0;
    int max_row_len = 0;


    ////
    uint32_t* pixels_d;
    uint32_t* pixels_ycbcr_d;
    uint32_t* tp_pixels_d;
    uint32_t* tp_pixels_ycbcr_d;
    int* box_colors_d;
    int* vote_space_d;
    int vote_width = ceil(width/TILE_SIZE);
    int vote_height = ceil(height/TILE_SIZE);
    ////
    int* vote_space_s;
    vote_space_s = (int *)malloc(sizeof(int)*vote_height*vote_width);

    uint32_t* pixels_ycbcr;
    pixels_ycbcr = (uint32_t *)malloc(sizeof(uint32_t)*height*width);

    int* box_colors_h;
    box_colors_h = (int *)malloc(sizeof(int)*NUM_REAL_ROWS*NUM_REAL_COLS*3); //stuffing red,green,blue across

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    if (cpu_only) {

        printf("\nConverting memory on CPU..."); fflush(stdout);
        startTime(&timer);
        for(y=0; y < height; y++) {
            for (x=0; x < width; x++){
                pixels_ycbcr[y*width + x] = rgb_to_ycbcr(pixels[y*width + x]);
            }
        }

        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    else {


        // Copy resources to GPU
        printf("\nCopying memory to GPU..."); fflush(stdout);
        startTime(&timer);
        cuda_ret = cudaMalloc((void**)&pixels_d, sizeof(uint32_t)*width*height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to allocate device memory \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMalloc((void**)&tp_pixels_d, sizeof(uint32_t)*width*height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to allocate device memory \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMalloc((void**)&pixels_ycbcr_d, sizeof(uint32_t)*width*height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to allocate device memory \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMalloc((void**)&tp_pixels_ycbcr_d, sizeof(uint32_t)*width*height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to allocate device memory \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMalloc((void**)&vote_space_d, sizeof(int)*vote_width*vote_height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to allocate device memory \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMalloc((void**)&box_colors_d, sizeof(int)*NUM_REAL_ROWS*NUM_REAL_COLS*3);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to allocate device memory \n", __FILE__, __LINE__);
            exit(-1);
        }


        cuda_ret = cudaMemcpy(pixels_d, pixels, sizeof(uint32_t)*width*height, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to copy memory to the device \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMemcpy(tp_pixels_d, tp_pixels, sizeof(uint32_t)*width*height, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to copy memory to the device \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMemset(pixels_ycbcr_d, 0, sizeof(uint32_t)*width*height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to set memory on the device \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMemset(tp_pixels_ycbcr_d, 0, sizeof(uint32_t)*width*height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to set memory on the device \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMemset(vote_space_d, 0, sizeof(int)*vote_width*vote_height);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to set memory on the device \n", __FILE__, __LINE__);
            exit(-1);
        }
        cuda_ret = cudaMemset(box_colors_d, 0, sizeof(int)*NUM_REAL_ROWS*NUM_REAL_COLS*3);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to set memory on the device \n", __FILE__, __LINE__);
            exit(-1);
        }

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));


        printf("\nRunning GPU conversion method..."); fflush(stdout);
        startTime(&timer);
        // Launch GPU method
        launch_gpu_rgb_to_ycbcr(width, height, pixels_d, pixels_ycbcr_d, 0);
        launch_gpu_rgb_to_ycbcr(width, height, tp_pixels_d, tp_pixels_ycbcr_d, 1);
        cudaDeviceSynchronize();

        // cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }





    if (cpu_only) {
        printf("\nFinding boxes on CPU..."); fflush(stdout);
        startTime(&timer);
        bucket_matrix.col_len = 1;
        for (y = 0; y < height; y++) {
            int square_recent = 0;
            int last_x = 0;
            for (x = 0; x < width; ) {// x++) {
                int next_x = x + 1;
                int eager = 0;
                if (pixel_data[width*y+x].bucketed == 0) {
                    // pixel = (uint32_t)pixels[y*width + x];
                    pixel = (uint32_t)pixels_ycbcr[y*width + x];

                    int not_done = 1;
                    int next_y = y + 1;
                    int horizontal = 0;
                    int vertical = 0;
                    int stops = 0;
                    int len_line = bucket_matrix.bucket_row[cur_bucket_row].col_count;
                    int luma_delta_side = 11;
                    int chroma_delta_side = 15;
                    int luma_delta_center = 7;
                    int chroma_delta_center = 15;
                    if (square_recent && (x > (min_sq_sep + last_x)) && (x < (min_sq_width + last_x)) && (len_line < 6) ) {
                        // eager for new box
                        luma_delta_side = 20;
                        chroma_delta_side = 30;
                        luma_delta_center = 10;
                        chroma_delta_center = 15;
                        eager = 1;
                    }


                    while(not_done) {
                        // move along toward the right
                        if ((next_x < width) && (pixel_data[width*y+next_x].bucketed == 0) ) {
                            // check_pixel = (uint32_t)pixels[y*width + next_x];
                            check_pixel = (uint32_t)pixels_ycbcr[y*width + next_x];
                            // if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
                            if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
                                horizontal++;
                                next_x++;
                                stops = 0;
                            }
                            else {
                                stops +=1;
                                horizontal++;
                                next_x++;
                            }
                            if (stops > MAX_STOPS) {
                                not_done = 0;
                                break;
                            }
                        }
                        else {
                            not_done = 0;
                            break;
                        }
                    }
                    horizontal -= stops;
                    next_x -= stops;
                    if ((horizontal > min_sq_width) && (horizontal < max_sq_width)) {
                        // horizontal (top) is good enough...
                        // check vertical (left)
                        check_y = y;
                        not_done = 1;
                        stops = 0;
                        while (not_done) {
                            if ((next_y < height) && (pixel_data[width*next_y+x].bucketed == 0)) {
                                check_y++;
                                // check_pixel = (uint32_t)pixels[check_y*width + x];
                                check_pixel = (uint32_t)pixels_ycbcr[check_y*width + x];
                                // if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
                                if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
                                    vertical++;
                                    next_y++;
                                    stops = 0;
                                }
                                else {
                                    stops += 1;
                                    vertical++;
                                    next_y++;
                                }
                                if (stops > MAX_STOPS) {
                                    not_done = 0;
                                    break;
                                }
                            }
                            else {
                                not_done = 0;
                                break;
                            }
                        }
                        vertical -= stops;
                        next_y -= stops;
                        if ((vertical > min_sq_width) && (vertical < max_sq_width)) {
                            //vertical (left) is good enough...
                            // take the shorter one, and check to make sure the
                            // rest is valid
                            int use_length;
                            if (vertical >= horizontal) {
                                // use the horizontal size
                                use_length = horizontal - horizontal/10;
                            }
                            else {
                                // horizontal > vertical
                                use_length = vertical - vertical/10;
                            }
                            int valid = 1;
                            // check and see if right is valid
                            check_y = y;
                            stops = 0;
                            for (i=0; i < use_length; i++) {
                                // check_pixel = (uint32_t)pixels[check_y*width + x+use_length];
                                check_pixel = (uint32_t)pixels_ycbcr[check_y*width + x+use_length];
                                // if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)) {
                                if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side, chroma_delta_side)) {
                                    stops = 0;
                                }
                                else {
                                    stops += 1;
                                }
                                check_y++;
                                if (stops > MAX_STOPS) {
                                    valid = 0;
                                    break;
                                }
                            }
                            check_y -= stops;
                            if (valid) {
                                // check and see if bottom is valid
                                stops = 0;
                                for (i=0; i < use_length; i++) {
                                    // check_pixel = (uint32_t)pixels[check_y*width + x+i];
                                    check_pixel = (uint32_t)pixels_ycbcr[check_y*width + x+i];
                                    // if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)) {
                                    if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side, chroma_delta_side)) {
                                        stops = 0;
                                    }
                                    else {
                                        stops += 1;
                                    }
                                    if (stops > MAX_STOPS) {
                                        valid = 0;
                                        break;
                                    }
                                }
                            }
                            if (valid) {
                                // check the middle parts
                                check_y = y+1;
                                int votes_for = 0;
                                int votes_against = 0;
                                for (i=1; i < (use_length -2); i++) {
                                    for (j=1; j < (use_length - 2); j++) {
                                        // check_pixel = (uint32_t)pixels[check_y*width + x+j];
                                        check_pixel = (uint32_t)pixels_ycbcr[check_y*width + x+j];
                                        // if (pixel_similar(pixel, check_pixel, luma_delta_center, chroma_delta_center)) {
                                        if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_center, chroma_delta_center)) {
                                            votes_for +=1;
                                        }
                                        else {
                                            votes_against +=1;
                                        }
                                    }
                                    check_y++;
                                }
                                if (votes_against*8 > votes_for) {
                                    valid = 0;
                                }
                                if (valid) {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                    // bucket and claim the box as a potential
                                    square_recent = 1;
                                    last_x = x + use_length;
                                    this_row = bucket_matrix.bucket_row[cur_bucket_row];
                                    if (this_row.row_top < 0) {
                                        // initialize row top
                                        bucket_matrix.bucket_row[cur_bucket_row].row_top = y;
                                    }
                                    else if (this_row.row_top + this_row.max_height < y) {
                                        // new row
                                        cur_bucket_row++;
                                        cur_bucket_col = 0;
                                        bucket_matrix.col_len++;
                                        bucket_matrix.bucket_row[cur_bucket_row].row_top = y;
                                    }
                                    if (cur_bucket_col > max_row_len) {
                                        max_row_len = cur_bucket_col;
                                    }
                                    // reload reference
                                    this_row = bucket_matrix.bucket_row[cur_bucket_row];
                                    if (use_length > this_row.max_height) {
                                        this_row.max_height = use_length;
                                    }
                                    if (this_row.max_row_top < y) {
                                        this_row.max_row_top = y;
                                    }
                                    if (this_row.min_height > use_length) {
                                        this_row.min_height = use_length;
                                    }

                                    // increment row count
                                    this_row.col_count++;
                                    this_row.start_y_count++;
                                    this_row.start_y_total += y;
                                    bucket_matrix.bucket_col[cur_bucket_col].start_x_count++;
                                    bucket_matrix.bucket_col[cur_bucket_col].start_x_total += x;

                                    this_box = this_row.box[cur_bucket_col];
                                    this_box.start_x = x;
                                    this_box.start_y = y;
                                    this_box.mid_x = x + use_length/2;
                                    this_box.mid_y = y + use_length/2;
                                    this_box.width = use_length;
                                    this_box.height = use_length;
                                    // mark rows starting with this row
                                    check_y = y;
                                    uint32_t red = 0;
                                    uint32_t green = 0;
                                    uint32_t blue = 0;
                                    for (i=0; i < use_length; i++) {
                                        // i height
                                        for (j=0; j < use_length; j++) {
                                            // j length
                                            // check_pixel = (uint32_t)check_row[x+j];
                                            check_pixel = (uint32_t)pixels[check_y*width + x+j];
                                            red += (check_pixel & RED_MASK) >> 16;
                                            green += (check_pixel & GREEN_MASK) >> 8;
                                            blue += (check_pixel & BLUE_MASK);
                                            if ( (i==0) || (i == use_length-1) || (j==0) || (j==use_length-1)) {
                                                // draw a cyan box around the target
                                                // check_row[x+j] = 0xFF00FFFF;
                                                pixels[check_y*width + x+j] = 0xFF00FFFF;
                                            }
                                            pixel_data[width*(y+i) + (x+j)].bucketed = 1;
                                            pixel_data[width*(y+i) + (x+j)].bucket_row = cur_bucket_row;
                                            pixel_data[width*(y+i) + (x+j)].bucket_col = cur_bucket_col;
                                            pixel_data[width*(y+i) + (x+j)].last_x = x + use_length;
                                        }
                                        check_y++;
                                    }
                                    int use_length_sqr = use_length*use_length;
                                    this_box.red = red/use_length_sqr;
                                    this_box.green = green/use_length_sqr;
                                    this_box.blue = blue/use_length_sqr;

                                    // save changes
                                    this_row.box[cur_bucket_col] = this_box;
                                    bucket_matrix.bucket_row[cur_bucket_row] = this_row;
                                    // increment which column value
                                    cur_bucket_col++;
                                }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            }
                        }
                    }
                }
                // load previous line square reference
                else if (pixel_data[width*y+x].last_x != -1) {
                    square_recent = 1;
                    last_x = pixel_data[width*y+x].last_x;
                }
                if (eager) {
                    x = x+1;
                }
                else {
                    x = next_x;
                }
            }
        }
        // End CPU calculation
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    else {
        printf("\nRunning GPU boxing methods...\n"); fflush(stdout);

        printf("\nScanning Vertical..."); fflush(stdout);
        startTime(&timer);
        // Launch GPU method
        launch_gpu_scanline(width, height, pixels_d, pixels_ycbcr_d);
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        printf("\nScanning horizontal..."); fflush(stdout);
        startTime(&timer);
        launch_tp_gpu_scanline(width, height, pixels_d, tp_pixels_ycbcr_d);
        // launch_non_tp_gpu_scanline(width, height, pixels_d, pixels_ycbcr_d);
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        printf("\nVoting for tiles..."); fflush(stdout);
        startTime(&timer);
        launch_gpu_small_tile_vote(width, height, pixels_d, vote_space_d, vote_width, vote_height);
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        printf("\nVote smoothing..."); fflush(stdout);
        startTime(&timer);
        launch_gpu_tiled_vote_smoothing(width, height, pixels_d, vote_space_d, vote_width, vote_height);
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Copy resources back to device

        printf("\n Copying votes back to host..."); fflush(stdout);
        startTime(&timer);
        cuda_ret = cudaMemcpy(vote_space_s, vote_space_d, sizeof(int)*vote_width*vote_height, cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to copy memory from the device \n", __FILE__, __LINE__);
            exit(-1);
        }
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        printf("\n Determining boxes with CPU..."); fflush(stdout);
        startTime(&timer);

        for(y=0; y < vote_height; y++){
            for(x=0; x < vote_width;){// x++){
                int vote;
                int horizontal = 0;
                int vertical = 0;
                int box_width = 0;
                int box_height = 0;
                int box_x = x*TILE_SIZE;
                int box_y = y*TILE_SIZE;
                int next_x = x + 1;
                int check_y = y + 1;
                vote = (int)vote_space_s[y*vote_width + x];
                if (vote > 0) {
                    horizontal++;
                    vertical++;
                    // // mark
                    // vote_space_s[y*vote_width + x] = -1;
                    // work across
                    while(vote > 0) {
                        if (next_x < vote_width) {
                            vote = (int)vote_space_s[y*vote_width + next_x];
                            if (vote > 0) {
                                horizontal++;
                                next_x++;
                            }
                        }
                        else {
                            vote = 0;
                            break;
                        }
                    }
                    box_width = horizontal*TILE_SIZE;
                    if ((min_sq_width < box_width) && (box_width < max_sq_width)) {
                        // work downwards
                        vote = 1;
                        while(vote > 0) {
                            if (check_y < vote_height) {
                                vote = (int)vote_space_s[check_y*vote_width + x];
                                if (vote > 0){
                                    vertical++;
                                    check_y++;
                                }
                            }
                            else {
                                vote = 0;
                                break;
                            }
                        }
                        box_height = vertical*TILE_SIZE;
                        if ((min_sq_width < box_height) && (box_height < max_sq_width)){
                            // claim and bucket
                            this_row = bucket_matrix.bucket_row[cur_bucket_row];
                            // increment row count
                            this_row.col_count++;
                            this_box = this_row.box[cur_bucket_col];
                            this_box.start_x = box_x;
                            this_box.start_y = box_y;
                            this_box.width = box_width;
                            this_box.height = box_height;
                            int box_color_row = cur_bucket_row;
                            int box_color_col = cur_bucket_col;
                            // mark up the vote space
                            for (i=y; i < (y+vertical); i++){
                                for (j=x; j < (x+horizontal); j++){
                                    vote_space_s[i*vote_width + j] = -1;
                                }
                            }
                            launch_get_color_boxes(width, height, box_y, box_height, box_x,
                                                   box_width, box_colors_d, pixels_d, box_color_row,
                                                   box_color_col);
                            // cudaDeviceSynchronize();
                            // save changes
                            this_row.box[cur_bucket_col] = this_box;
                            bucket_matrix.bucket_row[cur_bucket_row] = this_row;
                            if (cur_bucket_col == 5) {
                                // new row
                                cur_bucket_col = 0;
                                cur_bucket_row++;
                                bucket_matrix.col_len++;
                            }
                            else {
                                cur_bucket_col++;
                            }
                            if (cur_bucket_col > max_row_len) {
                                max_row_len = cur_bucket_col;
                            }
                        }
                    }
                }
                x = next_x;
            }
        }
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        printf("\n Loading GPU color values into host..."); fflush(stdout);
        startTime(&timer);
        cuda_ret = cudaMemcpy(box_colors_h, box_colors_d, sizeof(int)*6*4*3, cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to copy memory to the device \n", __FILE__, __LINE__);
            exit(-1);
        }
        cudaDeviceSynchronize();

        // retrieve colors
        for (i=0; i < 4; i++){
            this_row = bucket_matrix.bucket_row[i];
            for (j=0; j < 6; j++) {
                this_box = this_row.box[j];
                int box_area = this_box.width*this_box.height;
                int box_idx = (i*6 + j)*3;
                this_box.red = (float)box_colors_h[box_idx]/box_area;
                this_box.green = (float)box_colors_h[box_idx+1]/box_area;
                this_box.blue = (float)box_colors_h[box_idx+2]/box_area;
                // save changes
                this_row.box[j] = this_box;
            }
            // save changes
            bucket_matrix.bucket_row[i] = this_row;
        }
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }


    ///////////////////////////////////////////////////////////////////////////////

    printf ("\n Row len (columns) %d, Column len (rows) %d\n", max_row_len+1, bucket_matrix.col_len);

    printf("\nSorting/placing boxes..."); fflush(stdout);
    startTime(&timer);

    int box_count=0;
    int box_size=0;
    int left_most=width;
    int right_most=0;
    int top_most=height;
    int bottom_most=0;

    for (i=0; i < bucket_matrix.col_len; i++) {
        this_row = bucket_matrix.bucket_row[i];
        for (j=0; j < (this_row.col_count); j++) {
            this_box = this_row.box[j];
            box_size += this_box.width;
            box_count++;
            if (this_box.start_x > right_most) {
                right_most = this_box.start_x;
            }
            else if (this_box.start_x < left_most) {
                left_most = this_box.start_x;
            }
            if (this_box.start_y > bottom_most) {
                bottom_most = this_box.start_y;
            }
            else if (this_box.start_y < top_most) {
                top_most = this_box.start_y;
            }
        }
    }

    // int avg_width = box_size/box_count;
    // distance between left point of leftmost box and left point of rightmost
    // is 5 boxes
    int big_width = (right_most - left_most)/5;

    int bucket_enum;
    for (i=0; i < bucket_matrix.col_len; i++) {
        this_row = bucket_matrix.bucket_row[i];
        bucket_enum = 1 + 6*i;
        if (this_row.col_count == 6) {
            // just assign the next 6 buckets
            for (j=0; j < 6; j++) {
                int current_smallest = width;
                int next_idx = 0;
                for (k=0; k < (this_row.col_count); k++) {
                    this_box = this_row.box[k];
                    if (this_box.bucket_color == bucket_type_default) {
                        if (this_box.start_x < current_smallest) {
                            current_smallest = this_box.start_x;
                            next_idx = k;
                        }
                    }
                }
                this_row.box[next_idx].bucket_color = (bucket_type)bucket_enum;
                bucket_enum++;
            }
        }
        else {
            // determine the missed buckets
            for (j=0; j < 6; j++) {
                int bucket_found = 0;
                for (k=0; k < (this_row.col_count); k++) {
                    if (bucket_found == 0) {
                        this_box = this_row.box[k];
                        if (this_box.bucket_color == bucket_type_default) {
                            int prob_start = left_most + j*big_width - big_width/3;
                            int prob_end = prob_start+big_width - big_width/3;
                            if ((this_box.start_x > prob_start) && (this_box.start_x < prob_end)){
                                this_row.box[k].bucket_color = (bucket_type)(bucket_enum+j);
                                // move on to next box;
                                bucket_found = 1;
                            }
                        }
                    }
                }
            }
        }
        // save the row changes
        bucket_matrix.bucket_row[i] = this_row;
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));



    global_needs_s global_needs_h;

    global_needs_h.have_red = 0;
    global_needs_h.need_red = 0;
    global_needs_h.have_green = 0;
    global_needs_h.need_green = 0;
    global_needs_h.have_blue = 0;
    global_needs_h.need_blue = 0;
    global_needs_h.red_percent = 0;
    global_needs_h.green_percent = 0;
    global_needs_h.blue_percent = 0;

    if(cpu_only) {

        printf("\nCalculating correctness and printing colors on CPU..."); fflush(stdout);
        startTime(&timer);
        // Determine correctness
        for (i=bucket_type_dark_skin; i < bucket_type_invalid; i++){
            for (j=0; j < bucket_matrix.col_len; j++) {
                this_row = bucket_matrix.bucket_row[j];
                for (k=0; k < (this_row.col_count); k++) {
                    this_box = this_row.box[k];
                    int color_idx = (int)this_box.bucket_color;
                    if (color_idx == i) {
                        global_needs_h.have_red += this_box.red;
                        global_needs_h.have_green += this_box.green;
                        global_needs_h.have_blue += this_box.blue;
                        global_needs_h.need_red += bucket_values[color_idx].red;
                        global_needs_h.need_green += bucket_values[color_idx].green;
                        global_needs_h.need_blue += bucket_values[color_idx].blue;
                        // printf("RGB %d %d %d looks like %.1f %.1f %.1f for %s\n", bucket_values[color_idx].red, bucket_values[color_idx].green,
                            // bucket_values[color_idx].blue, this_box.red, this_box.green, this_box.blue, bucket_values[color_idx].name);
                        float box_red_percent = this_box.red/bucket_values[color_idx].red*100;
                        float box_green_percent = this_box.green/bucket_values[color_idx].green*100;
                        float box_blue_percent = this_box.blue/bucket_values[color_idx].blue*100;
                        // printf("    RGB percent %.1f %.1f %.1f\n", box_red_percent, box_green_percent, box_blue_percent);
                        bucket_matrix.bucket_row[j].box[k].red_percent = box_red_percent;
                        bucket_matrix.bucket_row[j].box[k].green_percent = box_green_percent;
                        bucket_matrix.bucket_row[j].box[k].blue_percent = box_blue_percent;
                        // draw the box as it is supposed to look
                        for (y=this_box.start_y; y < (this_box.start_y + this_box.height); y++){
                            for (x=this_box.start_x; x < (this_box.start_x + this_box.width); x++) {
                                unsigned char r_val = bucket_values[color_idx].red;
                                unsigned char g_val = bucket_values[color_idx].green;
                                unsigned char b_val = bucket_values[color_idx].blue;
                                pixels[y*width + x] = 0xFF << 24 | r_val << 16 | g_val << 8 | b_val;
                            }
                        }
                    }
                }
            }
        }

        global_needs_h.red_percent = global_needs_h.have_red/global_needs_h.need_red*100;
        global_needs_h.green_percent = global_needs_h.have_green/global_needs_h.need_green*100;
        global_needs_h.blue_percent = global_needs_h.have_blue/global_needs_h.need_blue*100;

        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }
    else {
        //Launch kernelized version
        printf("\nLaunching correctness calculations on GPU..."); fflush(stdout);
        startTime(&timer);

        for (i=bucket_type_dark_skin; i < bucket_type_invalid; i++){
            for (j=0; j < bucket_matrix.col_len; j++) {
                this_row = bucket_matrix.bucket_row[j];
                for (k=0; k < (this_row.col_count); k++) {
                    this_box = this_row.box[k];
                    int color_idx = (int)this_box.bucket_color;
                    if (color_idx == i) {
                        global_needs_h.have_red += this_box.red;
                        global_needs_h.have_green += this_box.green;
                        global_needs_h.have_blue += this_box.blue;
                        global_needs_h.need_red += bucket_values[color_idx].red;
                        global_needs_h.need_green += bucket_values[color_idx].green;
                        global_needs_h.need_blue += bucket_values[color_idx].blue;
                        float box_red_percent = this_box.red/bucket_values[color_idx].red*100;
                        float box_green_percent = this_box.green/bucket_values[color_idx].green*100;
                        float box_blue_percent = this_box.blue/bucket_values[color_idx].blue*100;
                        bucket_matrix.bucket_row[j].box[k].red_percent = box_red_percent;
                        bucket_matrix.bucket_row[j].box[k].green_percent = box_green_percent;
                        bucket_matrix.bucket_row[j].box[k].blue_percent = box_blue_percent;
                        // draw the box as it is supposed to look
                        uint32_t color_val = bucket_values[color_idx].red << 16 | bucket_values[color_idx].green << 8 | \
                                             bucket_values[color_idx].blue;
                        // launch_color_boxes(width, height, this_box.start_y, this_box.height, this_box.start_x,
                                           // this_box.width, color_val, pixels_d);
                        launch_small_color_boxes(width, height, this_box.start_y, this_box.height, this_box.start_x,
                                           this_box.width, color_val, pixels_d);
                    }
                }
            }
        }

        global_needs_h.red_percent = global_needs_h.have_red/global_needs_h.need_red*100;
        global_needs_h.green_percent = global_needs_h.have_green/global_needs_h.need_green*100;
        global_needs_h.blue_percent = global_needs_h.have_blue/global_needs_h.need_blue*100;



        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Copy from Device
        printf("\nCopying data from device to host..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMemcpy(pixels, pixels_d, sizeof(uint32_t)*width*height, cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) {
            fprintf(stderr, "[%s:%d] Unable to copy memory to the device \n", __FILE__, __LINE__);
            exit(-1);
        }

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }


    printf("\n");

    for (i=bucket_type_dark_skin; i < bucket_type_invalid; i++){
        for (j=0; j < bucket_matrix.col_len; j++) {
            this_row = bucket_matrix.bucket_row[j];
            for (k=0; k < (this_row.col_count); k++) {
                this_box = this_row.box[k];
                int color_idx = (int)this_box.bucket_color;
                if (color_idx == i) {
                    printf("RGB %d %d %d looks like %.1f %.1f %.1f for %s\n", bucket_values[color_idx].red, bucket_values[color_idx].green,
                        bucket_values[color_idx].blue, this_box.red, this_box.green, this_box.blue, bucket_values[color_idx].name);
                    printf("    RGB percent %.1f %.1f %.1f\n", this_box.red_percent, this_box.green_percent, this_box.blue_percent);
                }
            }
        }
    }



    printf("\nOverall:\n");
    printf("RED %.2f %%\n", global_needs_h.red_percent);
    printf("GREEN %.2f %%\n", global_needs_h.green_percent);
    printf("BLUE %.2f %%\n", global_needs_h.blue_percent);



    ////////////////////

    printf("\nExporting image..."); fflush(stdout);
    startTime(&timer);
    // print the image
    print_bmp(output_name, pixels, height, width);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    if (!cpu_only) {
        cudaFree(pixels_d);
        cudaFree(tp_pixels_d);
        cudaFree(pixels_ycbcr_d);
        cudaFree(tp_pixels_ycbcr_d);
        cudaFree(vote_space_d);
        cudaFree(box_colors_d);
    }

    // other frees
    free(pixels);
    free(tp_pixels);
    free(pixels_ycbcr);
    free(pixel_data);
    free(box_colors_h);

    return 0;
}
