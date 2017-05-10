#include "image-analysis.h"

#define TILE_SIZE 16
#define HORIZONTAL_TILE_SIZE 30
#define VERTICAL_TILE_SIZE 24
// #define HORIZONTAL_TILE_SIZE 64
// #define VERTICAL_TILE_SIZE 64


__global__ void results_kernel(int width, int height, int box_y, int box_height, int box_x, int box_width,
                               uint32_t color, uint32_t* pixels_d) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if ((i < width) && (j < height)) {
        // for every pixel...
        if ((box_x <=  i) && (i < box_x + box_width) && (box_y <= j) && (j < (box_y + box_height))) {
            pixels_d[j*width+i] = color;
        }
    }
}

__global__ void results_small_box_kernel(int width, int height, int box_y, int box_height, int box_x, int box_width,
                                         uint32_t color, uint32_t* pixels_d) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int x = box_x + i;
    int y = box_y + j;

    if ((x < width) && (y < height)) {
        // for every pixel...
        if ((box_x <=  x) && (x < (box_x + box_width)) && (box_y <= y) && (y < (box_y + box_height))) {
            pixels_d[y*width+x] = color;
        }
    }
}

__global__ void rgb_to_ycbcr_kernel(int width, int height, uint32_t* pixels_d, uint32_t* pixels_ycbcr_d) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if ((i < width) && (j < height)) {
        // for every pixel...
        pixels_ycbcr_d[j*width + i] = rgb_to_ycbcr(pixels_d[j*width + i]);
    }
}

__global__ void rgb_to_tp_ycbcr_kernel(int width, int height, uint32_t* tp_pixels_d, uint32_t* tp_pixels_ycbcr_d) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if ((i < width) && (j < height)) {
        // for every pixel...
        tp_pixels_ycbcr_d[j + i*height] = rgb_to_ycbcr(tp_pixels_d[j + i*height]);
    }
}

__global__ void gpu_scanline (int width, int height, uint32_t* pixels_d, uint32_t* pixels_ycbcr_d) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y;
    __device__ __shared__ uint32_t pix_y[MAX_SCAN_Y];
    for (y=threadIdx.y; y < height; y+=blockDim.y){
        if (y < MAX_SCAN_Y){
            pix_y[y] = (uint32_t)pixels_ycbcr_d[y*width + x];
        }
    }
    __syncthreads();


    if ((x < width) && (threadIdx.y == 0)) {
        int line_recent = 0;
        int last_y = 0;
        int i;
        int max_sq_width = width/6;
        int min_sq_width = width/14;
        int min_sq_height = height/14;
        int min_sq_sep = min_sq_height/4;

        uint32_t pixel = 0;

        for (y = 0; y < height; ) {// y++) {
            int next_y = y + 1;
            int eager = 0;
            if (y < MAX_SCAN_Y) {
                // load shared
                pixel = pix_y[y];
            }
            else {
                // load global
                pixel = (uint32_t)pixels_ycbcr_d[y*width + x];
            }
            uint32_t check_pixel;
            int not_done = 1;
            int vertical = 0;
            int stops = 0;
            int len_line = 0;
            int luma_delta_side = 4;
            int chroma_delta_side = 5;
            if (line_recent && (y > (min_sq_sep + last_y)) && (y < (min_sq_width + last_y)) && (len_line < 4)){
                // eager for new line
                luma_delta_side = 6;
                chroma_delta_side = 7;
                eager = 1;
            }
            while (not_done) {
                if (next_y < height) {
                    if (next_y < MAX_SCAN_Y) {
                        // load shared
                        check_pixel = pix_y[next_y];
                    }
                    else {
                        check_pixel = (uint32_t)pixels_ycbcr_d[next_y*width + x];
                    }
                    // if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
                    if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side+stops, chroma_delta_side+stops)){
                        vertical++;
                        stops = 0;
                    }
                    else {
                        stops += 1;
                        vertical++;
                    }
                    next_y++;
                    if (stops > MAX_STOPS_GPU) {
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
                len_line++;
                for (i=y; i < y+vertical; i++) {
                    // debug --> color it cyan
                    // debug --> just the blue part
                    // this is where we mark the 1st alpha bit
                    pixels_d[i*width + x] |= 0x01000000;
                }
                line_recent = 1;
                y = next_y+min_sq_sep;
            }
            else if (eager) {
                y = y+1;
            }
            else {
                y = next_y;
            }
        }
    }
}


//////
__global__ void non_tp_gpu_scanline (int width, int height, uint32_t* pixels_d, uint32_t* pixels_ycbcr_d) {
    int y = blockIdx.x*blockDim.x + threadIdx.x;
    if (y < height) {
        int line_recent = 0;
        int last_x = 0;
        int x,i;
        int max_sq_width = width/6;
        int min_sq_width = width/14;
        int min_sq_height = height/14;
        int min_sq_sep = min_sq_height/4;

        // int cur_bucket_row = 0;
        // int cur_bucket_col = 0;
        // int max_row_len = 0;

        for (x = 0; x < width; ) {// x++) {
            int next_x = x + 1;
            int eager = 0;
            // Lame non-transposed version
            uint32_t pixel = (uint32_t)pixels_ycbcr_d[y*width + x];
            uint32_t check_pixel;
            int not_done = 1;
            int horizontal = 0;
            int stops = 0;
            int len_line = 0;
            int luma_delta_side = 4;
            int chroma_delta_side = 5;
            if (line_recent && (x > (min_sq_sep + last_x)) && (x < (min_sq_width + last_x)) && (len_line < 6)){
                // eager for new line
                luma_delta_side = 6;
                chroma_delta_side = 7;
                eager = 1;
            }
            while (not_done) {
                if (next_x < width) {
                    // lame non-transposed version
                    check_pixel = (uint32_t)pixels_ycbcr_d[y*width + next_x];
                    // if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
                    if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side+stops, chroma_delta_side+stops)){
                        horizontal++;
                        stops = 0;
                    }
                    else {
                        stops += 1;
                        horizontal++;
                    }
                    next_x++;
                    if (stops > MAX_STOPS_GPU) {
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
                //horizontal (top) is good enough...
                for (i=x; i < x+horizontal; i++) {
                    // debug --> color it cyan
                    // NOT TRANSPOSED HERE
                    // debug --> just the green part
                    // this is where we mark the 2nd alpha bit
                    pixels_d[y*width + i] |= 0x02000000;
                }
                line_recent = 1;
                len_line++;
                x = next_x+min_sq_sep;
            }
            else if (eager) {
                x = x+1;
            }
            else {
                x = next_x;
            }
        }
    }
}
/////

__global__ void tp_gpu_scanline (int width, int height, uint32_t* pixels_d, uint32_t* tp_pixels_ycbcr_d) {
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x;

    __device__ __shared__ uint32_t pix_x[MAX_SCAN_X];
    for (x=threadIdx.x; x < width; x+=blockDim.x){
        if (x < MAX_SCAN_X){
            pix_x[x] = (uint32_t)tp_pixels_ycbcr_d[y + x*height];
        }
    }
    __syncthreads();


    if ((y < height) && (threadIdx.x == 0)) {
        int line_recent = 0;
        int last_x = 0;
        int x,i;
        int max_sq_width = width/6;
        int min_sq_width = width/14;
        int min_sq_height = height/14;
        int min_sq_sep = min_sq_height/4;
        uint32_t pixel;

        // int cur_bucket_row = 0;
        // int cur_bucket_col = 0;
        // int max_row_len = 0;

        for (x = 0; x < width; ) {// x++) {
            int next_x = x + 1;
            int eager = 0;
            if (x < MAX_SCAN_X) {
                // load shared
                pixel = pix_x[x];
            }
            else {
                // load global
                pixel = (uint32_t)tp_pixels_ycbcr_d[y + x*height];
            }
            uint32_t check_pixel;
            int not_done = 1;
            int horizontal = 0;
            int stops = 0;
            int len_line = 0;
            int luma_delta_side = 4;
            int chroma_delta_side = 5;
            if (line_recent && (x > (min_sq_sep + last_x)) && (x < (min_sq_width + last_x)) && (len_line < 6)){
                // eager for new line
                luma_delta_side = 6;
                chroma_delta_side = 7;
                eager = 1;
            }
            while (not_done) {
                if (next_x < width) {
                    if (next_x < MAX_SCAN_X) {
                        // load shared
                        check_pixel = pix_x[next_x];
                    }
                    else {
                        // load global
                        check_pixel = (uint32_t)tp_pixels_ycbcr_d[y + next_x*height];
                    }
                    // if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
                    if (pixel_similar_ycbcr(pixel, check_pixel, luma_delta_side+stops, chroma_delta_side+stops)){
                        horizontal++;
                        stops = 0;
                    }
                    else {
                        stops += 1;
                        horizontal++;
                    }
                    next_x++;
                    if (stops > MAX_STOPS_GPU) {
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
                //horizontal (top) is good enough...
                for (i=x; i < x+horizontal; i++) {
                    // debug --> color it cyan
                    // NOT TRANSPOSED HERE
                    // debug --> just the green part
                    // this is where we mark the 2nd alpha bit
                    pixels_d[y*width + i] |= 0x02000000;
                }
                line_recent = 1;
                len_line++;
                x = next_x+min_sq_sep;
            }
            else if (eager) {
                x = x+1;
            }
            else {
                x = next_x;
            }
        }
    }
}

__global__ void small_vote_kernel(int width, int height, uint32_t* pixels_d, int* vote_space, int vote_width,
                                  int vote_height){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    __device__ __shared__ int ds_vote_shared[TILE_SIZE][TILE_SIZE];
    __device__ __shared__ int ds_vote_column[TILE_SIZE];

    uint32_t check_pixel;

    ds_vote_shared[threadIdx.y][threadIdx.x] = 0;
    ds_vote_column[threadIdx.x] = 0;
    __syncthreads();
    if ((i < width) && (j < height)) {
        // for every pixel...
        // pixels_ycbcr_d[j*width + i] = rgb_to_ycbcr(pixels_d[j*width + i]);
        // read the pixel
        check_pixel = (uint32_t)pixels_d[j*width + i];
        if (check_pixel & 0x03000000) {
            // whole chunk
            ds_vote_shared[threadIdx.y][threadIdx.x] = 15;
        }
        else if (check_pixel & 0x01000000) {
            // half chunk
            ds_vote_shared[threadIdx.y][threadIdx.x] = 5;

        }
        else if (check_pixel & 0x02000000) {
            // half chunk
            ds_vote_shared[threadIdx.y][threadIdx.x] = 5;
        }
        else {
            // not a good sign
            ds_vote_shared[threadIdx.y][threadIdx.x] = -20;
        }

    }
    else {
        // pixels outside edges disqualify their chunk
        ds_vote_shared[threadIdx.y][threadIdx.x] = -100;
    }
    __syncthreads();
    if (threadIdx.y == 0){
        // just horizontal threads
        for(int y=0; y<TILE_SIZE; y++){
            ds_vote_column[threadIdx.x] += ds_vote_shared[y][threadIdx.x];
        }
    }
    __syncthreads();
    if ((threadIdx.x == 0) && (threadIdx.y == 0)){
        // just one thread
        int votes = 0;
        for (int x=0; x<TILE_SIZE; x++) {
            votes += ds_vote_column[x];
        }
        if (votes > TILE_SIZE*TILE_SIZE*13) {
            vote_space[blockIdx.y*vote_width + blockIdx.x] = 1;
        }
        else {
            vote_space[blockIdx.y*vote_width + blockIdx.x] = 0;
        }

    }

}

__global__ void vote_kernel_smoothing(int width, int height, uint32_t* pixels_d, int* vote_space, int vote_width,
                                      int vote_height){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int x,y;
    __device__ __shared__ int ds_vote_space[TILE_SIZE][TILE_SIZE];
    if ((i < vote_width) && (j < vote_height)) {
        ds_vote_space[threadIdx.y][threadIdx.x] = vote_space[j*vote_width + i];
    }
    else {
        ds_vote_space[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();




    int min_sq_width = width/14;

    if (( i < vote_width) && (j < vote_height) && (vote_space[j*vote_width + i])){
        // Tile was voted for previously
        int current_width = 0;
        int current_height = 0;
        int not_done = 1;
        x = i;
        y = j;
        int ty = y-blockIdx.y*blockDim.y;
        int tx = x-blockIdx.x*blockDim.x;
        int check_value;
        //debug
        int result = 0;
        while(not_done) {
            // check values to the right
            x++;
            if (x < vote_width) {

                tx = x-blockIdx.x*blockDim.x;
                if ((0 <= tx) && (tx < blockDim.x) && (0 <= ty) && (ty < blockDim.y )) {
                    // load shared
                    check_value = ds_vote_space[y-blockIdx.y*blockDim.y][threadIdx.x];
                }
                else {
                    // load global
                    check_value = vote_space[y*vote_width + x];
                }

                if (check_value){
                    current_width++;
                }
                else {
                    not_done = 1;
                    break;
                }
            }
            else {
                not_done = 1;
                break;
            }
        }
        not_done = 1;
        x = i;
        while(not_done) {
            // check values to the left
            x--;
            if (0 <= x) {

                tx = x-blockIdx.x*blockDim.x;
                if ((0 <= tx) && (tx < blockDim.x) && (0 <= ty) && (ty < blockDim.y )) {
                    // load shared
                    check_value = ds_vote_space[y-blockIdx.y*blockDim.y][threadIdx.x];
                }
                else {
                    // load global
                    check_value = vote_space[y*vote_width + x];
                }

                if (check_value){
                    current_width++;
                }
                else {
                    not_done = 1;
                    break;
                }
            }
            else {
                not_done = 1;
                break;
            }
        }
        // now have horizontal
        if (current_width*TILE_SIZE > min_sq_width) {
            // continue from the center
            x = i;
            tx = x-blockIdx.x*blockDim.x;
            not_done = 1;
            while(not_done){
                // check values below
                y--;
                if (0 <= y) {

                    ty = y-blockIdx.y*blockDim.y;
                    if ((0 <= tx) && (tx < blockDim.x) && (0 <= ty) && (ty < blockDim.y )) {
                        // load shared
                        check_value = ds_vote_space[y-blockIdx.y*blockDim.y][threadIdx.x];
                    }
                    else {
                        // load global
                        check_value = vote_space[y*vote_width + x];
                    }

                    if (check_value){
                        current_height++;
                    }
                    else {
                        not_done = 1;
                        break;
                    }
                }
                else {
                    not_done = 1;
                    break;
                }
            }
            not_done = 1;
            y = j;
            while(not_done){
                // check values above
                y++;
                if (y < vote_height) {

                    ty = y-blockIdx.y*blockDim.y;
                    if ((0 <= tx) && (tx < blockDim.x) && (0 <= ty) && (ty < blockDim.y )) {
                        // load shared
                        check_value = ds_vote_space[y-blockIdx.y*blockDim.y][threadIdx.x];
                    }
                    else {
                        // load global
                        check_value = vote_space[y*vote_width + x];
                    }

                    if (check_value){
                        current_height++;
                    }
                    else {
                        not_done = 1;
                        break;
                    }
                }
                else {
                    not_done = 1;
                    break;
                }
            }
            // now have vertical
            if (current_height*TILE_SIZE > min_sq_width) {
                result = 1;
            }
        }
        if (result == 0) {
            // deny the vote
            vote_space[j*vote_width + i]=0;
        }
    }
}

__global__ void get_color_boxes_kernel(int width, int height, int box_y, int box_height, int box_x, int box_width,
                                       int* box_colors_d, uint32_t* pixels_d, int color_row, int color_col) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int x = box_x + i;
    int y = box_y + j;
    uint32_t color;


    if ((x < width) && (y < height)) {
        // for every pixel...
        if ((box_x <=  x) && (x < (box_x + box_width)) && (box_y <= y) && (y < (box_y + box_height))) {
            color = (uint32_t)pixels_d[y*width+x];
            int red = (int)((color & RED_MASK) >> 16);
            int green = (int)((color & GREEN_MASK) >> 8);
            int blue = (int)(color & BLUE_MASK);
            int color_idx = (color_row*NUM_REAL_COLS + color_col)*3;
            atomicAdd(&box_colors_d[color_idx], red);
            atomicAdd(&box_colors_d[color_idx+1], green);
            atomicAdd(&box_colors_d[color_idx+2], blue);
        }
    }
}


/* ****************** */
/* Launcher functions */
/* ****************** */


void launch_color_boxes(int width, int height, int box_y, int box_height, int box_x, int box_width, uint32_t color,
                        uint32_t* pixels_d) {

    //Determine how many threads we need
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y, 1);
    if (width % dimBlock.x != 0) dimGrid.x++;
    if (height % dimBlock.y != 0) dimGrid.y++;
    results_kernel<<<dimGrid, dimBlock>>>(width, height, box_y, box_height, box_x, box_width, color, pixels_d);
}

void launch_small_color_boxes(int width, int height, int box_y, int box_height, int box_x, int box_width, uint32_t color,
                              uint32_t* pixels_d) {

    //Determine how many threads we need
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid(box_width/dimBlock.x, box_height/dimBlock.y, 1);
    if (box_width % dimBlock.x != 0) dimGrid.x++;
    if (box_height % dimBlock.y != 0) dimGrid.y++;
    results_small_box_kernel<<<dimGrid, dimBlock>>>(width, height, box_y, box_height, box_x, box_width, color, pixels_d);

}

void launch_gpu_scanline(int width, int height, uint32_t* pixels_d, uint32_t* pixels_ycbcr_d) {

    //Determine how many threads we need
    // dim3 dimBlock(HORIZONTAL_TILE_SIZE, 1, 1);
    // dim3 dimGrid(width/dimBlock.x, 1, 1);
    // if (width % dimBlock.x != 0) dimGrid.x++;
    dim3 dimBlock(1, VERTICAL_TILE_SIZE, 1);
    dim3 dimGrid(width, 1, 1);
    gpu_scanline<<<dimGrid, dimBlock>>>(width, height, pixels_d, pixels_ycbcr_d);
}

void launch_non_tp_gpu_scanline(int width, int height, uint32_t* pixels_d, uint32_t* pixels_ycbcr_d) {

    //Determine how many threads we need

    cudaDeviceSynchronize();
    dim3 dimBlock(VERTICAL_TILE_SIZE, 1, 1);
    dim3 dimGrid(height/dimBlock.x, 1, 1);
    if (height % dimBlock.x != 0) dimGrid.x++;
    non_tp_gpu_scanline<<<dimGrid, dimBlock>>>(width, height, pixels_d, pixels_ycbcr_d);
}

void launch_tp_gpu_scanline(int width, int height, uint32_t* pixels_d, uint32_t* tp_pixels_ycbcr_d) {

    //Determine how many threads we need

    cudaDeviceSynchronize();
    // dim3 dimBlock(VERTICAL_TILE_SIZE, 1, 1);
    // dim3 dimGrid(height/dimBlock.x, 1, 1);
    // if (height % dimBlock.x != 0) dimGrid.x++;
    dim3 dimBlock(HORIZONTAL_TILE_SIZE, 1, 1);
    dim3 dimGrid(1, height, 1);
    tp_gpu_scanline<<<dimGrid, dimBlock>>>(width, height, pixels_d, tp_pixels_ycbcr_d);
}

void launch_gpu_rgb_to_ycbcr(int width, int height, uint32_t* pixels_d, uint32_t* pixels_ycbcr_d, int tpd) {
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y, 1);
    if (width % dimBlock.x != 0) dimGrid.x++;
    if (height % dimBlock.y != 0) dimGrid.y++;
    if (tpd) {
        rgb_to_ycbcr_kernel<<<dimGrid, dimBlock>>>(width, height, pixels_d, pixels_ycbcr_d);
    }
    else {
        rgb_to_tp_ycbcr_kernel<<<dimGrid, dimBlock>>>(width, height, pixels_d, pixels_ycbcr_d);
    }
}

void launch_gpu_small_tile_vote(int width, int height, uint32_t* pixels_d, int* vote_space, int vote_width,
                                int vote_height) {
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y, 1);
    if (width % dimBlock.x != 0) dimGrid.x++;
    if (height % dimBlock.y != 0) dimGrid.y++;
    small_vote_kernel<<<dimGrid, dimBlock>>>(width, height, pixels_d, vote_space, vote_width, vote_height);
}

void launch_gpu_tiled_vote_smoothing(int width, int height, uint32_t* pixels_d, int* vote_space,
                                     int vote_width, int vote_height) {
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid(vote_width/dimBlock.x, vote_height/dimBlock.y, 1);
    if (vote_width % dimBlock.x != 0) dimGrid.x++;
    if (vote_height % dimBlock.y != 0) dimGrid.y++;
    vote_kernel_smoothing<<<dimGrid, dimBlock>>>(width, height, pixels_d, vote_space, vote_width, vote_height);
}

void launch_get_color_boxes(int width, int height, int box_y, int box_height, int box_x,
                                                   int box_width, int* box_colors_d, uint32_t* pixels_d,
                                                   int color_row,
                                                   int color_col){
    //Determine how many threads we need
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid(box_width/dimBlock.x, box_height/dimBlock.y, 1);
    if (box_width % dimBlock.x != 0) dimGrid.x++;
    if (box_height % dimBlock.y != 0) dimGrid.y++;
    get_color_boxes_kernel<<<dimGrid, dimBlock>>>(width, height, box_y, box_height, box_x, box_width, box_colors_d,
                                                  pixels_d, color_row, color_col);

}