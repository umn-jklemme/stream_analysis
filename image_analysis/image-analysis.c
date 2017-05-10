#include <gtk/gtk.h>
#include <FreeImage.h>
#include <string.h>
#include <stdlib.h>
#include "image-analysis.h"

void destroy(GtkWidget * widget, gpointer data)
{
    gtk_main_quit();
}

static void print_usage(char * appname)
{
    printf("Usage: %s [options]\n", appname);
    printf("Available options are\n");
    printf(" -i <image_file>    The image to verify and load\n");
    printf(" -h                 Print this help screen and exit\n");
}

// #define MAX_DIFF 10

int pixel_similar(uint32_t pixel_a, uint32_t pixel_b, int max_diff_luma, int max_diff_chroma)
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


    // if (abs(a_red - b_red) > 25) {
    //     return 0;
    // }
    // else if (abs(a_gre - b_gre) > 25) {
    //     return 0;
    // }
    // else if (abs(a_blu - b_blu) > 25) {
    //     return 0;
    // }

    if (abs(a_Y_val - b_Y_val) > max_diff_luma) {
        return 0;
    }
    else if (abs(a_Cb_val - b_Cb_val) > max_diff_chroma) {
        return 0;
    }
    else if (abs(a_Cr_val - b_Cr_val) > max_diff_chroma) {
        return 0;
    }

    // else {
    //     int cb_dif = abs(a_Cb_val - b_Cb_val);
    //     int cr_dif = abs(a_Cr_val - b_Cr_val);
    //     if ((cb_dif*cb_dif + cr_dif*cr_dif) > (max_diff_chroma*max_diff_chroma)){
    //         return 0;
    //     }
    // }
    return 1;
}

int main(int argc, char **argv)
{
    GtkWidget *window, *imagebox;
    GdkVisual *visual;
    GdkImage *image;
    FIBITMAP *dib;
    int x,y,z;
    int opt, i,j,k;
    char image_name[100] = "";

    /* pull in any arguments */
    while ((opt = getopt(argc, argv, "i:h")) != -1) {
        switch (opt) {
            case 'i':
                strncpy(image_name, optarg, 100);
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

    /* Initialize the FreeImage library */
    FreeImage_Initialise(TRUE);

    /* Inspect the image */
    FREE_IMAGE_FORMAT image_fif = FreeImage_GetFileType(image_name, 0);

    if (image_fif == FIF_UNKNOWN) {
        printf("Failed to find image from file\n");
        FreeImage_DeInitialise();
        return -1;
    }

    dib = FreeImage_Load(image_fif, image_name, 0);

    gtk_init(&argc, &argv);

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

    gtk_signal_connect(GTK_OBJECT(window), "destroy",
                GTK_SIGNAL_FUNC(destroy), NULL);

    visual = gdk_visual_get_system();

    int width = FreeImage_GetWidth(dib);
    int height = FreeImage_GetHeight(dib);

    image = gdk_image_new(GDK_IMAGE_NORMAL,visual, width, height);

    g_print("picture: %d bpp\n"
        "system:  %d bpp   byteorder: %d\n"
        "  redbits: %d   greenbits: %d   bluebits: %d\n"
        "image:   %d bpp   %d bytes/pixel\n",
        FreeImage_GetBPP(dib),
        visual->depth,visual->byte_order,
        visual->red_prec,visual->green_prec,visual->blue_prec,
        image->depth,image->bpp );

    if (FreeImage_GetBPP(dib) != (image->bpp << 3)) {
        FIBITMAP *ptr;

        switch (image->bpp) {
            case 1:
                ptr = FreeImage_ConvertTo8Bits(dib);
                break;

            case 2:
                if (image->depth == 15) {
                    ptr = FreeImage_ConvertTo16Bits555(dib);
                } else {
                    ptr = FreeImage_ConvertTo16Bits565(dib);
                }

                break;
            case 3:
                ptr = FreeImage_ConvertTo24Bits(dib);
                break;

            default:
            case 4:
                ptr = FreeImage_ConvertTo32Bits(dib);
                break;
        }

        FreeImage_Unload(dib);
        dib = ptr;
    }

    /* The part where we go through the bits */
    pixel_data_s *pixel_data;
    pixel_data = (pixel_data_s* )malloc(sizeof(pixel_data_s)*height*width);
    if (pixel_data == NULL) {
        printf("Error allocating space for parsing!\n");
        FreeImage_DeInitialise();
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

    // initialize buckets
    bucket_mat_s bucket_matrix;
    bucket_row_s this_row;
    bucket_data_s this_box;
    bucket_matrix.col_len = 1;
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
            this_box.valid=-1;
            this_row.box[j] = this_box;
        }
        bucket_matrix.bucket_row[i] = this_row;
    }

    bucket_values_s bucket_values[bucket_type_invalid];
    for (i=0; i < bucket_type_invalid; i++) {
        // initialize target values
        bucket_values[i];
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

    BYTE *ptr = FreeImage_GetBits(dib);
    uint32_t *row;
    uint32_t *check_row;
    unsigned char palp, pred, pgre, pblu;
    uint32_t pixel, check_pixel;

    // largest dimensions of a valid square
    int max_sq_width = width/6;
    int max_sq_height = height/4;
    // smallest dimensions of a valid square
    int min_sq_width = width/14;
    int min_sq_height = height/14;

    int min_sq_sep = min_sq_height/4;

    int cur_bucket_row = 0;
    int cur_bucket_col = 0;
    int max_row_len = 0;


    // row = (uint32_t* )malloc(sizeof(uint32_t)*height*width);
    // make a copy?
    // use a reference to the original...
    row = (uint32_t *)(ptr);
    /* FIBITMAP is vertically backwards, so start from the bottom line */
    /* and work up */
    row += height*width;
    /* This will give the code the apperance of starting at the top left */
    /* and working downward */
    /* So x=y=0 is the upper left corner pixel */

    for (y = 0; y < height; y++) {
        /* Decrement row at the beginning of every iteration */
        int square_recent = 0;
        int last_x = 0;
        row -= width;
        z = 0;
        for (x = 0; x < width; ) {// x++) {
            int next_x = x + 1;
            int eager = 0;
            if (pixel_data[width*y+x].bucketed == 0) {
                pixel = (uint32_t)row[x];
                // palp = (unsigned char)(( pixel & 0xFF000000) >> 24); /* Alpha */
                // pred = (unsigned char)(( pixel & RED_MASK) >> 16); /* Red */
                // pgre = (unsigned char)(( pixel & GREEN_MASK) >> 8);  /* Green */
                // pblu = (unsigned char)(( pixel & BLUE_MASK));       /* Blue */
                // now it gets interesting
                int not_done = 1;
                // int next_x = x + 1;
                int next_y = y + 1;
                int horizontal = 0;
                int vertical = 0;
                int top = 0;
                int bottom = 0;
                int left = 0;
                int right = 0;
                int stops = 0;
                int len_line = bucket_matrix.bucket_row[cur_bucket_row].col_count;
                int luma_delta_side = 11;
                int chroma_delta_side = 15;
                int luma_delta_center = 7;
                int chroma_delta_center = 15;
                if (square_recent && (x > (min_sq_sep + last_x)) && (x < (min_sq_width + last_x)) && (len_line < 6) ) {
                    // eager for new box
                    // row[x] = 0xFFFF0000;
                    luma_delta_side = 20;
                    chroma_delta_side = 30;
                    luma_delta_center = 10;
                    chroma_delta_center = 15;
                    eager = 1;
                }


                while(not_done) {
                    // move along toward the right
                    if ((next_x < width) && (pixel_data[width*y+next_x].bucketed == 0) ) {
                        check_pixel = (uint32_t)row[next_x];
                        if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
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
                    check_row = (uint32_t *)(row);
                    not_done = 1;
                    stops = 0;
                    while (not_done) {
                        if ((next_y < height) && (pixel_data[width*next_y+x].bucketed == 0)) {
                            check_row -= width;
                            check_pixel = (uint32_t)check_row[x];
                            if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)){
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

                        /* Code to check left and top edges */
                        // row[x] = row[x] & 0xFF00FFFF;
                        // for (i=0; i < horizontal; i++) {
                        //     row[x+i] = 0xFF00FFFF;
                        //     pixel_data[width*y+x+i].bucketed = 1;
                        // }
                        // for (i=0; check_row < row; check_row+=width) {
                        //     check_row[x] = 0xFF00FFFF;
                        //     pixel_data[width*(y+i) + x].bucketed = 1;
                        //     i++;
                        // }
                        /**/

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
                        check_row = (uint32_t *)(row);
                        stops = 0;
                        for (i=0; i < use_length; i++) {
                            check_pixel = (uint32_t)check_row[x+use_length];
                            // debug
                            // check_row[x+use_length] = 0xFF0000FF;
                            if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)) {
                                stops = 0;
                                check_row -= width;
                            }
                            else {
                                stops += 1;
                                check_row -= width;
                            }
                            if (stops > MAX_STOPS) {
                                valid = 0;
                                break;
                            }
                        }
                        check_row += width*stops;
                        if (valid) {
                            // check and see if bottom is valid
                            stops = 0;
                            for (i=0; i < use_length; i++) {
                                check_pixel = (uint32_t)check_row[x+i];
                                // debug
                                // check_row[x+i] = 0xFF0000FF;
                                if (pixel_similar(pixel, check_pixel, luma_delta_side, chroma_delta_side)) {
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
                            check_row = (uint32_t *)(row);
                            check_row -= width;
                            int votes_for = 0;
                            int votes_against = 0;
                            for (i=1; i < (use_length -2); i++) {
                                for (j=1; j < (use_length - 2); j++) {
                                    check_pixel = (uint32_t)check_row[x+j];
                                    if (pixel_similar(pixel, check_pixel, luma_delta_center, chroma_delta_center)) {
                                        votes_for +=1;
                                    }
                                    else {
                                        votes_against +=1;
                                    }
                                }
                                check_row -= width;
                            }
                            if (votes_against*8 > votes_for) {
                                valid = 0;
                            }
                            // check_pixel = (uint32_t)check_row[x+use_length/2];
                            // // check_row[x+use_length/2]=0xFF00FFFF;
                            // check_row[x+use_length/2]=0x00FF0000;
                            if (valid) {
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                // bucket and claim the box as a potential
                                square_recent = 1;
                                last_x = x + use_length;
                                this_row = bucket_matrix.bucket_row[cur_bucket_row];
                                // printf("Y: %d, Row top: %d, max:%d\n", y ,this_row.row_top, this_row.max_height);
                                if (this_row.row_top < 0) {
                                    // initialize row top
                                    bucket_matrix.bucket_row[cur_bucket_row].row_top = y;
                                }
                                else if (this_row.row_top + this_row.max_height < y) {
                                    // new row
                                    // printf("!!!! NEW ROW> %d had columns %d\n", y, cur_bucket_col);
                                    cur_bucket_row++;
                                    cur_bucket_col = 0;
                                    bucket_matrix.col_len++;
                                    bucket_matrix.bucket_row[cur_bucket_row].row_top = y;
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
                                // printf("----_> Row top: %d, max:%d\n", this_row.row_top, this_row.max_height);
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
                                check_row = (uint32_t *)(row);
                                uint32_t red = 0;
                                uint32_t green = 0;
                                uint32_t blue = 0;
                                for (i=0; i < use_length; i++) {
                                    // i height
                                    for (j=0; j < use_length; j++) {
                                        // j length
                                        check_pixel = (uint32_t)check_row[x+j];
                                        red += (check_pixel & RED_MASK) >> 16;
                                        green += (check_pixel & GREEN_MASK) >> 8;
                                        blue += (check_pixel & BLUE_MASK);
                                        if ( (i==0) || (i == use_length-1) || (j==0) || (j==use_length-1)) {
                                            // draw a cyan box around the target
                                            check_row[x+j] = 0xFF00FFFF;
                                        }
                                        pixel_data[width*(y+i) + (x+j)].bucketed = 1;
                                        pixel_data[width*(y+i) + (x+j)].bucket_row = cur_bucket_row;
                                        pixel_data[width*(y+i) + (x+j)].bucket_col = cur_bucket_col;
                                        pixel_data[width*(y+i) + (x+j)].last_x = x + use_length;
                                    }
                                    check_row -= width;
                                }
                                int use_length_sqr = use_length*use_length;
                                this_box.red = red/use_length_sqr;
                                this_box.green = green/use_length_sqr;
                                this_box.blue = blue/use_length_sqr;

                                // save changes
                                this_row.box[cur_bucket_col] = this_box;
                                bucket_matrix.bucket_row[cur_bucket_row] = this_row;
                                // printf ("    > col %d has start %d\n", cur_bucket_col, bucket_matrix.bucket_row[cur_bucket_row].box[cur_bucket_col].start_x);
                                /* checking */
                                // for (i=0; i < use_length; i++) {
                                //     row[x+i] = 0xFF00FFFF;
                                //     pixel_data[width*y+x+i].bucketed = 1;
                                // }
                                // check_row = (uint32_t *)(row);
                                // check_row -= (width*(use_length));
                                // for (i=0; check_row < row; check_row+=width) {
                                //     check_row[x] = 0xFF00FFFF;
                                //     pixel_data[width*(y+i) + x].bucketed = 1;
                                //     i++;
                                // }
                                /* */

                                // increment which column value
                                cur_bucket_col++;
                                if (cur_bucket_col > max_row_len) {
                                    max_row_len = cur_bucket_col;
                                }
                            }
                        }
                    }
                }
            }
            // load previous line square reference
            else if (pixel_data[width*y+x].last_x != -1) {
                // printf (".");
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

    printf ("\n Row len %d, Column len %d\n", max_row_len, bucket_matrix.col_len);

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

    int avg_width = box_size/box_count;
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
                                // move on to next box
                                // printf("\nBucketed at row %d, col %d with %d\n", i, k, bucket_enum + j);
                                // printf("    . %d between %d and %d\n", this_box.start_x, prob_start, prob_end);
                                // printf("    . current bucket %d\n", j);
                                bucket_found = 1;
                                // bucket_enum++;

                            }
                            // else {
                            //     printf("\nCant bucket at row %d, col %d\n", i, k);
                            //     printf("    > %d not between %d and %d\n", this_box.start_x, prob_start, prob_end);
                            //     printf("    > current bucket %d, enum %d\n", j, bucket_enum + j);
                            // }
                        }
                    }
                }
            }
        }
        // save the row changes
        bucket_matrix.bucket_row[i] = this_row;
    }

    float have_red = 0;
    int need_red = 0;
    float have_green = 0;
    int need_green = 0;
    float have_blue = 0;
    int need_blue = 0;
    int count = 0;
    // Determine correctness
    for (i=bucket_type_dark_skin; i < bucket_type_invalid; i++){
        for (j=0; j < bucket_matrix.col_len; j++) {
            this_row = bucket_matrix.bucket_row[j];
            for (k=0; k < (this_row.col_count); k++) {
                this_box = this_row.box[k];
                int color_idx = (int)this_box.bucket_color;
                if (color_idx == i) {
                    have_red += this_box.red;
                    have_green += this_box.green;
                    have_blue += this_box.blue;
                    need_red += bucket_values[color_idx].red;
                    need_green += bucket_values[color_idx].green;
                    need_blue += bucket_values[color_idx].blue;
                    printf("RGB %d %d %d looks like %f %f %f for %s\n", bucket_values[color_idx].red, bucket_values[color_idx].green,
                        bucket_values[color_idx].blue, this_box.red, this_box.green, this_box.blue, bucket_values[color_idx].name);
                    count++;
                    // draw the thing
                    row = (uint32_t *)(ptr);
                    // row += (height - this_box.start_y+this_box.height)*width;
                    row += (height - this_box.start_y)*width;
                    for (y=this_box.start_y; y < (this_box.start_y + this_box.height); y++){
                        for (x=this_box.start_x; x < (this_box.start_x + this_box.width); x++) {
                            unsigned char r_val = bucket_values[color_idx].red;
                            unsigned char g_val = bucket_values[color_idx].green;
                            unsigned char b_val = bucket_values[color_idx].blue;
                            row[x] = 0xFF << 24 | r_val << 16 | g_val << 8 | b_val;
                        }
                        row -= width;
                    }
                }
            }
        }
    }
    float red_percent = have_red/need_red*100;
    float green_percent = have_green/need_green*100;
    float blue_percent = have_blue/need_blue*100;
    printf("\nRED %f %%\n", red_percent);
    printf("\nGREEN %f %%\n", green_percent);
    printf("\nBLUE %f %%\n", blue_percent);


    // sets up image for display
    for (y = 0; y < image->height; y++) {
        memcpy(image->mem + (y * image->bpl),
            ptr + ((image->height - y - 1) * image->bpl),
            image->bpl);
    }

    FreeImage_Unload(dib);

    imagebox = gtk_image_new_from_image(image, NULL);
    gtk_container_add(GTK_CONTAINER(window), imagebox);

    gtk_widget_show(imagebox);
    gtk_widget_show(window);

    gtk_main();

    // release the FreeImage library
    FreeImage_DeInitialise();

    // other frees
    free(row);
    free(pixel_data);

    return 0;
}
