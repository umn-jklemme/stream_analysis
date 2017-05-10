#ifndef IMAGE_ANALYSIS_H
#define IMAGE_ANALYSIS_H

#define RED_MASK 0x00FF0000
#define GREEN_MASK 0x0000FF00
#define BLUE_MASK 0x000000FF

#define Y_MASK 0x00FF0000
#define CB_MASK 0x0000FF00
#define CR_MASK 0x000000FF

#define MAX_STOPS 4
#define MAX_STOPS_GPU 7
#define MAX_SCAN_Y 1080
#define MAX_SCAN_X 1920

int pixel_similar(uint32_t pixel_a, uint32_t pixel_b, int max_diff_luma, int max_diff_chroma);
int pixel_similar_ycbcr(uint32_t pixel_a, uint32_t pixel_b, int max_diff_luma, int max_diff_chroma);
uint32_t rgb_to_ycbcr(uint32_t pixel);

typedef enum _bucket_type {
    bucket_type_default,
    bucket_type_dark_skin,      /* 1 */
    bucket_type_light_skin,     /* 2 */
    bucket_type_blue_sky,       /* 3 */
    bucket_type_foliage,        /* 4 */
    bucket_type_blue_flower,    /* 5 */
    bucket_type_bluish_green,   /* 6 */
    bucket_type_orange,         /* 7 */  //
    bucket_type_purplish_blue,  /* 8 */  //
    bucket_type_moderate_red,   /* 9 */  //
    bucket_type_purple,         /* 10 */ //
    bucket_type_yellow_green,   /* 11 */ //
    bucket_type_orange_yellow,  /* 12 */ //
    bucket_type_blue,           /* 13 */
    bucket_type_green,          /* 14 */
    bucket_type_red,            /* 15 */
    bucket_type_yellow,         /* 16 */
    bucket_type_magenta,        /* 17 */
    bucket_type_cyan,           /* 18 */
    bucket_type_white,          /* 19 */ //
    bucket_type_neutral_8,      /* 20 */ //
    bucket_type_neutral_6_5,    /* 21 */ //
    bucket_type_neutral_5,      /* 22 */ //
    bucket_type_neutral_3_5,    /* 23 */ //
    bucket_type_black,          /* 24 */ //
    bucket_type_invalid
} bucket_type;

typedef struct _bucket_values_s {
    int     red;
    int     green;
    int     blue;
    char    name[20];
} bucket_values_s;


typedef struct _global_needs_s {
    float   have_red;
    float   have_green;
    float   have_blue;
    int     need_red;
    int     need_green;
    int     need_blue;
    float   red_percent;
    float   green_percent;
    float   blue_percent;
} global_needs_s;


typedef struct _bucket_data_s {
    bucket_type     bucket_color;
    int             start_x;        /* upper left */
    int             start_y;        /* upper left */
    int             mid_x;       /* middle */
    int             mid_y;       /* middle */
    int             width;
    int             height;
    float           red;
    float           green;
    float           blue;
    float           red_percent;
    float           green_percent;
    float           blue_percent;
    int             did_global_add;
    int             valid;
} bucket_data_s;

#define NUM_SAVE_COLS 20
#define NUM_REAL_COLS 6

// Column data
typedef struct _bucket_col_s {
    int             start_x_count;
    int             start_x_total;
    int             max_x_start;
    int             min_width;
} bucket_col_s;


// Row data
typedef struct _bucket_row_s {
    bucket_data_s   box[NUM_SAVE_COLS];
    int             max_height;
    int             min_height;
    int             row_top;
    int             max_row_top;
    int             valid;
    int             start_y_count;
    int             start_y_total;
    int             col_count;
} bucket_row_s;

#define NUM_SAVE_ROWS 10
#define NUM_REAL_ROWS 4

// Matrix data
typedef struct _bucket_mat_s {
    bucket_row_s    bucket_row[NUM_SAVE_ROWS];
    int             col_len;
    bucket_col_s    bucket_col[NUM_SAVE_COLS];
} bucket_mat_s;

typedef struct _pixel_data_s {
    int             bucketed;
    bucket_type     color_type;
    int             bucket_row;
    int             bucket_col;
    int             last_x;
} pixel_data_s;


#endif  /* IMAGE_ANALYSIS_H */