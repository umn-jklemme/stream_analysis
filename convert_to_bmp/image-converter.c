#include <gtk/gtk.h>
#include <FreeImage.h>
#include <string.h>
#include <stdlib.h>
#include "image-converter.h"

void destroy(GtkWidget * widget, gpointer data)
{
    gtk_main_quit();
}

static void print_usage(char * appname)
{
    printf("Usage: %s [options]\n", appname);
    printf("Available options are\n");
    printf(" -i <image_file>     The image to load\n");
    printf(" -o <image_file.bmp> The image to output\n");
    printf(" -h                  Print this help screen and exit\n");
}

int main(int argc, char **argv)
{
    GdkVisual *visual;
    GdkImage *image;
    FIBITMAP *dib;
    int x,y,z;
    int opt, i,j;
    FILE *bmp;
    char image_name[100] = "";
    char output_name[100] = "";

    /* pull in any arguments */
    while ((opt = getopt(argc, argv, "i:o:h")) != -1) {
        switch (opt) {
            case 'i':
                strncpy(image_name, optarg, 100);
                break;

            case 'o':
                strncpy(output_name, optarg, 100);
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

    bmp = fopen(output_name, "w");
    if (bmp < 0) {
        printf("Unable to open %s for writing.\n", output_name);
        FreeImage_DeInitialise();
        return -1;
    }

    dib = FreeImage_Load(image_fif, image_name, 0);

    gtk_init(&argc, &argv);

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


    BYTE *ptr = FreeImage_GetBits(dib);
    uint32_t *row;
    unsigned char palp, pred, pgre, pblu;
    uint32_t pixel;



    row = (uint32_t *)(ptr);
    /* FIBITMAP is vertically backwards, so start from the bottom line */
    /* and work up */
    row += height*width;
    /* This will give the code the apperance of starting at the top left */
    /* and working downward */
    /* So x=y=0 is the upper left corner pixel */
    fprintf(bmp, "P3 %d %d 255\n", width, height);
    for (y = 0; y < height; y++) {
        /* Decrement row at the beginning of every iteration */
        row -= width;
        for (x = 0; x < width; x++) {
            pixel = (uint32_t)row[x];
            palp = (unsigned char)(( pixel & 0xFF000000) >> 24); /* Alpha */
            pred = (unsigned char)(( pixel & RED_MASK) >> 16); /* Red */
            pgre = (unsigned char)(( pixel & GREEN_MASK) >> 8);  /* Green */
            pblu = (unsigned char)(( pixel & BLUE_MASK));       /* Blue */
            fprintf(bmp, "%u %u %u\n", pred, pgre, pblu);
        }
    }
    printf("\n unloading...\n");
    fclose(bmp);
    FreeImage_Unload(dib);

    // release the FreeImage library
    FreeImage_DeInitialise();

    return 0;
}
