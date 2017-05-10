CC = gcc
CPP = g++
COMPILERFLAGS = -O3
INCLUDE = -I../../Dist
VGALIBRARIES = -lfreeimage -lvga
VGAINCLUDE = -I/usr/include/asm
GTKLIBRARIES = -lfreeimage `pkg-config --libs gtk+-2.0`
GTKINCLUDE = `pkg-config --cflags gtk+-2.0`
CFLAGS = $(COMPILERFLAGS) $(INCLUDE)

all: default

default: image-converter

image-converter: image-converter.c
	$(CC) $(CFLAGS) $< -o $@ $(GTKLIBRARIES) $(GTKINCLUDE)
	strip $@

clean:
	rm -f core image-converter
