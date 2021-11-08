#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


float getMax(layer l, int x, int y, int c) {
    int offset = -(l.size / 2);
    if (l.size % 2 == 0) {
        offset++;
    }
    float max = -INFINITY;
    for (int i = 0; i < l.size; i++) {
        for (int j = 0; j < l.size; j++) {
            int row = y + i + offset;
            int col = x + j + offset;
            if (row >= 0 && row < l.height && col >= 0 && col < l.width) {
                int index = col + l.width * (row + c * l.height);
                float val = l.x->data[index];
                if (val > max) {
                    max = val;
                }
            }
            
        }
    }
    return max;
}

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (int i = 0; i < l.channels; i++) {
        int y = 0;
        for (int j = 0; j < outh; j++) {
            int x = 0;
            for (int k = 0; k < outw; k++) {
                float max = getMax(l, x, y, i);
                out.data[k + outw * (j + outh * i)] = max;
                x += l.stride;
            }
            y += l.stride;
        }
    }
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    
    printf("in.rows %i\n", in.rows);
    printf("in.cols %i\n", in.cols);
    printf("outw %i\n", outw);
    printf("outh %i\n", outh);
    printf("l.width %i\n", l.width);
    printf("l.height %i\n", l.height);
    printf("channels %i\n", l.channels);
    printf("dx.rows %i\n", dx.rows);
    printf("dx.cols %i\n", dx.cols);
    printf("dy.rows %i\n", dy.rows);
    printf("dy.cols %i\n", dy.cols);
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.



    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

