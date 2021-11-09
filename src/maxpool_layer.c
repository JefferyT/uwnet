#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

float getMax(layer l, int x, int y, int c, int im)
{
    int width = l.width * l.height * l.channels;
    int offset = -(l.size / 2);
    if (l.size % 2 == 0)
    {
        offset++;
    }
    float max = -INFINITY;
    for (int i = 0; i < l.size; i++)
    {
        for (int j = 0; j < l.size; j++)
        {
            int row = y + i + offset;
            int col = x + j + offset;
            if (row >= 0 && row < l.height && col >= 0 && col < l.width)
            {
                int index = col + l.width * (row + c * l.height) + width * im;
                float val = l.x->data[index];
                if (val > max)
                {
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
    // // Saving our input
    // // Probably don't change this
    // free_matrix(*l.x);
    // *l.x = copy_matrix(in);

    // int outw = (l.width - 1) / l.stride + 1;
    // int outh = (l.height - 1) / l.stride + 1;
    // matrix out = make_matrix(in.rows, outw * outh * l.channels);


    // // TODO: 6.1 - iterate over the input and fill in the output with max values
    // for (int im = 0; im < in.rows; im++)
    // {
    //     for (int i = 0; i < l.channels; i++)
    //     {
    //         int y = 0;
    //         for (int j = 0; j < outh; j++)
    //         {
    //             int x = 0;
    //             for (int k = 0; k < outw; k++)
    //             {
    //                 float max = getMax(l, x, y, i, im);
    //                 out.data[k + outw * (j + outh * i) + out.cols * im] = max;
    //                 x += l.stride;
    //             }
    //             y += l.stride;
    //         }
    //     }
    // }

    // return out;
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO:6.1 - iterate over the input and fill in the output with max values
    for (int batch = 0; batch < in.rows; batch++) {
        matrix batch_image = make_matrix(l.height * l.channels, l.width);
        free_matrix(batch_image);
        batch_image.data = in.data + batch * in.cols;

        matrix batch_out = make_matrix(outh*l.channels, outw);
        free_matrix(batch_out);
        batch_out.data = out.data + batch * out.cols;

        for(int k=0; k<l.channels; k++){
            int c = 0;
            for(int i=0; i<l.height; i+=l.stride){
                for(int j=0; j<l.width; j+=l.stride){
                    float max = 0;
                    int first = 1;
                    for(int x=0; x<l.size; x++){
                        for(int y=0; y<l.size; y++){
                            int xoff = i - l.size%2 + x;
                            int yoff = j - l.size%2 + y;

                            float f = 0;
                            if(xoff >= 0 && x < batch_image.rows && yoff >= 0 && y < batch_image.cols){
                                f = batch_image.data[l.width * l.height * k + xoff * batch_image.cols + yoff ];
                            }

                            if(first){
                                max = f;
                                first = 0;
                            } else {
                                max = MAX(max, f);
                            }
                        }
                    }
                    batch_out.data[outh*outw*k + c] = max;
                    c++;
                }
            }
        }
    }


    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    matrix dx = make_matrix(dy.rows, l.width * l.height * l.channels);

    int outw = (l.width - 1) / l.stride + 1;
    int outh = (l.height - 1) / l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    int offset = -(l.size / 2);
    if (l.size % 2 == 0)
    {
        offset++;
    }
    int width = l.width * l.height * l.channels;
    for (int im = 0; im < in.rows; im++)
    {
        for (int i = 0; i < l.channels; i++)
        {
            int y = 0;
            for (int j = 0; j < outh; j++)
            {
                int x = 0;
                for (int k = 0; k < outw; k++)
                {
                    // finds the max of the window
                    // indices of the input matrix
                    int max_row = -1;
                    int max_col = -1;
                    float max = -INFINITY;
                    // loops through each row of the window
                    for (int rowOff = 0; rowOff < l.size; rowOff++)
                    {
                        // loops through each column of the window
                        for (int colOff = 0; colOff < l.size; colOff++)
                        {
                            // row is the row of the input matrix
                            int row = y + rowOff + offset;
                            // col is the column of the input matrix
                            int col = x + colOff + offset;
                            if (row >= 0 && row < l.height && col >= 0 && col < l.width)
                            {

                                int index = col + l.width * (row + i * l.height) + width * im;
                                float val = in.data[index];
                                if (val > max)
                                {
                                    max = val;
                                    max_row = row;
                                    max_col = col;
                                }
                            }
                        }
                    }
                    float val = dy.data[k + outw * (j + outh * i) + dy.cols * im];
                    dx.data[max_col + l.width * (max_row + i * l.height)] += val;
                    x += l.stride;
                }
                y += l.stride;
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay) {}

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
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update = update_maxpool_layer;
    return l;
}



