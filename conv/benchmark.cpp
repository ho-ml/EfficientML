#include "conv.h"
#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>

// define
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1

#define BATCH_SIZE 1
#define IN_CHANNEL 32
#define IN_HEIGHT 112
#define IN_WIDTH 112
#define OUT_CHANNEL 32
#define OUT_HEIGHT (IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1
#define OUT_WIDTH (IN_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1

// input
float input_activation[BATCH_SIZE * IN_CHANNEL * IN_HEIGHT * IN_WIDTH];

// convolution
float weight_conv[OUT_CHANNEL * IN_CHANNEL * KERNEL_SIZE * KERNEL_SIZE];
float output_conv[BATCH_SIZE * OUT_CHANNEL * OUT_HEIGHT * OUT_WIDTH];

// pointwise convolution
float weight_pw[OUT_CHANNEL * IN_CHANNEL * 1 * 1];
float output_pw[BATCH_SIZE * OUT_CHANNEL * IN_HEIGHT * IN_WIDTH];

// depthwise convolution
float weight_dw[OUT_CHANNEL * 1 * KERNEL_SIZE * KERNEL_SIZE];
float output_dw[BATCH_SIZE * OUT_CHANNEL * OUT_HEIGHT * OUT_WIDTH];

void initialize_tensor(float A[], int size) {
    for (int i = 0; i < size; i++)
        A[i] = (float)(rand()) / (float)(RAND_MAX);
}

using namespace conv;

bool run_switch(std::string target, std::string type) {
    if (target == "ALL" || target == type)
        return true;
    
    return false;
}

int main(int argc, char* argv[]) {
    auto target = "ALL";
    if (argc == 2)
        target = argv[1];

    // initialize
    initialize_tensor(input_activation, BATCH_SIZE * IN_CHANNEL * IN_HEIGHT * IN_WIDTH);
    initialize_tensor(weight_conv, OUT_CHANNEL * IN_CHANNEL * KERNEL_SIZE * KERNEL_SIZE);
    initialize_tensor(weight_pw, OUT_CHANNEL * IN_CHANNEL * 1 * 1);
    initialize_tensor(weight_dw, OUT_CHANNEL * 1 * KERNEL_SIZE * KERNEL_SIZE);

    struct conv_params params;

    // input activation
    params.input.N = BATCH_SIZE;
    params.input.C = IN_CHANNEL;
    params.input.H = IN_HEIGHT;
    params.input.W = IN_WIDTH;
    params.input.data = input_activation;

    // Conv2D
    if (run_switch(target, "CONV2D")) {
        params.output.N = BATCH_SIZE;
        params.output.C = OUT_CHANNEL;
        params.output.H = OUT_HEIGHT;
        params.output.W = OUT_WIDTH;
        params.output.data = output_conv;

        params.kernel.Cout = OUT_CHANNEL;
        params.kernel.Cin = IN_CHANNEL;
        params.kernel.KH = KERNEL_SIZE;
        params.kernel.KW = KERNEL_SIZE;
        params.kernel.data = weight_conv;

        params.config.kernel_size = KERNEL_SIZE;
        params.config.stride = STRIDE;
        params.config.padding = PADDING;

        evaluate(CONV2D, &params);
    }

    // Pointwise
    if (run_switch(target, "POINTWISE")) {
        params.output.N = BATCH_SIZE;
        params.output.C = OUT_CHANNEL;
        params.output.H = IN_HEIGHT;
        params.output.W = IN_WIDTH;
        params.output.data = output_pw;

        params.kernel.Cout = OUT_CHANNEL;
        params.kernel.Cin = IN_CHANNEL;
        params.kernel.KH = 1;
        params.kernel.KW = 1;
        params.kernel.data = weight_pw;

        params.config.kernel_size = 1;
        params.config.stride = 1;
        params.config.padding = 0;

        evaluate(POINTWISE, &params);
    }

    // Depthwise
    if (run_switch(target, "DEPTHWISE")) {
        params.output.N = BATCH_SIZE;
        params.output.C = OUT_CHANNEL;
        params.output.H = OUT_HEIGHT;
        params.output.W = OUT_WIDTH;
        params.output.data = output_dw;

        params.kernel.Cout = OUT_CHANNEL;
        params.kernel.Cin = 1;
        params.kernel.KH = KERNEL_SIZE;
        params.kernel.KW = KERNEL_SIZE;
        params.kernel.data = weight_dw;

        params.config.kernel_size = KERNEL_SIZE;
        params.config.stride = STRIDE;
        params.config.padding = PADDING;

        evaluate(DEPTHWISE, &params);
    }

    return 0;
}