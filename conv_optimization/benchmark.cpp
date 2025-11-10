#include "block.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>

#define MAX_PRECISION_ERROR 0.01

#define BATCH_SIZE 1
#define IN_CHANNEL 32
#define HEIGHT 112
#define WIDTH 112
#define EXPANSION_RATIO 6
#define OUT_CHANNEL 32
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1

float input_tensor[BATCH_SIZE * IN_CHANNEL * HEIGHT * WIDTH];
float output_tensor[BATCH_SIZE * OUT_CHANNEL * HEIGHT * WIDTH];
float naive_output[BATCH_SIZE * OUT_CHANNEL * HEIGHT * WIDTH];

const int exp_channel = IN_CHANNEL * EXPANSION_RATIO;
float weight_expansion[exp_channel * IN_CHANNEL * 1 * 1];
float weight_depthwise[exp_channel * 1 * KERNEL_SIZE * KERNEL_SIZE];
float weight_reduction[OUT_CHANNEL * exp_channel * 1 * 1];

bool check_identical(float tensorA[], float tensorB[], int size) {
    for (int i = 0; i < size; i++) {
        if ((abs(tensorA[i] - tensorB[i]) / tensorA[i]) > MAX_PRECISION_ERROR) {
            printf("%f, %f", tensorA[i], tensorB[i]);
            return false;
        } 
    }

    return true;
}

void initialize_tensor(float A[], int size) {
    for (int i = 0; i < size; i++) {
        A[i] = (float)(rand()) / (float)(RAND_MAX);
    }
}

using namespace block;

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
    initialize_tensor(input_tensor, BATCH_SIZE * IN_CHANNEL * HEIGHT * WIDTH);
    initialize_tensor(weight_expansion, exp_channel * IN_CHANNEL * 1 * 1);
    initialize_tensor(weight_depthwise, exp_channel * 1 * KERNEL_SIZE * KERNEL_SIZE);
    initialize_tensor(weight_reduction, OUT_CHANNEL * exp_channel * 1 * 1);

    // 객체 생성
    InvertedResidualBlock block_op = InvertedResidualBlock();

    // 파라미터 설정
    struct block_params params;

    // input activation
    params.input.N = BATCH_SIZE; params.input.C = IN_CHANNEL;
    params.input.H = HEIGHT; params.input.W = WIDTH; 
    params.input.data_ptr = input_tensor;

    // output activation
    params.output.N = BATCH_SIZE; params.output.C = OUT_CHANNEL;
    params.output.H = HEIGHT; params.output.W = WIDTH;
    
    // convolution configuration
    params.conv_params.K = KERNEL_SIZE; params.conv_params.S = STRIDE; params.conv_params.P = PADDING;
    params.expansion_ratio = EXPANSION_RATIO;

    // weight (expansion)
    params.weight_expansion.N = exp_channel; params.weight_expansion.C = IN_CHANNEL;
    params.weight_expansion.H = 1; params.weight_expansion.W = 1;
    params.weight_expansion.data_ptr = weight_expansion;

    // weight (depthwise)
    params.weight_depthwise.N = exp_channel; params.weight_depthwise.C = 1;
    params.weight_depthwise.H = KERNEL_SIZE; params.weight_depthwise.W = KERNEL_SIZE; 
    params.weight_depthwise.data_ptr = weight_depthwise;

    // weight (reduction)
    params.weight_reduction.N = OUT_CHANNEL; params.weight_reduction.C = exp_channel;
    params.weight_reduction.H = 1; params.weight_reduction.W = 1;
    params.weight_reduction.data_ptr = weight_reduction;

    // baseline
    params.output.data_ptr = naive_output;
    block_op.evaluate(InvertedResidualBlock::NAIVE, &params);
    params.output.data_ptr = output_tensor;

    // im2col
    if (run_switch(target, "im2col")) {
        block_op.evaluate(InvertedResidualBlock::IM2COL, &params);
        if (!check_identical(naive_output, output_tensor, BATCH_SIZE * OUT_CHANNEL * HEIGHT * WIDTH))
            printf("incorrect output of block_im2col\n");
    }

    // inplace
    if (run_switch(target, "inplace")) {
        block_op.evaluate(InvertedResidualBlock::INPLACE, &params);
        if (!check_identical(naive_output, output_tensor, BATCH_SIZE * OUT_CHANNEL * HEIGHT * WIDTH))
            printf("incorrect output of block_inplace\n");
    }

    // nchw
    if (run_switch(target, "nchw")) {
        block_op.evaluate(InvertedResidualBlock::NCHW, &params);
        if (!check_identical(naive_output, output_tensor, BATCH_SIZE * OUT_CHANNEL * HEIGHT * WIDTH))
            printf("incorrect output of block_nchw\n");
    }

    // fast
    if (run_switch(target, "fast")) {
        block_op.evaluate(InvertedResidualBlock::FAST, &params);
        if (!check_identical(naive_output, output_tensor, BATCH_SIZE * OUT_CHANNEL * HEIGHT * WIDTH))
            printf("incorrect output of block_fast\n");
    }

    return 0;
}