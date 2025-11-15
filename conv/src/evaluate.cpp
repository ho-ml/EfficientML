#include "conv.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define RUNS 1
#define MAX_PRECISION_ERROR 0.01

namespace conv {
    long get_memory_usage() {
        long rss = 0;
        std::ifstream status_file("/proc/self/status");

        if (!status_file.is_open()) {
            return 0;
        }

        std::string line;
        while (std::getline(status_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line.substr(6));
                iss >> rss;
                break;
            }
        }

        status_file.close();
        return rss;
    }

    bool check_identical(float actA[], float actB[], int size) {
        for (int i = 0; i < size; i++) {
            if ((abs(actA[i] - actB[i]) / actA[i]) > MAX_PRECISION_ERROR) {
                printf("%f, %f", actA[i], actB[i]);
                return false;
            }
        }

        return true;
    }

    float interval_to_ms(struct timeval *start, struct timeval *end) {
        float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
        return us_seconds / 1000;
    }

    void display(float nt, float ot, long nm, long om) {
        std::cout << "  Naive:" << std::endl;
        std::cout << "    Time: " << nt << " ms" << std::endl;
        std::cout << "    Memory: " << nm << " KB" << std::endl;

        std::cout << "  Optimized:" << std::endl;
        std::cout << "    Time: " << ot << " ms" << std::endl;
        std::cout << "    Memory: " << om << " KB" << std::endl;

        std::cout << "  Speedup: " << nt / ot << "x" << std::endl;
        if (om > 0) {
            std::cout << "  Memory Reduction: " << (1.0 - (float)om / nm) * 100 << "%" << std::endl;
        }
    }

    void evaluate(CONV_TYPE type, struct conv_params *params) {
        // initialize
        int output_size = params->output.N * params->output.C * params->output.H * params->output.W;
        float *naive_output = new float[output_size];
        float *optimized_output = new float[output_size];

        struct timeval start, end;
        float naive_time, optimized_time;
        long naive_mem, optimized_mem;

        switch(type) {
        case CONV2D: {
            Conv2d conv_op;

            // naive convolution
            params->output.data = naive_output;
            long mem_before = get_memory_usage();
            gettimeofday(&start, NULL);
            for (int i = 0; i < RUNS; i++)
                conv_op.naive_conv(params);
            gettimeofday(&end, NULL);
            naive_time = interval_to_ms(&start, &end) / RUNS;
            naive_mem = get_memory_usage() - mem_before;

            // im2col convolution
            params->output.data = optimized_output;
            mem_before = get_memory_usage();
            gettimeofday(&start, NULL);
            for (int i = 0; i < RUNS; i++)
                conv_op.conv_im2col(params);
            gettimeofday(&end, NULL);
            optimized_time = interval_to_ms(&start, &end) / RUNS;
            optimized_mem = get_memory_usage() - mem_before;

            // display results
            display(naive_time, optimized_time, naive_mem, optimized_mem);
            if (!check_identical(naive_output, optimized_output, output_size))
                printf("incorrect output of conv_im2col\n");

            break;
        }

        case POINTWISE: {
            Pointwise pw_op;

            // naive pointwise
            params->output.data = naive_output;
            long mem_before = get_memory_usage();
            gettimeofday(&start, NULL);
            for (int i = 0; i < RUNS; i++)
                pw_op.naive_pw(params);
            gettimeofday(&end, NULL);
            naive_time = interval_to_ms(&start, &end) / RUNS;
            naive_mem = get_memory_usage() - mem_before;

            // NHWC pointwise
            params->output.data = optimized_output;
            mem_before = get_memory_usage();
            gettimeofday(&start, NULL);
            for (int i = 0; i < RUNS; i++)
                pw_op.pw_nhwc(params);
            gettimeofday(&end, NULL);
            optimized_time = interval_to_ms(&start, &end) / RUNS;
            optimized_mem = get_memory_usage() - mem_before;

            // display results
            display(naive_time, optimized_time, naive_mem, optimized_mem);
            if (!check_identical(naive_output, optimized_output, output_size))
                printf("incorrect output of pw_nhwc\n");

            break;
        }

        case DEPTHWISE: {
            Depthwise dw_op;
            int input_size = params->input.N * params->input.C * params->input.H * params->input.W;
            int tmp_size = params->output.H * params->output.W;

            // naive depthwise
            params->output.data = naive_output;
            gettimeofday(&start, NULL);
            for (int i = 0; i < RUNS; i++)
                dw_op.naive_dw(params);
            gettimeofday(&end, NULL);
            naive_time = interval_to_ms(&start, &end) / RUNS;
            naive_mem = (input_size + output_size) * sizeof(float) / 1024;

            // in-place depthwise
            float *ori_input = params->input.data;
            gettimeofday(&start, NULL);
            for (int i = 0; i < RUNS; i++)
                dw_op.dw_inplace(params);
            gettimeofday(&end, NULL);
            optimized_time = interval_to_ms(&start, &end) / RUNS;
            optimized_mem = (input_size + tmp_size) * sizeof(float) / 1024;

            // display results
            display(naive_time, optimized_time, naive_mem, optimized_mem);
            if (!check_identical(naive_output, ori_input, output_size))
                printf("incorrect output of dw_inplace\n");

            break;
        }

        default:
            break;
        }

        // clean
        delete[] naive_output;
        delete[] optimized_output;
    }

}
