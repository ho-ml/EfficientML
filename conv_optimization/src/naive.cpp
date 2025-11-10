#include "block.h"

#include <sys/time.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define RUNS 1

namespace block {
    void InvertedResidualBlock::naive_block(const struct block_params *params) {
        // initialize
        int n, ic, oc, h, w;
        int kh, kw, h_in, w_in;
        int idx_i, idx_w, idx_o;

        // tensor
        const struct tensor *input = &params->input, *output = &params->output;
        const struct tensor *weight_exp = &params->weight_expansion, *weight_dw = &params->weight_depthwise, *weight_red = &params->weight_reduction;
        const struct conv_params *conv_params = &params->conv_params;

        // data
        float *input_data = input->data_ptr, *output_data = output->data_ptr;
        float *weight_exp_data = weight_exp->data_ptr, *weight_dw_data = weight_dw->data_ptr, *weight_red_data = weight_red->data_ptr;
        
        // intermediate activations
        int c_exp = input->C * params->expansion_ratio;
        float *act_exp = new float[input->N * c_exp * input->H * input->W];
        float *act_dw = new float[input->N * c_exp * input->H * input->W];

        // Expansion: [N, c_in, H, W] -> [N, c_exp, H, W]
        for (n = 0; n < input->N; n++) {
            for (oc = 0; oc < c_exp; oc++) {
                for(h = 0; h < input->H; h++) {
                    for (w = 0; w < input->W; w++) {
                        float acc = 0;

                        for (ic = 0; ic < input->C; ic++) {
                            idx_i = n * (input->C * input->H * input->W) + ic * (input->H * input->W) + h * input->W + w;
                            idx_w = oc * input->C + ic;
                            acc += input_data[idx_i] * weight_exp_data[idx_w];
                        }

                        idx_o = n * (c_exp * input->H * input->W) + oc * (input->H * input->W) + h * input->W + w;
                        act_exp[idx_o] = acc;
                    }
                }
            }
        }
        
        // Depthwise: [N, c_exp, H, W] -> [N, c_exp, H, W]
        for (n = 0; n < input->N; n++) {
            for (oc = 0; oc < c_exp; oc++) {
                for(h = 0; h < input->H; h++) {
                    for (w = 0; w < input->W; w++) {
                        float acc = 0;
                        
                        // 3 x 3 convolution
                        for (kh = 0; kh < conv_params->K; kh++) {
                            for (kw = 0; kw < conv_params->K; kw++) {
                                h_in = h * conv_params->S - conv_params->P + kh;
                                w_in = w * conv_params->S - conv_params->P + kw;

                                if (h_in >= 0 && h_in < input->H && w_in >= 0 && w_in < input->W) {
                                    idx_i = n * (c_exp * input->H * input->W) + oc * (input->H * input->W) + h_in * input->W + w_in;
                                    idx_w = oc * (conv_params->K * conv_params->K) + kh * conv_params->K + kw;
                                    acc += act_exp[idx_i] * weight_dw_data[idx_w];
                                } 
                            }
                        }

                        idx_o = n * (c_exp * input->H * input->W) + oc * (input->H * input->W) + h * input->W + w;
                        act_dw[idx_o] = acc;
                    }
                }
            }
        }

        // Reduction: [N, c_exp, H, W] -> [N, c_out, H, W]
        for (n = 0; n < input->N; n++) {
            for (oc = 0; oc < output->C; oc++) {
                for(h = 0; h < input->H; h++) {
                    for (w = 0; w < input->W; w++) {
                        float acc = 0;

                        for (ic = 0; ic < c_exp; ic++) {
                            idx_i = n * (c_exp * input->H * input->W) + ic * (input->H * input->W) + h * input->W + w;
                            idx_w = oc * c_exp + ic;
                            acc += act_dw[idx_i] * weight_red_data[idx_w];
                        }

                        idx_o = n * (output->C * input->H * input->W) + oc * (input->H * input->W) + h * input->W + w;
                        output_data[idx_o] = acc;
                    }
                }
            }
        }

        // free
        delete[] act_exp;
        delete[] act_dw;
    }

    float interval_to_ms(struct timeval *start, struct timeval *end) {
        float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
        return us_seconds / 1000;
    }
    
    void InvertedResidualBlock::evaluate(IMP_TYPE type, const struct block_params *params) {
        struct timeval start, end;
        float ms;
        std::string function_name;
        
        // start time
        gettimeofday(&start, NULL);

        // choose implementation
        switch (type) {
        case NAIVE:
            function_name = "naive_block";
            for (int i = 0; i < RUNS; i++)
                this->naive_block(params);
            break;
        case IM2COL:
            function_name = "block_im2col";
            for (int i = 0; i < RUNS; i++)
                this->block_im2col(params);
            break;
        case INPLACE:
            function_name = "block_inplace";
            for (int i = 0; i < RUNS; i++)
                this->block_inplace(params);
            break;
        case NCHW:
            function_name = "block_nchw";
            for (int i = 0; i < RUNS; i++)
                this->block_nchw(params);
            break;
        case FAST:
            function_name = "block_fast";
            for (int i = 0; i < RUNS; i++)
                this->block_fast(params);
            break;
        default:
            break;
        }
        
        // end time
        gettimeofday(&end, NULL);

        // time taken
        ms = interval_to_ms(&start, &end);
        std::cout << function_name << ": " << ms << " ms" << std::endl;
    }
}