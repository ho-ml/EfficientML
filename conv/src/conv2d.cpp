#include "conv.h"
#include <stdio.h>
#include <assert.h>

// naive convolution
namespace conv {
    void Conv2d::check_shape(const struct conv_params *params) {
        int expected_H = (params->input.H + 2 * params->config.padding - params->config.kernel_size) / params->config.stride + 1;
        int expected_W = (params->input.W + 2 * params->config.padding - params->config.kernel_size) / params->config.stride + 1;
        
        // output resolution
        assert(params->output.H == expected_H);
        assert(params->output.W == expected_W);
        
        // weight dimensions
        assert(params->kernel.Cin == params->input.C);
        assert(params->kernel.Cout == params->output.C);
    }

    void Conv2d::naive_conv(const struct conv_params *params) {
        // initialize
        int n, oc, oh, ow;
        int ic, ih, iw, kh, kw;
        int idx_i, idx_k, idx_o;

        const struct activation *input = &params->input, *output = &params->output;
        const struct weight *kernel = &params->kernel;
        const struct conv_config *config = &params->config;

        float *input_data = input->data, *output_data = output->data;
        float *kernel_data = kernel->data;

        // sanity check
        check_shape(params);

        // implementation
        for (n = 0; n < output->N; n++) {
            for (oc = 0; oc < output->C; oc++) {
                for (oh = 0; oh < output->H; oh++) {
                    for (ow = 0; ow < output->W; ow++) {
                        float acc = 0;

                        // K x K convolutions
                        for (kh = 0; kh < kernel->KH; kh++) {
                            for (kw = 0; kw < kernel->KW; kw++) {
                                ih = oh * config->stride - config->padding + kh;
                                iw = ow * config->stride - config->padding + kw;

                                if (ih >= 0 && ih < input->H && iw >= 0 && iw < input->W) {
                                    for (ic = 0; ic < input->C; ic++) {
                                        idx_i = n * (input->C * input->H * input->W) + ic * (input->H * input->W) + ih * input->W + iw;
                                        idx_k = oc * (kernel->Cin * kernel->KH * kernel->KW) + ic * (kernel->KH * kernel->KW) + kh * kernel->KW + kw;
                                        acc += input_data[idx_i] * kernel_data[idx_k];
                                    }
                                }
                            }
                        }

                        idx_o = n * (output->C * output->H * output->W) + oc * (output->H * output->W) + oh * output->W + ow;
                        output_data[idx_o] = acc;
                    }
                }
            }
        }
    }

    void Conv2d::conv_im2col(const struct conv_params *params) {
        // initialize
        int n, oc, oh, ow, ic, ih, iw, kh, kw;
        int buffer_idx, input_idx, ridx, cidx;

        const struct activation *input = &params->input, *output = &params->output;
        const struct weight *kernel = &params->kernel;
        const struct conv_config *config = &params->config;

        float *input_data = input->data, *output_data = output->data;
        float *kernel_data = kernel->data;

        // sanity check
        check_shape(params);

        // gemm matrix dimension
        // kernel: (Cout, Cin * KH * KW)
        // im2col: (Cin * KH * KW, Hout * Wout)
        // output: (Cout, Hout * Wout)
        int M_gemm = output->C;
        int K_gemm = kernel->Cin * kernel->KH * kernel->KW;
        int N_gemm = output->H * output->W;
        float *col_buffer = new float[K_gemm * N_gemm];

        // implementation
        for (n = 0; n < output->N; n++) {
            const float *input_batch = input_data + n * (input->C * input->H * input->W);

            // transform image to column matrix
            cidx = 0;
            for (oh = 0; oh < output->H; oh++) {
                for (ow = 0; ow < output->W; ow++) {
                    ridx = 0;

                    for (ic = 0; ic < input->C; ic++) {
                        for (kh = 0; kh < kernel->KH; kh++) {
                            for (kw = 0; kw < kernel->KW; kw++) {
                                ih = oh * config->stride - config->padding + kh;
                                iw = ow * config->stride - config->padding + kw;
                                buffer_idx = ridx * N_gemm + cidx;

                                if (ih >= 0 && ih < input->H && iw >= 0 && iw < input->W) {
                                    input_idx = ic * (input->H * input->W) + ih * input->W + iw;
                                    col_buffer[buffer_idx] = input_batch[input_idx];
                                }
                                else {
                                    col_buffer[buffer_idx] = 0.0f;
                                }

                                ridx++;
                            }
                        }
                    }

                    cidx++;
                }
            }

            // gemm (gemm.cpp에서 구현)
            float *output_batch = output_data + n * (output->C * output->H * output->W);
            gemm(kernel_data, col_buffer, output_batch, M_gemm, N_gemm, K_gemm);
        }

        // free the buffer
        delete[] col_buffer;
    }
}