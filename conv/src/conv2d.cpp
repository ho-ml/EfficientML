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
            for (oh = 0; oh < output->H; oh++) {
                for (ow = 0; ow < output->W; ow++) {
                    for (oc = 0; oc < output->C; oc++) {
                        float acc = 0;

                        // K x K convolutions
                        for (kh = 0; kh < kernel->KH; kh++) {
                            for (kw = 0; kw < kernel->KW; kw++) {
                                ih = oh * config->stride - config->padding + kh;
                                iw = ow * config->stride - config->padding + kw;

                                if (ih >= 0 && ih < input->H && iw >= 0 && iw < input->W) {
                                    for (ic = 0; ic < input->C; ic++) {
                                        idx_i = n * (input->H * input->W * input->C) + ih * (input->W * input->C) + iw * input->C + ic;
                                        idx_k = oc * (kernel->KH * kernel->KW * kernel->Cin) + kh * (kernel->KW * kernel->Cin) + kw * kernel->Cin + ic;
                                        acc += input_data[idx_i] * kernel_data[idx_k];
                                    }
                                }
                            }
                        }

                        idx_o = n * (output->H * output->W * output->C) + oh * (output->W * output->C) + ow * output->C + oc;
                        output_data[idx_o] = acc;
                    }
                }
            }
        }
    }

    void mgemm(
        const float *input, const float *kernel, float *output,
        const int Cin, const int Hin, const int Win,
        const int Cout, const int Hout, const int Wout,
        const int K, const int S, const int P
    ) {
        /*
            implicit gemm implementation
            - im2col:   (Hout * Wout, K * K * Cin)
            - kernel:   (Cout, K * K * Cin)
            - output:   (Hout * Wout, Cout)
        */
        
        // initialize
        const int ROW = Hout * Wout;
        const int COL = K * K * Cin;
        int oc, oh, ow, kh, kw, ic, ih, iw;
        
        // implementation
        for (int col = 0; col < COL; col++) {
            kh = col / (K * Cin);
            kw = (col % (K * Cin)) / Cin;
            ic = col % Cin;

            for (int row = 0; row < ROW; row++) {
                // indexing
                oh = row / Wout; ow = row % Wout;
                ih = oh * S - P + kh; iw = ow * S - P + kw;
                
                // multiplication
                if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                    for (oc = 0; oc < Cout; oc++)
                        output[row * Cout + oc] += input[ih * (Win * Cin) + iw * Cin + ic] * kernel[oc * COL + col];
                }
            }
        }
    }

    void Conv2d::conv_im2col(const struct conv_params *params) {
        // initialize
        int n;

        const struct activation *input = &params->input, *output = &params->output;
        const struct weight *kernel = &params->kernel;
        const struct conv_config *config = &params->config;

        float *input_data = input->data, *output_data = output->data;
        float *kernel_data = kernel->data;

        // sanity check
        check_shape(params);
        
        // clear output
        for (n = 0; n < output->N * output->H * output->W * output->C; n++)
            output_data[n] = 0;

        // implementation
        for (n = 0; n < output->N; n++) {
            // move pointers
            float *input_batch = input_data + n * (input->H * input->W * input->C);
            float *output_batch = output_data + n * (output->H * output->W * output->C);

            // implicit gemm
            mgemm(
                input_batch, kernel_data, output_batch,
                input->C, input->H, input->W,
                output->C, output->H, output->W,
                config->kernel_size, config->stride, config->padding
            );
        }

    }
}