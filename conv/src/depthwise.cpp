#include "conv.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

namespace conv {
    void Depthwise::check_shape(const struct conv_params *params) {
        int expected_H = (params->input.H + 2 * params->config.padding - params->config.kernel_size) / params->config.stride + 1;
        int expected_W = (params->input.W + 2 * params->config.padding - params->config.kernel_size) / params->config.stride + 1;

        // output shape
        assert(params->output.H == expected_H);
        assert(params->output.W == expected_W);
        assert(params->output.C == params->input.C);

        // kernel dimension
        assert(params->kernel.Cin == 1);
        assert(params->kernel.Cout == params->output.C);
    }

    void Depthwise::naive_dw(const struct conv_params *params) {
        // initialize
        int n, c, oh, ow;
        int ih, iw, kh, kw;

        const struct activation *input = &params->input, *output = &params->output;
        const struct weight *kernel = &params->kernel;
        const struct conv_config *config = &params->config;

        float *input_data = input->data, *output_data = output->data;
        float *kernel_data = kernel->data;

        // sanity check
        check_shape(params);

        // implementation
        for (n = 0; n < output->N; n++) {
            float *output_ptrn = output_data + n * (output->C * output->H * output->W);
            float *input_ptrn = input_data + n * (input->C * input->H * input->W);

            for (c = 0; c < output->C; c++) {
                float *output_ptrc = output_ptrn + c * (output->H * output->W);
                float *input_ptrc = input_ptrn + c * (input->H * input->W);
                float *kernel_ptrc = kernel_data + c * (kernel->KH * kernel->KW);

                for (oh = 0; oh < output->H; oh++) {
                    for (ow = 0; ow < output->W; ow++) {
                        float acc = 0;

                        // K x K convolutions
                        for (kh = 0; kh < kernel->KH; kh++) {
                            for (kw = 0; kw < kernel->KW; kw++) {
                                ih = oh * config->stride - config->padding + kh;
                                iw = ow * config->stride - config->padding + kw;

                                if (ih >= 0 && ih < input->H && iw >= 0 && iw < input->W)
                                    acc += input_ptrc[ih * input->W + iw] * kernel_ptrc[kh * kernel->KW + kw];
                            }
                        }

                        output_ptrc[oh * output->W + ow] = acc;
                    }
                }
            }
        }

    }

    void Depthwise::dw_inplace(const struct conv_params *params) {
        // initialize
        int n, c, oh, ow;
        int ih, iw, kh, kw;

        const struct activation *input = &params->input, *output = &params->output;
        const struct weight *kernel = &params->kernel;
        const struct conv_config *config = &params->config;
        
        float *input_data = input->data;
        float *kernel_data = kernel->data;
        float *tmp = new float[output->H *output->W];

        // sanity check
        assert(input->H == output->H);
        assert(input->W == output->W);

        // implementation
        for (n = 0; n < output->N; n++) {
            float *input_ptrn = input_data + n * (input->C * input->H * input->W);

            for (c = 0; c < output->C; c++) {
                float *input_ptrc = input_ptrn + c * (input->H * input->W);
                float *kernel_ptrc = kernel_data + c * (kernel->KH * kernel->KW);

                for (oh = 0; oh < output->H; oh++) {
                    for (ow = 0; ow < output->W; ow++) {
                        float acc = 0;

                        // K x K convolutions
                        for (kh = 0; kh < kernel->KH; kh++) {
                            for (kw = 0; kw < kernel->KW; kw++) {
                                ih = oh * config->stride - config->padding + kh;
                                iw = ow * config->stride - config->padding + kw;

                                if (ih >= 0 && ih < input->H && iw >= 0 && iw < input->W)
                                    acc += input_ptrc[ih * input->W + iw] * kernel_ptrc[kh * kernel->KW + kw];
                            }
                        }

                        tmp[oh * output->W + ow] = acc;
                    }
                }

                // update the input
                memcpy(input_ptrc, tmp, output->H * output->W * sizeof(float));
            }
        }

        delete[] tmp;
    }
}