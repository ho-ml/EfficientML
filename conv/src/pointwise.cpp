#include "conv.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

namespace conv {
    void Pointwise::check_shape(const struct conv_params *params) {
        // output resolution
        assert(params->output.H == params->input.H);
        assert(params->output.W == params->input.W);

        // kernel dimension
        assert(params->config.kernel_size == 1);
        assert(params->kernel.Cin == params->input.C);
        assert(params->kernel.Cout == params->output.C);
    }

    void Pointwise::naive_pw(const struct conv_params *params) {
        // initialize
        int n, h, w, ic, oc;

        const struct activation *input = &params->input, *output = &params->output;
        const struct weight *kernel = &params->kernel;

        float *input_data = input->data, *output_data = output->data;
        float *kernel_data = kernel->data;

        // sanity check
        check_shape(params);

        // implementation
        for (n = 0; n < output->N; n++) {
            float *output_ptrn = output_data + n * (output->C * output->H * output->W);
            float *input_ptrn = input_data + n * (input->C * input->H * input->W);

            for (oc = 0; oc < output->C; oc++) {
                float *output_ptrc = output_ptrn + oc * (output->H * output->W);
                float *kernel_ptrc = kernel_data + oc * input->C;

                for (h = 0; h < output->H; h++) {
                    for (w = 0; w < output->W; w++) {
                        float acc = 0;

                        for (ic = 0; ic < input->C; ic++) {
                            int idx_i = ic * (input->H * input->W) + h * input->W + w;
                            acc += input_ptrn[idx_i] * kernel_ptrc[ic];
                        }

                        output_ptrc[h * output->W + w] = acc;
                    }
                }
            }
        }
    }

    void to_nhwc(float *nhwc, float *nchw, int N, int C, int H, int W) {
        int n, h, w, c;

        for (n = 0; n < N; n++) {
            float *nhwc_ptrn = nhwc + n * (H * W * C);
            float *nchw_ptrn = nchw + n * (C * H * W);

            for (h = 0; h < H; h++) {
                for (w = 0; w < W; w++) {
                    float *nhwc_ptrr = nhwc_ptrn + h * (W * C) + w * C;
                    
                    for (c = 0; c < C; c++)                
                        nhwc_ptrr[c] = nchw[c * (H * W) + h * W + w];
                }
            }
        }
    }

    void to_nchw(float *nchw, float *nhwc, int N, int C, int H, int W) {
        int n, c, h, w;

        for (n = 0; n < N; n++) {
            float *nchw_ptrn = nchw + n * (C * H * W);
            float *nhwc_ptrn = nhwc + n * (H * W * C);

            for (c = 0; c < C; c++) {
                float *nchw_ptrc = nchw_ptrn + c * (H * W);

                for (h = 0; h < H; h++) {
                    for (w = 0; w < W; w++) {
                        nchw_ptrc[h * W + w] = nhwc[h * (W * C) + w * C + c];
                    }
                }
            } 
        }
    }

    void Pointwise::pw_nhwc(const struct conv_params *params) {
        // initialize
        int n, h, w, ic, oc;

        const struct activation *input = &params->input, *output = &params->output;
        const struct weight *kernel = &params->kernel;

        float *input_data = input->data, *output_data = output->data;
        float *kernel_data = kernel->data;

        // sanity check
        check_shape(params);

        int input_size = input->N * input->C * input->H * input->W;
        int output_size = output->N * output->C * output->H * output->W;

        // Convert input from NCHW to NHWC
        float *input_nhwc = new float[input_size];
        float *output_nhwc = new float[output_size];
        to_nhwc(input_nhwc, input_data, input->N, input->C, input->H, input->W);

        // implementation
        for (n = 0; n < output->N; n++) {
            float *input_ptrn = input_nhwc + n * (input->H * input->W * input->C);
            float *output_ptrn = output_nhwc + n * (output->H * output->W * output->C);

            for (h = 0; h < output->H; h++) {
                for (w = 0; w < output->W; w++) {
                    float *input_ptrr = input_ptrn + h * (input->W * input->C) + w * input->C;
                    float *output_ptrr = output_ptrn + h * (output->W * output->C) + w * output->C;

                    for (oc = 0; oc < output->C; oc++) {
                        float acc = 0;
                        float *kernel_ptrc = kernel_data + oc * input->C;

                        for (ic = 0; ic < input->C; ic++)
                            acc += input_ptrr[ic] * kernel_ptrc[ic];

                        output_ptrr[oc] = acc;
                    }
                }
            }
        }

        // convert output from NHWC to NCHW
        to_nchw(output_data, output_nhwc, output->N, output->C, output->H, output->W);

        // free
        delete[] input_nhwc;
        delete[] output_nhwc;
    }
}