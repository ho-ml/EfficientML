#include "conv.h"

#include <stdio.h>
#include <assert.h>

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

    }

    void Depthwise::dw_inplace(const struct conv_params *params) {

    }
}