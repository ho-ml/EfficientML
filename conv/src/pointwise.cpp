#include "conv.h"

#include <stdio.h>
#include <assert.h>

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
        

    }

    void Pointwise::pw_nchw(const struct conv_params *params) {
        
    }
}