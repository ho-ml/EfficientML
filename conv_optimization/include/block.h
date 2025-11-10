struct tensor {
    int N;          // batch size or out channels
    int C;          // channels
    int H;          // height
    int W;          // width
    float *data_ptr;
};

struct conv_params {
    int K;
    int S;
    int P;
};

struct block_params {
    struct tensor input;
    struct tensor output;
    struct conv_params conv_params;

    int expansion_ratio;

    struct tensor weight_expansion;
    struct tensor weight_depthwise;
    struct tensor weight_reduction;
};

namespace block {
    class InvertedResidualBlock {
        public:
            enum IMP_TYPE {
                NAIVE,
                IM2COL,
                INPLACE,
                NCHW,
                FAST
            };

            void naive_block(const struct block_params *params);
            void block_im2col(const struct block_params *params);
            void block_inplace(const struct block_params *params);
            void block_nchw(const struct block_params *params);
            void block_fast(const struct block_params *params);
            void evaluate(IMP_TYPE type, const struct block_params *params);
    };
}