struct activation {
    int N;
    int C;
    int H;
    int W;
    float *data;  // (N, H, W, C)
};

struct weight {
    int Cout;
    int Cin;
    int KH;
    int KW;
    float *data;  // (Cout, KH, KW, Cin)
};

struct conv_config {
    int kernel_size;
    int stride;
    int padding;
};

struct conv_params {
    struct activation input;
    struct activation output;
    struct weight kernel;
    struct conv_config config;
};

namespace conv {
    enum CONV_TYPE {
        CONV2D,
        POINTWISE,
        DEPTHWISE
    };

    class Conv2d {
        public:
            void naive_conv(const struct conv_params *params);
            void conv_im2col(const struct conv_params *params);

        private:
            void check_shape(const struct conv_params *params);
    };

    class Pointwise {
        public:
            void naive_pw(const struct conv_params *params);
            void pw_nchw(const struct conv_params *params);

        private:
            void check_shape(const struct conv_params *params);
    };

    class Depthwise {
        public:
            void naive_dw(const struct conv_params *params);
            void dw_inplace(const struct conv_params *params);

        private:
            void check_shape(const struct conv_params *params);
    };

    void evaluate(CONV_TYPE type, struct conv_params *params);
}