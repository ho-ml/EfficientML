# **Lab 5 Report**

Optimize LLM on Edge Devices

## 1. Loop Unrolling

- fill in the starter code in `kernel/template/loop_unrolling.cc` to implement loop unrolling

- run the `./evaluate.sh loop_unrolling` to evaluate performance improvement

### 1.1 Implementation

Please copy and paste your implementation in `loop_unrolling.cc`

```cpp

void MatmulOperator::mat_mul_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col += 4) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int8 activation
                const int8_t *a_int8 = &A->int8_data_ptr[row * k + ch];

                // pointer of the int4 weights
                uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
                uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];

                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];

                // scale of weight
                float s_w0 = params->scales[(col * k + ch) / block_size];
                float s_w1 = params->scales[((col + 1) * k + ch) / block_size];
                float s_w2 = params->scales[((col + 2) * k + ch) / block_size];
                float s_w3 = params->scales[((col + 3) * k + ch) / block_size];
#ifdef QM_ARM
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                for (int qj = 0; qj < 16; qj++) {
                    // load a packed byte
                    uint8_t packed_int4_0 = w0_int4[qj];
                    uint8_t packed_int4_1 = w1_int4[qj];
                    uint8_t packed_int4_2 = w2_int4[qj];
                    uint8_t packed_int4_3 = w3_int4[qj];

                    // decode low 4 bits into sint8
                    int8_t w0_de_0 = (packed_int4_0 & 0x0F) - 8;
                    int8_t w1_de_0 = (packed_int4_1 & 0x0F) - 8;
                    int8_t w2_de_0 = (packed_int4_2 & 0x0F) - 8;
                    int8_t w3_de_0 = (packed_int4_3 & 0x0F) - 8;

                    // decode high 4 bits into sint8
                    int8_t w0_de_16 = (packed_int4_0 >> 4) - 8;
                    int8_t w1_de_16 = (packed_int4_1 >> 4) - 8;
                    int8_t w2_de_16 = (packed_int4_2 >> 4) - 8;
                    int8_t w3_de_16 = (packed_int4_3 >> 4) - 8;
                    
                    // int8 multiply and accumulatation
                    intermediate_sum0 += (a_int8[qj] * w0_de_0 + a_int8[qj + 16] * w0_de_16);
                    intermediate_sum1 += (a_int8[qj] * w1_de_0 + a_int8[qj + 16] * w1_de_16);
                    intermediate_sum2 += (a_int8[qj] * w2_de_0 + a_int8[qj + 16] * w2_de_16);
                    intermediate_sum3 += (a_int8[qj] * w3_de_0 + a_int8[qj + 16] * w3_de_16);
                }

                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;

                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w0_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_w1_2nd = params->scales[((col + 1) * k + ch) / block_size + 1];
                float s_w2_2nd = params->scales[((col + 2) * k + ch) / block_size + 1];
                float s_w3_2nd = params->scales[((col + 3) * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
            
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                int intermediate_sum0_2nd = 0, intermediate_sum1_2nd = 0, intermediate_sum2_2nd = 0,
                    intermediate_sum3_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // load a packed byte
                    uint8_t packed_int4_0 = w0_int4[qj];
                    uint8_t packed_int4_1 = w1_int4[qj];
                    uint8_t packed_int4_2 = w2_int4[qj];
                    uint8_t packed_int4_3 = w3_int4[qj];

                    // decode low 4 bits into sint8
                    int8_t w0_de_0 = (packed_int4_0 & 0x0F) - 8;
                    int8_t w1_de_0 = (packed_int4_1 & 0x0F) - 8;
                    int8_t w2_de_0 = (packed_int4_2 & 0x0F) - 8;
                    int8_t w3_de_0 = (packed_int4_3 & 0x0F) - 8;

                    // decode high 4 bits into sint8
                    int8_t w0_de_32 = (packed_int4_0 >> 4) - 8;
                    int8_t w1_de_32 = (packed_int4_1 >> 4) - 8;
                    int8_t w2_de_32 = (packed_int4_2 >> 4) - 8;
                    int8_t w3_de_32 = (packed_int4_3 >> 4) - 8;
                    
                    // int8 multiply and accumulatation
                    // block 1
                    intermediate_sum0 += a_int8[qj] * w0_de_0;
                    intermediate_sum1 += a_int8[qj] * w1_de_0;
                    intermediate_sum2 += a_int8[qj] * w2_de_0;
                    intermediate_sum3 += a_int8[qj] * w3_de_0;
                    
                    // block 2
                    intermediate_sum0_2nd += a_int8[qj + 32] * w0_de_32;
                    intermediate_sum1_2nd += a_int8[qj + 32] * w1_de_32;
                    intermediate_sum2_2nd += a_int8[qj + 32] * w2_de_32;
                    intermediate_sum3_2nd += a_int8[qj + 32] * w3_de_32;
                }
                
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc0 += (float)intermediate_sum0_2nd * s_a_2nd * s_w0_2nd;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc1 += (float)intermediate_sum1_2nd * s_a_2nd * s_w1_2nd;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc2 += (float)intermediate_sum2_2nd * s_a_2nd * s_w2_2nd;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                acc3 += (float)intermediate_sum3_2nd * s_a_2nd * s_w3_2nd;
                
                // process two blocks
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
};

```

### 1.2 Comparative Study

How does the performance in GOPs, achieved through loop unrolling on your computer, compare to the reference implementation? Please explain the performance difference.

- Result
    | Section | Total time(ms) | Average time(ms) | GOPs |
    | :--- | :---: | :---: | :---: |
    | **reference** | 2376.385986 | 237.638000 | 1.103120 |
    | **loop unrolling** | 1146.931030 | 114.693001 | 2.285613 |

- Comment
    - loop의 branch 횟수를 줄여 branch prediction 실패로 인한 overhead 방지
    - `uint8` 에서 `sint8` 디코딩 시 `-8.0` 대신 `-8` 을 사용하여 추가적인 fp 연산 방지
    - `sint8` 자료형을 `signed char` 대신 `int8_t` 를 사용하여 여러 플랫폼에서 일관된 동작을 보장


## 2. Multithreading

- fill in the starter code in `kernel/template/multithreading.cc` to implement multithreading

- run the `./evaluate.sh multithreading` to evaluate performance improvement

### 2.1 Implementation

Please copy and paste your implementation in `multithreading.cc`

```cpp
struct multithreading_thread_args {
    int start, end;
    const struct matmul_params* params;
};

static void* multithreading_worker_func(void* args) {
    struct multithreading_thread_args* mat_args = (struct multithreading_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col++) {
            float acc = 0;

            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights
                uint8_t* w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                // pointer of the int8 activation
                const int8_t* a_int8 = &A->int8_data_ptr[row * k + ch];
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
#ifdef QM_ARM
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0;
                // process 16 bytes of weigths (128 bit)
                for (int qj = 0; qj < 16; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    int8_t w_de_0 = (packed_int4_0 & 0x0F) - 8;
                    int8_t w_de_16 = (packed_int4_0 >> 4) - 8;

                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum += a_int8[qj + 16] * w_de_16;
                }

                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];

                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0, intermediate_sum_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    int8_t w_de_0 = (packed_int4_0 & 0x0F) - 8;
                    int8_t w_de_16 = (packed_int4_0 >> 4) - 8;

                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum_2nd += a_int8[qj + 32] * w_de_16;
                }

                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                acc += (float)intermediate_sum_2nd * s_a_2nd * s_w_2nd;
                
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc;
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_multithreading(struct matmul_params* params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_thread_args threads_args[num_thread];

    // Thread creation
    for (int t = 0; t < num_thread; t++) {
        threads_args[t].params = params;
        threads_args[t].start = t * (n / num_thread);
        threads_args[t].end = (t + 1) * (n / num_thread);
        
        pthread_create(&thread_pool[t], NULL, multithreading_worker_func, &threads_args[t]);
    }

    // Join threads
    for (int t = 0; t < num_thread; t++) {
        pthread_join(thread_pool[t], NULL);
    }
};

```

### 2.2 Comparative Study

How does the performance in GOPs, achieved through multithreading on your computer, compare to the reference implementation? Please explain the performance difference.

- Result
    | Section | Total time(ms) | Average time(ms) | GOPs |
    | :--- | :---: | :---: | :---: |
    | **reference** | 2376.385986 | 237.638000 | 1.103120 |
    | **multithreading** | 508.687988 | 50.868000 | 5.153335 |

- Comment
    - 출력 차원을 4개의 thread로 나누어 병렬 처리
    - reference 대비 4배 이상 빠른 속도 달성

## 3. SIMD Programming

- fill in the starter code in `kernel/template/simd_programming.cc` to implement SIMD Programming

- run the `./evaluate.sh simd_programming` to evaluate performance improvement

### 3.1 Implementation

Please copy and paste your implementation in `simd_programming.cc`

```cpp

void MatmulOperator::mat_mul_simd_programming(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
#ifdef QM_ARM
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            
            // pointer of the int4 weights
            const uint8_t *w_start = &B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const int8_t *a_start = &A->int8_data_ptr[row * k];
            // scale of activation
            float *s_a = &params->A_scales[row * k / 32];
            // scale of weight
            float *s_w = &params->scales[col * k / 32];

            const int num_block = k / block_size;
            // lowbit mask
            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            // offsets
            const int8x16_t offsets = vdupq_n_s8(8);

            // Compute each block
            for (int q = 0; q < num_block; q++) {
                // load 32x4bit (16 bytes) weight
                const uint8x16_t w0 = vld1q_u8(w_start);
                w_start += 16;

                // unpack the weights using lowbit mask
                // `vshrq_n_u8`: right shift operation
                // `vreinterpretq_s8_u8`: convert uint8x16_t to int8x16_t
                int8x16_t w_de_0 = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w_de_16 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w0, 4), mask_low4bit));

                // apply zero_point to weights using offsets
                w_de_0 = vsubq_s8(w_de_0, offsets);
                w_de_16 = vsubq_s8(w_de_16, offsets);

                // load 32 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                a_start += 32;

                // int32x4 vector to store intermediate sum
                int32x4_t int_sum0 = vdupq_n_s32(0);
                
                // dot product
                // `vdotq_s32`: dot product and accumulate result into destination register
                int_sum0 = vdotq_s32(int_sum0, a0, w_de_0);
                int_sum0 = vdotq_s32(int_sum0, a1, w_de_16);
                
                // scaling and accumulation
                // `vmlaq_n_f32`: vector mac with scalar
                // `vcvtq_f32_s32`: convert int32x4_t to float32x4_t
                float s_0 = *s_a++ * *s_w++;
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
            }

            C->data_ptr[row * n + col] = vaddvq_f32(sumv0);
#endif
#ifdef QM_x86
            __m256 acc0 = _mm256_setzero_ps();
            // pointer of the int4 weights
            const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];
            // scale of weight
            float *s_ptr = &params->scales[col * k / 32];
            // scale of activation
            float *sa_ptr = &params->A_scales[row * k / 32];

            const int num_block = k / block_size;
            // lowbit mask
            const __m256i lowMask = _mm256_set1_epi8(0xF);
            // zero point (offset)
            const __m256i zero_point = _mm256_set1_epi8(8);
            // vector which is filled with 1
            const __m256i ones = _mm256_set1_epi16(1);
            
            // Compute two blocks in each iteration
            for (int q = 0; q < num_block; q += 2) {
                // load 256 bit from w_strat
                __m256i raw_w = _mm256_loadu_si256(w_start);
                
                // unpack the weights using lowbit mask
                // `_mm256_srli_epi16`: right shift operation
                __m256i w_0, w_128;
                w_0 = _mm256_and_si256(raw_w, lowMask);
                w_128 = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), lowMask);

                // apply zero_point to weights
                w_0 = _mm256_sub_epi8(w_0, zero_point);
                w_128 = _mm256_sub_epi8(w_128, zero_point);

                // Perform int8 dot product with _mm256_maddubs_epi16
                // `__m256i _mm256_maddubs_epi16(__m256i s1, __m256i s2)`:
                // (1) multiplies vertically s1(unsigned) with the corresponding s2(signed)
                // (2) add each adjacent pair of signed words
                // (3) pack the saturated result to the destination vector
                // 
                // utilize _mm256_maddubs_epi16 which only takes unsigned s1:
                // A x W = (A x sign(W)) x abs(W)
        
                // __m256 vector to store lower and upper halves sum
                __m256i dot, dot2;
                
                // Get absolute values of weights
                const __m256i uw = _mm256_sign_epi8(w_0, w_0);
                const __m256i uw2 = _mm256_sign_epi8(w_128, w_128);

                // Load activation
                __m256i activation = a_start[0];
                __m256i activation2 = a_start[1];
                
                // Change the sign of activation depending on the sign of corresponding weights
                const __m256i sa = _mm256_sign_epi8(activation, w_0);
                const __m256i sa2 = _mm256_sign_epi8(activation2, w_128);
                
                // int8 dot product
                dot = _mm256_maddubs_epi16(uw, sa);
                dot2 = _mm256_maddubs_epi16(uw2, sa2);

                // Convert int32 vectors to floating point vectors
                const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
                const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
                __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
                __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);

                // Create vectors for scales
                __m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
                __m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);

                // apply scales to intermediate results
                acc0 = _mm256_fmadd_ps(intermediate, v_s, acc0);
                acc0 = _mm256_fmadd_ps(intermediate2, v_s2, acc0);

                // move pointer
                s_ptr += 2;
                sa_ptr += 2;
                w_start += 1;
                a_start += 2;
            }

            float *ptr = (float *)&acc0;
            C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
#endif
        }
    }
};

```

### 3.2 Comparative Study

How does the performance in GOPs, achieved through SIMD Programming on your computer, compare to the reference implementation? Please explain the performance difference.

- Result
    | Section | Total time(ms) | Average time(ms) | GOPs |
    | :--- | :---: | :---: | :---: |
    | **reference** | 2376.385986 | 237.638000 | 1.103120 |
    | **simd programming** | 562.036987 | 56.202999 | 4.664177 |

- Comment
    - k개의 embed dimension에 대해 SIMD를 통해 16byte씩 나누어 병렬 처리
    - reference 대비 약 4배 빠른 속도 달성

## 4. Multithreading with Loop Unrolling

- fill in the starter code in `kernel/template/multithreading_loop_unrolling.cc` to implement multithreading and loop unrolling 

- run the `./evaluate.sh multithreading_loop_unrolling` to evaluate performance improvement

### 4.1 Implementation

Please copy and paste your implementation in `multithreading_loop_unrolling.cc`

```cpp

struct multithreading_loop_unrolling_thread_args {
    int start, end;
    const struct matmul_params *params;
};

static void *multithreading_loop_unrolling_worker_func(void *args) {
    struct multithreading_loop_unrolling_thread_args *mat_args =
        (struct multithreading_loop_unrolling_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col += 4) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;

            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];

                // pointer of the int4 weights
                uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
                uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];

                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];

                // scale of weight
                float s_w0 = params->scales[(col * k + ch) / block_size];
                float s_w1 = params->scales[((col + 1) * k + ch) / block_size];
                float s_w2 = params->scales[((col + 2) * k + ch) / block_size];
                float s_w3 = params->scales[((col + 3) * k + ch) / block_size];
#ifdef QM_ARM
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                for (int qj = 0; qj < 16; qj++) {
                    // load a packed byte
                    uint8_t packed_int4_0 = w0_int4[qj];
                    uint8_t packed_int4_1 = w1_int4[qj];
                    uint8_t packed_int4_2 = w2_int4[qj];
                    uint8_t packed_int4_3 = w3_int4[qj];

                    // decode low 4 bits into sint8
                    int8_t w0_de_0 = (packed_int4_0 & 0x0F) - 8;
                    int8_t w1_de_0 = (packed_int4_1 & 0x0F) - 8;
                    int8_t w2_de_0 = (packed_int4_2 & 0x0F) - 8;
                    int8_t w3_de_0 = (packed_int4_3 & 0x0F) - 8;
                    
                    // decode high 4 bits into sint8
                    int8_t w0_de_16 = (packed_int4_0 >> 4) - 8;
                    int8_t w1_de_16 = (packed_int4_1 >> 4) - 8;
                    int8_t w2_de_16 = (packed_int4_2 >> 4) - 8;
                    int8_t w3_de_16 = (packed_int4_3 >> 4) - 8;

                    // int8 multiply and accumulatation
                    intermediate_sum0 += (a_int8[qj] * w0_de_0 + a_int8[qj + 16] * w0_de_16);
                    intermediate_sum1 += (a_int8[qj] * w1_de_0 + a_int8[qj + 16] * w1_de_16);
                    intermediate_sum2 += (a_int8[qj] * w2_de_0 + a_int8[qj + 16] * w2_de_16);
                    intermediate_sum3 += (a_int8[qj] * w3_de_0 + a_int8[qj + 16] * w3_de_16);
                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w0_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_w1_2nd = params->scales[((col + 1) * k + ch) / block_size + 1];
                float s_w2_2nd = params->scales[((col + 2) * k + ch) / block_size + 1];
                float s_w3_2nd = params->scales[((col + 3) * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];

                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                int intermediate_sum0_2nd = 0, intermediate_sum1_2nd = 0, intermediate_sum2_2nd = 0,
                    intermediate_sum3_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // load a packed byte
                    uint8_t packed_int4_0 = w0_int4[qj];
                    uint8_t packed_int4_1 = w1_int4[qj];
                    uint8_t packed_int4_2 = w2_int4[qj];
                    uint8_t packed_int4_3 = w3_int4[qj];

                    // decode low 4 bits into sint8
                    int8_t w0_de_0 = (packed_int4_0 & 0x0F) - 8;
                    int8_t w1_de_0 = (packed_int4_1 & 0x0F) - 8;
                    int8_t w2_de_0 = (packed_int4_2 & 0x0F) - 8;
                    int8_t w3_de_0 = (packed_int4_3 & 0x0F) - 8;

                    // decode high 4 bits into sint8
                    int8_t w0_de_32 = (packed_int4_0 >> 4) - 8;
                    int8_t w1_de_32 = (packed_int4_1 >> 4) - 8;
                    int8_t w2_de_32 = (packed_int4_2 >> 4) - 8;
                    int8_t w3_de_32 = (packed_int4_3 >> 4) - 8;
                    
                    // int8 multiply and accumulatation
                    // block 1
                    intermediate_sum0 += a_int8[qj] * w0_de_0;
                    intermediate_sum1 += a_int8[qj] * w1_de_0;
                    intermediate_sum2 += a_int8[qj] * w2_de_0;
                    intermediate_sum3 += a_int8[qj] * w3_de_0;
                    
                    // block 2
                    intermediate_sum0_2nd += a_int8[qj + 32] * w0_de_32;
                    intermediate_sum1_2nd += a_int8[qj + 32] * w1_de_32;
                    intermediate_sum2_2nd += a_int8[qj + 32] * w2_de_32;
                    intermediate_sum3_2nd += a_int8[qj + 32] * w3_de_32;
                }
                
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc0 += (float)intermediate_sum0_2nd * s_a_2nd * s_w0_2nd;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc1 += (float)intermediate_sum1_2nd * s_a_2nd * s_w1_2nd;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc2 += (float)intermediate_sum2_2nd * s_a_2nd * s_w2_2nd;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                acc3 += (float)intermediate_sum3_2nd * s_a_2nd * s_w3_2nd;
                // process two blocks
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_multithreading_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    assert(params->block_size % 32 == 0);  // support block size to be multiples of 32
    assert(A->row == C->row);              // support block size to be multiples of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_loop_unrolling_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // Thread creation
    for (int t = 0; t < num_thread; t++) {
        threads_args[t].params = params;
        threads_args[t].start = t * (n / num_thread);
        threads_args[t].end = (t + 1) * (n / num_thread);
        
        pthread_create(&thread_pool[t], NULL, multithreading_loop_unrolling_worker_func, &threads_args[t]);
    }

    // Join threads
    for (int t = 0; t < num_thread; t++) {
        pthread_join(thread_pool[t], NULL);
    }
};


```

### 4.2 Comparative Study

How does the performance in GOPs, achieved through multithreading and loop unrolling on your computer, compare to the reference implementation? Please explain the performance difference.

- Result
    | Section | Total time(ms) | Average time(ms) | GOPs |
    | :--- | :---: | :---: | :---: |
    | **reference** | 2376.385986 | 237.638000 | 1.103120 |
    | **multithreading with loop unrolling** | 364.946991 | 36.493999 | 7.183070 |

- Comment
    - 위의 multithreading과 loop unrolling을 동시 적용
    - reference 대비 약 6배 빠른 속도 달성

## 5. Combination of All Techniques

### 5.1 Implementation

Please copy and paste your implementation in `all_techniques.cc`

```cpp

static void *all_techniques_worker_func(void *args) {
    struct w4a8_thread_args *mat_args = (struct w4a8_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;  // block_size = 32

    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start_j; col < mat_args->end_j; col++) {
#ifdef QM_ARM
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);
            float32x4_t sumv2 = vdupq_n_f32(0.0f);
            float32x4_t sumv3 = vdupq_n_f32(0.0f);
            // pointer of the int4 weights
            const unsigned char *w_start = &params->B.int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const signed char *a_start = &params->A.int8_data_ptr[row * k];
            // scale of activation
            float *s_a = &params->A_scales[row * k / 32];
            // scale of weight
            float *s_w = &params->scales[col * k / 32];
            
            // constants
            // lowbit mask
            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            // offset
            const int8x16_t offsets = vdupq_n_s8(8);

            // process four blocks each iteration
            for (int q = 0; q < num_block; q += 4) {
                // load 32x4bit (16 bytes) weight
                const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                w_start += 64;

                // unpack the weights using lowbit mask
                int8x16_t w0_de_0 = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w0_de_16 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w0, 4), mask_low4bit));

                int8x16_t w1_de_0 = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t w1_de_16 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w1, 4), mask_low4bit));

                int8x16_t w2_de_0 = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                int8x16_t w2_de_16 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w2, 4), mask_low4bit));

                int8x16_t w3_de_0 = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                int8x16_t w3_de_16 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w3, 4), mask_low4bit));

                // apply zero_point to weights using offsets
                w0_de_0 = vsubq_s8(w0_de_0, offsets);
                w0_de_16 = vsubq_s8(w0_de_16, offsets);
                w1_de_0 = vsubq_s8(w1_de_0, offsets);
                w1_de_16 = vsubq_s8(w1_de_16, offsets);
                w2_de_0 = vsubq_s8(w2_de_0, offsets);
                w2_de_16 = vsubq_s8(w2_de_16, offsets);
                w3_de_0 = vsubq_s8(w3_de_0, offsets);
                w3_de_16 = vsubq_s8(w3_de_16, offsets);

                // load 128 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                const int8x16_t a4 = vld1q_s8(a_start + 64);
                const int8x16_t a5 = vld1q_s8(a_start + 80);
                const int8x16_t a6 = vld1q_s8(a_start + 96);
                const int8x16_t a7 = vld1q_s8(a_start + 112);
                a_start += 128;

                // int32x4 vector to store intermediate sum
                int32x4_t int_sum0, int_sum1, int_sum2, int_sum3;

                // dot product
                int_sum0 = vdotq_s32(int_sum0, a0, w0_de_0);
                int_sum0 = vdotq_s32(int_sum0, a1, w0_de_16);
                int_sum1 = vdotq_s32(int_sum1, a2, w1_de_0);
                int_sum1 = vdotq_s32(int_sum1, a3, w1_de_16);
                int_sum2 = vdotq_s32(int_sum2, a4, w2_de_0);
                int_sum2 = vdotq_s32(int_sum2, a5, w2_de_16);
                int_sum3 = vdotq_s32(int_sum3, a6, w3_de_0);
                int_sum3 = vdotq_s32(int_sum3, a7, w3_de_16);

                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;
                float s_2 = *s_a++ * *s_w++;
                float s_3 = *s_a++ * *s_w++;

                // accumulation
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum1), s_1);
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum2), s_2);
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum3), s_3);
            }
            params->C.data_ptr[row * n + col] = vaddvq_f32(sumv0);
#endif
#ifdef QM_x86
            __m256 accumulator = _mm256_setzero_ps();
            float *s_ptr = &params->scales[col * k / 32];
            float *sa_ptr = &params->A_scales[row * k / 32];
            const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
            const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];
            const int num_block = k / block_size;

            // constant
            // lowbit mask
            const __m256i lowMask = _mm256_set1_epi8(0xF);
            // offset
            const __m256i zero_point = _mm256_set1_epi8(8);
            // vector which is filled with 1
            const __m256i ones = _mm256_set1_epi16(1);
            
            // Compute four blocks = 128 4-bit weights in each iteration
            for (int q = 0; q < num_block; q += 4) {
                // load 256 bit from w_strat
                __m256i raw_w = _mm256_loadu_si256(w_start);
                __m256i raw_w_next = _mm256_loadu_si256(w_start + 1);
                
                // unpack the weights using lowbit mask
                __m256i w_0, w_128, w_0_next, w_128_next;
                w_0 = _mm256_and_si256(raw_w, lowMask);
                w_128 = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), lowMask);
                w_0_next = _mm256_and_si256(raw_w_next, lowMask);
                w_128_next = _mm256_and_si256(_mm256_srli_epi16(raw_w_next, 4), lowMask);

                // apply zero_point to weights
                w_0 = _mm256_sub_epi8(w_0, zero_point);
                w_128 = _mm256_sub_epi8(w_128, zero_point);
                w_0_next = _mm256_sub_epi8(w_0_next, zero_point);
                w_128_next = _mm256_sub_epi8(w_128_next, zero_point);

                // Perform int8 dot product with _mm256_maddubs_epi16
                // __m256 vector to store lower and upper halves sum
                __m256i dot, dot2, dot3, dot4;

                // Get absolute values of weights
                const __m256i uw = _mm256_sign_epi8(w_0, w_0);
                const __m256i uw2 = _mm256_sign_epi8(w_128, w_128);
                const __m256i uw_next = _mm256_sign_epi8(w_0_next, w_0_next);
                const __m256i uw2_next = _mm256_sign_epi8(w_128_next, w_128_next);

                // Load activation
                __m256i activation = a_start[0];
                __m256i activation2 = a_start[1];
                __m256i activation_next = a_start[2];
                __m256i activation2_next = a_start[3];

                // Change the sign of activation depending on the sign of corresponding weights
                const __m256i sa = _mm256_sign_epi8(activation, w_0);
                const __m256i sa2 = _mm256_sign_epi8(activation2, w_128);
                const __m256i sa_next = _mm256_sign_epi8(activation_next, w_0_next);
                const __m256i sa2_next = _mm256_sign_epi8(activation2_next, w_128_next);

                // int8 dot product
                dot = _mm256_maddubs_epi16(uw, sa);
                dot2 = _mm256_maddubs_epi16(uw2, sa2);
                dot3 = _mm256_maddubs_epi16(uw_next, sa_next);
                dot4 = _mm256_maddubs_epi16(uw2_next, sa2_next);

                // Convert int32 vectors to floating point vectors
                const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
                const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
                const __m256i summed_pairs3 = _mm256_madd_epi16(ones, dot3);
                const __m256i summed_pairs4 = _mm256_madd_epi16(ones, dot4);
                __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
                __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);
                __m256 intermediate3 = _mm256_cvtepi32_ps(summed_pairs3);
                __m256 intermediate4 = _mm256_cvtepi32_ps(summed_pairs4);

                // Create vectors for scales
                __m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
                __m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);
                __m256 v_s3 = _mm256_set1_ps(s_ptr[2] * sa_ptr[2]);
                __m256 v_s4 = _mm256_set1_ps(s_ptr[3] * sa_ptr[3]);

                // apply scales to intermediate results
                accumulator = _mm256_fmadd_ps(intermediate, v_s, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate2, v_s2, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate3, v_s3, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate4, v_s4, accumulator);
                s_ptr += 4;
                sa_ptr += 4;
                w_start += 2;
                a_start += 4;
            }
            float *ptr = (float *)&accumulator;
            C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
#endif
        }
    }

    return NULL;
}

void MatmulOperator::mat_mul_all_techniques(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size % 32 == 0);  // support block size to be multiples of 32
    assert(A->row == C->row);              // support block size to be multiples of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = 8;
    pthread_t thread_pool[num_thread];
    struct w4a8_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // Thread creation
    for (int t = 0; t < num_thread; t++) {
        threads_args[t].params = params;
        threads_args[t].start_j = t * (C->column / num_thread);
        threads_args[t].end_j = (t + 1) * (C->column / num_thread);
        
        pthread_create(&thread_pool[t], NULL, all_techniques_worker_func, &threads_args[t]);
    }

    // Join threads
    for (int t = 0; t < num_thread; t++) {
        pthread_join(thread_pool[t], NULL);
    }
};

```

### 5.2 Comparative Study

How does the performance in GOPs, achieved through all optimization techniques on your computer, compare to the reference implementation? Please explain the performance difference.

- Result
    | Section | Total time(ms) | Average time(ms) | GOPs |
    | :--- | :---: | :---: | :---: |
    | **reference** | 2376.385986 | 237.638000 | 1.103120 |
    | **multithreading with loop unrolling** | 97.814003 | 9.781000 | 26.800254 |

- Comment
    - loop unrolling, mutltithreading, simd programming 동시 적용
    - reference 대비 약 25배 빠른 속도 달성

## 6. Additional Experiments

Any optimization techniques on your mind? Try to implement them to improve the performance further! Create a pull request in [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) and get verified by the TA.

### 6.1 