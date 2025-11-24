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
    | **reference** | 2383.989990 | 238.399002 | 1.099602 |
    | **loop unrolling** | 1158.503052 | 115.849998 | 2.262782 |

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

```

### 2.2 Comparative Study

How does the performance in GOPs, achieved through multithreading on your computer, compare to the reference implementation? Please explain the performance difference.

```text

```

## 3. SIMD Programming

- fill in the starter code in `kernel/template/simd_programming.cc` to implement SIMD Programming

- run the `./evaluate.sh simd_programming` to evaluate performance improvement

### 3.1 Implementation

Please copy and paste your implementation in `simd_programming.cc`

```cpp

```

### 3.2 Comparative Study

How does the performance in GOPs, achieved through SIMD Programming on your computer, compare to the reference implementation? Please explain the performance difference.

```text

```

## 4. Multithreading with Loop Unrolling

- fill in the starter code in `kernel/template/multithreading_loop_unrolling.cc` to implement multithreading and loop unrolling 

- run the `./evaluate.sh multithreading_loop_unrolling` to evaluate performance improvement

### 4.1 Implementation

Please copy and paste your implementation in `multithreading_loop_unrolling.cc`

```cpp

```

### 4.2 Comparative Study

How does the performance in GOPs, achieved through multithreading and loop unrolling on your computer, compare to the reference implementation? Please explain the performance difference.

```text

```

## 5. Combination of All Techniques

### 5.1 Implementation

Please copy and paste your implementation in `all_techniques.cc`

```cpp

```

### 5.2 Comparative Study

How does the performance in GOPs, achieved through all optimization techniques on your computer, compare to the reference implementation? Please explain the performance difference.

```text

```

## 6. Additional Experiments

Any optimization techniques on your mind? Try to implement them to improve the performance further! Create a pull request in [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) and get verified by the TA.
