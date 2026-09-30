// Avoid cancellation in device tanh implementations near zero. SiTU applies
// tanh to small gate/beta and up/linear_beta values; their absolute error is
// amplified by beta and subsequent residual normalization. The odd series
// through x^13 has truncation error below 2e-12 on this interval.
float fp32_tanh(float value) {
    if (abs(value) <= 0.25) {
        float x2 = value * value;
        float p = 21844.0 / 6081075.0;
        p = fma(x2, p, -1382.0 / 155925.0);
        p = fma(x2, p, 62.0 / 2835.0);
        p = fma(x2, p, -17.0 / 315.0);
        p = fma(x2, p, 2.0 / 15.0);
        p = fma(x2, p, -1.0 / 3.0);
        return fma(value * x2, p, value);
    }
    return tanh(value);
}
