#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void mat_mul(const float *X, const float *Y, float *Z, int m, int n, int k) {
    // X: m x n, Y: n x k, Z: m x k
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            float acc = 0.0f;
            for (int l = 0; l < n; ++l) {
                acc += X[i * n + l] * Y[l * k + j];
            }
            Z[i * k + j] = acc;
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    // iteration = (y.size + batch - 1) // batch
    // for i in range(iteration):
    //     xx = X[i * batch : (i + 1) * batch, :]
    //     yy = y[i * batch : (i + 1) * batch]
    //     Z = np.exp(xx @ theta)
    //     Z = Z / np.sum(Z, axis=1, keepdims=True)
    //     Y = np.zeros((batch, y.max() + 1))
    //     Y[np.arange(batch), yy] = 1
    //     grad = xx.T @ (Z - Y) / batch
    //     assert(grad.shape == theta.shape)
    //     theta -= lr * grad

    /// BEGIN YOUR CODE
    // X: m x n, y: 1 x m, theta: n x k
    size_t iteration = (m + batch - 1) / batch;
    for (size_t iter = 0; iter < iteration; ++iter) {
        // x @ theta = Z -- (batch x n) * (n x k) = (batch x k)
        const float *x = &X[iter * batch * n];
        float *Z = new float[batch * k];
        mat_mul(x, theta, Z, batch, n, k);
        // element-wise exp
        for (size_t i = 0; i < batch * k; ++i) {
            Z[i] = exp(Z[i]);
        }
        // row-wise normalization
        for (size_t i = 0; i < batch; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                sum += Z[i * k + j];
            }
            for (size_t j = 0; j < k; ++j) {
                Z[i * k + j] /= sum;
            }
        }

        // calculate (Z - Y); minus one-hot vector
        for (size_t i = 0; i < batch; ++i) {
            Z[i * k + y[iter * batch + i]] -= 1;
        }
        float *x_T = new float [n * batch];
        // transpose x
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < n; ++j) {
                x_T[j * batch + i] = x[i * n + j];
            }
        }
        float *grad = new float[n * k];
        mat_mul(x_T, Z, grad, n, batch, k);
        // SGD update
        for (size_t i = 0; i < n * k; ++i) {
            theta[i] -= lr * grad[i] / batch;
        }

        delete[] Z;
        delete[] x_T;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
