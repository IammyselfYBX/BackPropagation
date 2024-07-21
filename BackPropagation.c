/*
 * 反向传播算法
 *
 * 要求
 * 1.用户需要自己输入
 *  1)输入层,隐藏层,输出层的节点个数,
 *  2)网络的输入层的输入值，各层权重和偏置，期望的目标输出
 *  3)学习率
 *  4)迭代次数
 *  5)损失函数的误差epsilon
 * 2.误差函数用函数指针可以选择
 *  其中误差函数有
 *    1)二次误差函数
 *    2)交叉熵误差函数
 * 3.反向传播算法中包含学习率、偏执等参数
 * 4.激活函数用函数指针可以选择
 *  其中激活函数有
 *    1)线性激活函数
 *    2)sigmod
 *    3)relu
 *    4)softmax
 *    5)双曲正切
 * 如果使用交叉熵误差函数，就不会使用线性或softmax激活函数的导数
 * 5.每次更新权值需要输出新的权值
 * 6.用误差函数计算误差，当符合给出的误差范围epsilon时，跳出迭代
 * 7.包含过程:
 *    前向传播
 *    反向传播
 *    更新权值
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double (*ActivationFunction)(double);
typedef double (*ActivationFunctionDerivative)(double);
typedef double (*LossFunction)(double, double);
typedef double (*LossFunctionDerivative)(double, double);
double linear(double x) {
    return x;
}
double linear_derivative(double x) {
    return 1.0;
}
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}
double relu(double x) {
    return x > 0 ? x : 0;
}
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}
void softmax(double* input, double* output, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}
double tanh_function(double x) {
    return tanh(x);
}
double tanh_derivative(double x) {
    return 1.0 - x * x;
}
double mse(double y_true, double y_pred) {
    return 0.5 * (y_true - y_pred) * (y_true - y_pred);
}
double mse_derivative(double y_true, double y_pred) {
    return y_pred - y_true;
}
double cross_entropy(double y_true, double y_pred) {
    return -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred);
}
double cross_entropy_derivative(double y_true, double y_pred) {
    return y_pred - y_true;
}
void forward(double* input, double* weights, double* biases, double* output, int input_size, int output_size, ActivationFunction activation, int is_softmax) {
    for (int j = 0; j < output_size; j++) {
        output[j] = 0;
        for (int i = 0; i < input_size; i++) {
            output[j] += input[i] * weights[i * output_size + j];
        }
        output[j] += biases[j];
        if (!is_softmax) {
            output[j] = activation(output[j]);
        }
    }
    if (is_softmax) {
        softmax(output, output, output_size);
    }
}

void backpropagation(double* input, double* weights, double* biases, double* output, double* target, double* deltas, int input_size, int output_size, double learning_rate, ActivationFunctionDerivative activation_derivative, LossFunctionDerivative loss_derivative, int is_softmax) {
    for (int j = 0; j < output_size; j++) {
        double error = loss_derivative(target[j], output[j]);
        deltas[j] = is_softmax ? error : error * activation_derivative(output[j]);
        for (int i = 0; i < input_size; i++) {
            weights[i * output_size + j] -= learning_rate * deltas[j] * input[i];
        }
        biases[j] -= learning_rate * deltas[j];
    }
}

void print_weights_and_biases(double* weights, double* biases, int input_size, int output_size) {
    printf("\t权重:\n");
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            printf("\t%f ", weights[i * output_size + j]);
        }
        printf("\n");
    }
    printf("\t偏置:\n");
    for (int j = 0; j < output_size; j++) {
        printf("\t%f ", biases[j]);
    }
    printf("\n");
}

int main() {
    int input_nodes, hidden_layers, hidden_nodes, output_nodes;

    printf("请输入输入层节点数: ");
    scanf("%d", &input_nodes);

    printf("请输入隐藏层数: ");
    scanf("%d", &hidden_layers);

    printf("请输入每个隐藏层的节点数: ");
    scanf("%d", &hidden_nodes);

    printf("请输入输出层节点数: ");
    scanf("%d", &output_nodes);

    double* input = (double*)malloc(input_nodes * sizeof(double));
    double* target = (double*)malloc(output_nodes * sizeof(double));

    double** weights = (double**)malloc((hidden_layers + 1) * sizeof(double*));
    double** biases = (double**)malloc((hidden_layers + 1) * sizeof(double*));
    double** layers = (double**)malloc((hidden_layers + 2) * sizeof(double*));
    double** deltas = (double**)malloc((hidden_layers + 1) * sizeof(double*));

    for (int i = 0; i < hidden_layers + 1; i++) {
        if (i == 0) {
            weights[i] = (double*)malloc(input_nodes * hidden_nodes * sizeof(double));
        } else if (i == hidden_layers) {
            weights[i] = (double*)malloc(hidden_nodes * output_nodes * sizeof(double));
        } else {
            weights[i] = (double*)malloc(hidden_nodes * hidden_nodes * sizeof(double));
        }
        biases[i] = (double*)malloc(hidden_nodes * sizeof(double));
        deltas[i] = (double*)malloc(hidden_nodes * sizeof(double));
    }

    layers[0] = input;
    for (int i = 1; i < hidden_layers + 1; i++) {
        layers[i] = (double*)malloc(hidden_nodes * sizeof(double));
    }
    layers[hidden_layers + 1] = (double*)malloc(output_nodes * sizeof(double));

    printf("请输入输入值（%d个）: ", input_nodes);
    for (int i = 0; i < input_nodes; i++) {
        scanf("%lf", &input[i]);
    }

    printf("请输入目标输出值（%d个）: ", output_nodes);
    for (int i = 0; i < output_nodes; i++) {
        scanf("%lf", &target[i]);
    }

    for (int i = 0; i < hidden_layers + 1; i++) {
        int input_size = (i == 0) ? input_nodes : hidden_nodes;
        int output_size = (i == hidden_layers) ? output_nodes : hidden_nodes;
        printf("请输入第%d层的权重（%d x %d个）: ", i+1, input_size, output_size);
        for (int j = 0; j < input_size * output_size; j++) {
            scanf("%lf", &weights[i][j]);
        }
        printf("请输入第%d层的偏置（%d个）: ", i+1, output_size);
        for (int j = 0; j < output_size; j++) {
            scanf("%lf", &biases[i][j]);
        }
    }

    int choice;
    ActivationFunction activation;
    ActivationFunctionDerivative activation_derivative;
    int is_softmax = 0;
    printf("请选择激活函数（1. 线性 2. sigmoid 3. relu 4. softmax 5. tanh）: ");
    scanf("%d", &choice);
    switch (choice) {
        case 1:
            activation = linear;
            activation_derivative = linear_derivative;
            break;
        case 2:
            activation = sigmoid;
            activation_derivative = sigmoid_derivative;
            break;
        case 3:
            activation = relu;
            activation_derivative = relu_derivative;
            break;
        case 4:
            activation = sigmoid;  // 使用Softmax时，前向传播时特别处理
            activation_derivative = sigmoid_derivative;
            is_softmax = 1;
            break;
        case 5:
            activation = tanh_function;
            activation_derivative = tanh_derivative;
            break;
        default:
            printf("无效选择，使用默认 sigmoid 激活函数。\n");
            activation = sigmoid;
            activation_derivative = sigmoid_derivative;
    }

    int loss_choice;
    LossFunction loss;
    LossFunctionDerivative loss_derivative;
    printf("请选择误差函数（1. 二次误差 2. 交叉熵误差）: ");
    scanf("%d", &loss_choice);
    switch (loss_choice) {
        case 1:
            loss = mse;
            loss_derivative = mse_derivative;
            break;
        case 2:
            loss = cross_entropy;
            loss_derivative = cross_entropy_derivative;
            break;
        default:
            printf("无效选择，使用默认二次误差函数。\n");
            loss = mse;
            loss_derivative = mse_derivative;
    }

    double learning_rate;
    printf("请输入学习率: ");
    scanf("%lf", &learning_rate);

    int iterations;
    printf("请输入迭代次数: ");
    scanf("%d", &iterations);

    double epsilon;
    printf("请输入误差epsilon: ");
    scanf("%lf", &epsilon);

    printf("\n====================================\n\n");

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < hidden_layers + 1; i++) {
            int input_size = (i == 0) ? input_nodes : hidden_nodes;
            int output_size = (i == hidden_layers) ? output_nodes : hidden_nodes;
            forward(layers[i], weights[i], biases[i], layers[i + 1], input_size, output_size, activation, i == hidden_layers && is_softmax);
        }

        double total_loss = 0.0;
        for (int i = 0; i < output_nodes; i++) {
            total_loss += loss(target[i], layers[hidden_layers + 1][i]);
        }

        printf("------------------------------------\n");
        printf("迭代 %d: 损失值: %f\n", iter + 1, total_loss);

        if (fabs(total_loss) < epsilon) {
            printf("训练在第%d次迭代时收敛，误差: %f\n", iter + 1, total_loss);
            break;
        }

        for (int i = hidden_layers; i >= 0; i--) {
            int input_size = (i == 0) ? input_nodes : hidden_nodes;
            int output_size = (i == hidden_layers) ? output_nodes : hidden_nodes;
            backpropagation(layers[i], weights[i], biases[i], layers[i + 1], target, deltas[i], input_size, output_size, learning_rate, activation_derivative, loss_derivative, i == hidden_layers && is_softmax);
        }

        printf("第%d次迭代后的权重和偏置:\n", iter + 1);
        for (int i = 0; i < hidden_layers + 1; i++) {
            int input_size = (i == 0) ? input_nodes : hidden_nodes;
            int output_size = (i == hidden_layers) ? output_nodes : hidden_nodes;
            printf("\t第%d层:\n", i + 1);
            print_weights_and_biases(weights[i], biases[i], input_size, output_size);
        }
    }

    free(input);
    free(target);
    for (int i = 0; i < hidden_layers + 1; i++) {
        free(weights[i]);
        free(biases[i]);
        free(deltas[i]);
    }
    free(weights);
    free(biases);
    free(deltas);
    for (int i = 1; i < hidden_layers + 2; i++) {
        free(layers[i]);
    }
    free(layers);

    return 0;
}



