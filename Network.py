import random
import math


class Network:
    layers = []
    not_activated_layers = []
    weights = {}

    learn_images = []
    learn_matrixes = []
    learn_sets = []
    learn_outputs = []
    learn_outputs_errors = []

    def __init__(self, layers_amount, neuron_amount_array):
        self.learn_sets = []
        self.learn_images = []
        self.learn_matrixes = []
        self.learn_outputs = []
        self.learn_layers_errors = []

        self.layers = []
        self.not_activated_layers = []

        self.learning_rate = 0.01

        for i in range(0, layers_amount):
            self.layers.append([])
            self.not_activated_layers.append([])

            if not neuron_amount_array[i]:
                break

            for j in range(0, neuron_amount_array[i]):
                self.layers[i].append(0)
                self.not_activated_layers[i].append(0)

        self.weights = {}
        for i in range(0, layers_amount - 1):
            index = str(i) + '_' + str(i + 1)
            self.weights[index] = []

            neural_amount_first_layer = neuron_amount_array[i]
            neural_amount_second_layer = neuron_amount_array[i + 1]

            for x in range(0, neural_amount_first_layer):
                self.weights[index].append([])

                for y in range(0, neural_amount_second_layer):
                    rdn = random.uniform(-1.0, 1.0)
                    self.weights[index][x].append(rdn)

    def add_learn_image(self, image, expected_output):
        self.learn_images.append(image)
        self.learn_outputs.append(expected_output)

    def convert_images_to_matrixes(self, images, net_matrix):
        for image in images:
            matrix = []

            for x in range(0, image.shape[0]):
                matrix.append([])
                for y in range(0, image.shape[1]):
                    if image[x, y] == 255:
                        matrix[x].append(-1)
                    else:
                        matrix[x].append(1)

            net_matrix.append(matrix)

    def convert_matrixes_to_vectors(self, matrixes, vectors):
        for index, matrix in enumerate(matrixes):
            vectors.append([])
            for x in range(0, len(matrix)):
                for y in range(0, len(matrix[x])):
                    vectors[index].append(matrix[x][y])

    def learn(self):
        self.convert_images_to_matrixes(self.learn_images, self.learn_matrixes)
        self.convert_matrixes_to_vectors(self.learn_matrixes, self.learn_sets)

        for learn_stage in range(2000):
            for learn_index, learn_set in enumerate(self.learn_sets):
                self.activate(learn_set)

                net_error_e = self.calculate_net_error(learn_index)
                print(net_error_e)

                self.calculate_layers_error(learn_index)

        self.activate(self.learn_sets[0])
        c = 3

    def calculate_layers_error(self, learn_index):
        self.clear_errors()

        current_output = self.layers[-1]
        expected_output = self.learn_outputs[learn_index]

        current_layer_index = len(self.layers) - 1

        for i in range(0, len(self.learn_layers_errors[current_layer_index])):
            d = current_output[i] - expected_output[i]
            self.learn_layers_errors[current_layer_index][i] = d * (current_output[i] * (1 - current_output[i]))



        current_layer_index -= 1
        index = str(current_layer_index) + '_' + str(current_layer_index + 1)
        weights = self.weights[index]

        for i in range(0, len(self.learn_layers_errors[current_layer_index])):
            hidden_error = 0
            for j in range(0, len(self.learn_layers_errors[current_layer_index + 1])):
                hidden_error += self.learn_layers_errors[current_layer_index + 1][j] * weights[i][j]

            self.learn_layers_errors[current_layer_index][i] = hidden_error * (self.layers[current_layer_index][i] * (1 - self.layers[current_layer_index][i]))

        self.update_weights(1, 2)
        self.update_weights(0, 1)

    def clear_errors(self):
        self.learn_layers_errors = []
        for i in range(0, len(self.layers)):
            self.learn_layers_errors.append([])
            for j in range(0, len(self.layers[i])):
                self.learn_layers_errors[i].append(0)

    def update_weights(self, first_index, second_index):
        index = str(first_index) + '_' + str(second_index)
        weights = self.weights[index]

        for i in range(0, len(weights)):
            for j in range(0, len(weights[i])):
                weights[i][j] -= self.learning_rate * self.learn_layers_errors[second_index][j] * self.layers[first_index][i]

        self.weights[index] = weights

    def calculate_net_error(self, learn_index):
        sum = 0
        current_output = self.layers[-1]
        expected_output = self.learn_outputs[learn_index]

        for i in range(0, len(current_output)):
            sum += math.pow(current_output[i] - expected_output[i], 2)

        return sum / 2

    def neuron_activate(self, value, derivative=False):
        if derivative:
            return self.neuron_activate(value) * (1 - self.neuron_activate(value))
        else:
            return 1 / (1 + math.exp(-1 * value))

    def activate(self, input_vector):
        self.layers[0] = input_vector

        for layer_index, current_layer in enumerate(self.layers):
            for neuron_index in range(0, len(current_layer)):
                if layer_index == 0:
                    current_layer[neuron_index] = self.neuron_activate(current_layer[neuron_index])
                else:
                    current_layer[neuron_index] = 0

                    weight_matrix_index = str(layer_index - 1) + '_' + str(layer_index)
                    weights = self.weights[weight_matrix_index]
                    for weight_index in range(0, len(weights)):
                        current_layer[neuron_index] += weights[weight_index][neuron_index] * self.layers[layer_index - 1][weight_index]

                    self.not_activated_layers[layer_index][neuron_index] = current_layer[neuron_index]
                    current_layer[neuron_index] = self.neuron_activate(current_layer[neuron_index])
