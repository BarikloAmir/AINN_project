import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import datetime

def load_data():
    # loading training set features
    f = open("Datasets/train_set_features.pkl", "rb")
    train_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=train_set_features2, axis=0)
    train_set_features = train_set_features2[:, features_STDs > 52.3]

    # changing the range of data between 0 and 1
    train_set_features = np.divide(train_set_features, train_set_features.max())

    # loading training set labels
    f = open("Datasets/train_set_labels.pkl", "rb")
    train_set_labels = pickle.load(f)
    f.close()

    # ------------
    # loading test set features
    f = open("Datasets/test_set_features.pkl", "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > 48]

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading test set labels
    f = open("Datasets/test_set_labels.pkl", "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # ------------
    # preparing our training and test sets - joining datasets and lables
    train_set = []
    test_set = []

    for i in range(len(train_set_features)):
        label = np.array([0, 0, 0, 0, 0])
        label[int(train_set_labels[i])] = 1
        label = label.reshape(5, 1)

        index = [2, 3, 6]
        temp = np.delete(train_set_features[i], index)
        train_set.append((temp.reshape(102, 1), label))

    for i in range(len(test_set_features)):
        label = np.array([0, 0, 0, 0, 0])
        label[int(test_set_labels[i])] = 1
        label = label.reshape(5, 1)
        index2 = [2, 3, 6, 10 , 11 , 12 , 13 ]
        temp2 = np.delete(test_set_features[i], index2)
        test_set.append((temp2.reshape(102, 1), label))

    # shuffle
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set,test_set


def neuralNetTrain(train_set_input, test_set,learning_rate, epoch, batch_size, with_for:bool , image_number):

    # sigmoid function for value of list
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(x):
        return sigmoid(x) * (1 - sigmoid(x))

    # initializing the weights randomly and bayas with 0
    def generate_wt(d1, d2, bayas: bool):
        temp_list = []
        for i in range(d1 * d2):
            if not bayas:
                temp_list.append(np.random.randn())
            else:
                temp_list.append(0.0)
        return np.array(temp_list).reshape(d1, d2)

    # Weights and bayas of hidden layer 1
    w1 = generate_wt(150, 102, False)
    b1 = generate_wt(150, 1, True)

    # Weights of hidden layer 2
    w2 = generate_wt(60, 150, False)
    b2 = generate_wt(60, 1, True)

    # Weights of hidden output layer
    w3 = generate_wt(5, 60, False)
    b3 = generate_wt(5, 1, True)


    def forward_propagation(input_vector):
        # output of hidden layer1
        z1 = w1.dot(input_vector)
        z1 = np.add(z1, b1)
        a1 = sigmoid(z1)

        # Output of hidden layer2
        z2 = w2.dot(a1)
        z2 = np.add(z2, b2)
        a2 = sigmoid(z2)

        # output
        z3 = w3.dot(a2)
        z3 = np.add(z3, b3)
        output = sigmoid(z3)

        return output

    def back_propagation_with_for(input_vector, y, grad_w3, grad_w2, grad_w1, grad_b3, grad_b2, grad_b1):
        # output of hidden layer1
        z1 = w1.dot(input_vector)
        z1 = np.add(z1, b1)
        a1 = sigmoid(z1)

        # Output of hidden layer2
        z2 = w2.dot(a1)
        z2 = np.add(z2, b2)
        a2 = sigmoid(z2)

        # output
        z3 = w3.dot(a2)
        z3 = np.add(z3, b3)
        output = sigmoid(z3)

        for j in range(5):
            grad_b3_temp = sigmoid_deriv(z3[j, 0]) * (2 * output[j, 0] - 2 * y[j, 0])
            grad_b3[j, 0] += grad_b3_temp
            grad_w3[j] += a2.reshape(60) * grad_b3_temp

        grad_a2 = np.zeros((60, 1))
        for k in range(60):
            for j in range(5):
                grad_a2[k, 0] += w3[j, k] * sigmoid_deriv(z3[j, 0]) * (2 * output[j, 0] - 2 * y[j, 0])

        for j in range(60):
            grad_b2_temp = sigmoid_deriv(z2[j, 0]) * (grad_a2[j, 0])
            grad_b2[j, 0] += grad_b2_temp
            grad_w2[j] = a1.reshape(150) * grad_b2_temp

        grad_a1 = np.zeros((150, 1))
        for k in range(150):
            for j in range(60):
                grad_a1[k, 0] += w2[j, k] * sigmoid_deriv(z2[j, 0]) * (grad_a2[j, 0])

        for j in range(150):
            grad_b1_temp = sigmoid_deriv(z1[j, 0]) * (grad_a1[j, 0])
            grad_b1 += grad_b1_temp
            grad_w1[j] += input_vector.reshape(102) * grad_b1_temp

        return output, grad_w3, grad_w2, grad_w1, grad_b3, grad_b2, grad_b1

    def back_propagation(input_vector, y, grad_w3, grad_w2, grad_w1, grad_b3, grad_b2, grad_b1):
        # output of hidden layer1
        z1 = w1.dot(input_vector)
        z1 = np.add(z1, b1)
        a1 = sigmoid(z1)

        # Output of hidden layer2
        z2 = w2.dot(a1)
        z2 = np.add(z2, b2)
        a2 = sigmoid(z2)

        # output
        z3 = w3.dot(a2)
        z3 = np.add(z3, b3)
        output = sigmoid(z3)

        grad_output = 2 * (output - y)
        grad_b3_temp = (sigmoid_deriv(z3) * grad_output)
        grad_b3 += grad_b3_temp
        grad_w3 += grad_b3_temp @ (np.transpose(a2))

        grad_a2 = np.transpose(w3) @ (sigmoid_deriv(z3) * grad_output)
        grad_b2_temp = sigmoid_deriv(z2) * grad_a2
        grad_b2 += grad_b2_temp
        grad_w2 += grad_b2_temp @ (np.transpose(a1))

        grad_a1 = np.transpose(w2) @ (sigmoid_deriv(z2) * grad_a2)
        grad_b1_temp = sigmoid_deriv(z1) * grad_a1
        grad_b1 += grad_b1_temp
        grad_w1 += grad_b1_temp @ (np.transpose(input_vector))

        return output, grad_w3, grad_w2, grad_w1, grad_b3, grad_b2, grad_b1

    def check(output_real, correct_output):
        result = np.where(output_real == np.amax(output_real))
        if correct_output[result] == 1:
            return True
        return False

    def test(test_set):
        true = 0
        for image in test_set:
            output = forward_propagation(image[0])
            if check(output, image[1]):
                true += 1
        print("RESULT IN ONE TEST : ",100 * (true / len(test_set)))
        return 100 * (true / len(test_set))



    x = []
    y = []

    train_set = train_set_input[:image_number]
    for epoch_counter in range(epoch):
        random.shuffle(train_set)
        number = len(train_set) / batch_size
        batch_set = np.array_split(train_set, int(number))
        true = 0
        cost = 0
        for batch in batch_set:
            grad_w1 = generate_wt(150, 102, True)
            grad_b1 = generate_wt(150, 1, True)
            grad_w2 = generate_wt(60, 150, True)
            grad_b2 = generate_wt(60, 1, True)
            grad_w3 = generate_wt(5, 60, True)
            grad_b3 = generate_wt(5, 1, True)


            for image in batch:
                if not with_for:
                    output, grad_w3, grad_w2, grad_w1, grad_b3, grad_b2, grad_b1 = back_propagation(image[0], image[1],
                                                                                                    grad_w3, grad_w2,
                                                                                                    grad_w1, grad_b3,
                                                                                                    grad_b2, grad_b1)
                else:
                    output, grad_w3, grad_w2, grad_w1, grad_b3, grad_b2, grad_b1 = back_propagation_with_for(image[0],
                                                                                                             image[1],
                                                                                                             grad_w3,
                                                                                                             grad_w2,
                                                                                                             grad_w1,
                                                                                                             grad_b3,
                                                                                                             grad_b2,
                                                                                                             grad_b1)


                cost += np.sum(np.square(output - image[1]))

            w1 = w1 - (learning_rate * (grad_w1 / batch_size))
            w2 = w2 - (learning_rate * (grad_w2 / batch_size))
            w3 = w3 - (learning_rate * (grad_w3 / batch_size))

            b1 = b1 - (learning_rate * (grad_b1 / batch_size))
            b2 = b2 - (learning_rate * (grad_b2 / batch_size))
            b3 = b3 - (learning_rate * (grad_b3 / batch_size))

        x.append(epoch_counter+1)
        y.append(cost/ image_number)



    return  test(test_set), x , y









if __name__ == '__main__':

    start_time = datetime.datetime.now()
    train_set, test_set = load_data()
    print(len(train_set),len(test_set))
    accuracy = 0
    repeating_number = 5
    learning_rate = 0.5
    for i in range(repeating_number):
        new_accuracy,x,y= neuralNetTrain(train_set, test_set,learning_rate, 5, 10, False, 1960+738)
        learning_rate+=0.1
        accuracy += new_accuracy
        plt.plot(x,y)
    plt.show()
    print("TOTAL RESULT : ", accuracy/repeating_number)
    print("TOTAL RUNTIME : ", datetime.datetime.now() - start_time)