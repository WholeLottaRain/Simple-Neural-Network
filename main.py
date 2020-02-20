import numpy as np
import ast
from tkinter import *

# Default train settings
tacts = 10000
learn_tempo = 0.05


# nn class
class GotoStreet(object):
    def __init__(self, learn_temp=0.05):
        self.sigmoid_init = np.vectorize(func)
        self.learn_temp = np.array([learn_temp])
        self.weights_01 = np.random.normal(0.0, 1, (3, 6))
        self.weights_12 = np.random.normal(0.0, 1, (1, 3))

    def train(self, input_data, reality):
        # Calculation of the result (--->)
        input_01 = np.dot(self.weights_01, input_data)
        output_01 = self.sigmoid_init(input_01)
        input_12 = np.dot(self.weights_12, output_01)
        output_12 = self.sigmoid_init(input_12)
        actual_result = output_12
        # Weights correction        (<---)
        error_12 = np.array([actual_result - reality])
        gradient_12 = actual_result * (1 - actual_result)
        weights_delta_12 = error_12 * gradient_12
        self.weights_12 -= (np.dot(weights_delta_12, output_01.reshape(1, len(output_01)))) * self.learn_temp
        error_01 = weights_delta_12 * self.weights_12
        gradient_01 = output_01 * (1 - output_01)
        weights_delta_01 = error_01 * gradient_01
        self.weights_01 -= np.dot(input_data.reshape(len(input_data), 1), weights_delta_01).T * self.learn_temp

    def prediction(self, input_data):
        input01 = np.dot(self.weights_01, input_data)
        output01 = self.sigmoid_init(input01)
        input12 = np.dot(self.weights_12, output01)
        output12 = self.sigmoid_init(input12)
        return output12


# check prediction miss
def error_check(k1, k2):
    return np.mean((k1 - k2) ** 2)


# get train data from txt
def row_data_loader():
    array = []
    with open("data.txt") as file:
        row_data = [data.strip() for data in file]
    for k in range(len(row_data)):
        array.append(ast.literal_eval(row_data[k]))
    return array


def func(x):
    return 1 / (1 + np.exp(-x))


def train():
    global learn_tempo
    global tacts
    btn_train.configure(state=DISABLED)
    btn_input.configure(state=DISABLED)
    learn_tempo = float(txt_tempo.get())
    tacts = int(txt_tacts.get())
    train_data = row_data_loader()
    for i in range(tacts):
        entered = []
        correct_results = []
        for input_case, correct_result in train_data:
            neuron.train(np.array(input_case), correct_result)
            entered.append(np.array(input_case))
            correct_results.append(np.array(correct_result))
        print("Loss: " + str(error_check(neuron.prediction(np.array(entered).T), np.array(correct_results))))
    btn_train.configure(state="normal")
    btn_input.configure(state="normal")


# get the entered case and transform it to array
def get_input():
    test = []
    for i in range(len(txt_input.get())):
        test.append(int(txt_input.get()[i]))
    lbl_prob2.configure(text=str(neuron.prediction(np.array(test))))


# init
neuron = GotoStreet(learn_temp=learn_tempo)

# window config
window = Tk()
window.title("Simple Neuro")
lbl_tacts = Label(window, text="Tacts", font=("Consolas", 15), fg="Black")
lbl_tacts.grid(column=0, row=0)
lbl_tempo = Label(window, text="Tempo", font=("Consolas", 15), fg="Black")
lbl_tempo.grid(column=1, row=0)
lbl_input = Label(window, text="Input", font=("Consolas", 15), fg="Black")
lbl_input.grid(column=1, row=2)
lbl_prob = Label(window, text="%", font=("Consolas", 15), fg="Black")
lbl_prob.grid(column=0, row=2)
lbl_prob2 = Label(window, text="", font=("Consolas", 10), fg="green")
lbl_prob2.grid(column=0, row=3)
txt_input = Entry(window)
txt_input.grid(column=1, row=3)
txt_tacts = Entry(window)
txt_tacts.grid(column=0, row=1)
txt_tempo = Entry(window)
txt_tempo.grid(column=1, row=1)
btn_train = Button(window, text="Train", bg="green", command=train)
btn_train.grid(column=2, row=1)
btn_input = Button(window, text="Check", bg="green", command=get_input)
btn_input.grid(column=2, row=3)
txt_help = Label(window, text="Example: 110010\n1. Rain?\n2. Holiday?\n3. Sick?\n4. Friends?\n5. Doctor?\n6. Exams?\n")
txt_help.grid(column=1, row=4)
window.geometry('300x240')
window.mainloop()
