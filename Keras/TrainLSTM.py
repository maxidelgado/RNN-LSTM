# LSTM apilados con memoria entre lotes
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json

class EntrenarLSTM:
    __path = ''
    __train_size = 0
    __test_size = 0
    __look_back = 1
    __batch_size = 64
    __scaler = 0

    # fijar un random seed para reproductibilidad
    numpy.random.seed(7)

    def __init__(self, path, look_back, batch_size):
        self.__path = path
        self.__look_back = look_back
        self.__batch_size = batch_size

    def __createDataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.__look_back - 1):
            a = dataset[i:(i + self.__look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.__look_back, 0])

        return numpy.array(dataX), numpy.array(dataY)

    def cargarDataSet(self, columnas = [1], head = 1, tail = 3):
        dataframe = pandas.read_csv(
            self.__path, usecols=columnas,
            engine='python',skiprows=head, skipfooter=tail)
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        return dataset

    def normalizarDataSet(self, dataset):
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.__scaler.fit_transform(dataset)

        return dataset

    def separarDataSet(self, dataset):
        train_size, test_size = self.__calcularTamanoDataSet(len(dataset))
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1
        trainX, trainY = self.__createDataset(train)
        testX, testY = self.__createDataset(test)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
        print(len(trainX))

        return trainX, testX, trainY, testY

    def __calcularTamanoDataSet(self, length):
        self.__train_size = length * 0.67
        if (self.__train_size % self.__batch_size) != 0:
            self.__train_size = (self.__train_size // self.__batch_size) * self.__batch_size + self.__look_back + 1

        if ((length - self.__train_size) % self.__batch_size) != 0:
            self.__test_size = (length // self.__batch_size) * self.__batch_size + self.__look_back + 1

        return int(self.__train_size), int(self.__batch_size)

    def crearYEntrenarModelo(self, trainX, trainY, nombreModelo):
        model = Sequential()
        model.add(LSTM(32, batch_input_shape=(self.__batch_size, self.__look_back, 1), stateful=True, return_sequences=True))
        model.add(LSTM(32, batch_input_shape=(self.__batch_size, self.__look_back, 1), stateful=True, return_sequences=True))
        model.add(LSTM(32, batch_input_shape=(self.__batch_size, self.__look_back, 1), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())

        for i in range(100):
            print('Iteracion: %d' % i)
            model.fit(trainX, trainY, nb_epoch=5, batch_size=self.__batch_size, verbose=2, shuffle=False)
            model.reset_states()

        # serialize model to JSON
        model_json = model.to_json()
        with open(nombreModelo+ '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(nombreModelo + '.h5')
        print("Saved model to disk")

    def __cargarModelo(self, nombreJSON):
        # load json and create model
        json_file = open('/home/maxi/PycharmProjects/Prueba/Keras/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("/home/maxi/PycharmProjects/Prueba/Keras/model.h5")
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        print("Loaded model from disk")

    def predecirYGraficar(self, dataset, trainX, testX, trainY, testY, nombreJSON):
        # make predictions
        loaded_model = self.__cargarModelo(nombreJSON)
        trainPredict = loaded_model.predict(trainX, batch_size=self.__batch_size)
        loaded_model.reset_states()
        testPredict = loaded_model.predict(testX[0:self.__test_size, :], batch_size=self.__batch_size)

        # invert predictions
        trainPredict = self.__scaler.inverse_transform(trainPredict)
        trainY = self.__scaler.inverse_transform([trainY])
        testPredict = self.__scaler.inverse_transform(testPredict)
        testY = self.__scaler.inverse_transform([testY])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[:, :self.__test_size][0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[self.__look_back:len(trainPredict) + self.__look_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(dataset)
        testPredictPlot[:, :] = numpy.nan
        print(testPredictPlot.shape)
        testPredictPlot[len(trainPredict) + (self.__look_back * 2) + 1:len(dataset) - 28, :] = testPredict

        # plot baseline and predictions
        plt.plot(self.__scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()
