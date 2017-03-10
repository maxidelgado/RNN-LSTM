from Keras import TrainLSTM as tr

rnn = tr.EntrenarLSTM('/home/maxi/PycharmProjects/Prueba/coindesk-bpi-USD-close_data-2010-07-17_2017-03-08.csv',
                      look_back=12,
                      batch_size=64)
dataset = rnn.cargarDataSet(columnas=[1],head=0,tail=0)
dataset = rnn.normalizarDataSet(dataset=dataset)
trainX, testX, trainY, testY = rnn.separarDataSet(dataset=dataset)
rnn.crearYEntrenarModelo(trainX=trainX,trainY=trainY,nombreModelo='modelo-1')
rnn.predecirYGraficar(dataset=dataset,trainX=trainX,testX=testX,trainY=trainY,testY=testY,nombreJSON='modelo-1')
