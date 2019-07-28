# Face and Smile detection

* É usada uma CNN pré-treinada no dataset [FDDB](http://vis-www.cs.umass.edu/fddb/).
O modelo está disponível no [GitHub da OpenCV](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector).
Basicamente são utilizadas várias bounding boxes na imagem e o modelo retorna a probabilidade de cada bounding
box conter uma face.
* Aqui é utilizada a região da imagem com maior probabilidade de conter a face, e então é utilizada uma outra CNN
treinada para detectar sorrisos no dataset [SMILEsmileD](https://github.com/hromi/SMILEsmileD/tree/master/SMILEs).
Este dataset contém 9475 exemplos negativos e 3690 exemplos positivos (pessoas sorrindo).
* Para executar o script é necessário passar o path para o face detector da OpenCV contendo o .txt e o .caffemodel 
('../../face\_detection\_model'); o nível de confiança que é utilizado para otimizar a detecção de sorrisos quando uma webcam é utilizada (default é 0.6); o path para o .json da cnn treinada para detectar sorrisos; o .h5 contendo os pesos dela; e o path para a imagem.
* Caso a função smile\_detection\_webcam não esteja comentada, a webcam será ativada e o vídeo output.avi salvo.
* Será printada a probabilidade da imagem passada como parâmetro conter um sorriso.
