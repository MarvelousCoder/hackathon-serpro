# Face recognition methods

## Compare faces

* Usa uma RPN para propor possíveis rostos a imagem e uma CNN para classificar esse possiveis rostos
* Usa uma outra CNN para detectar 5 pontos chaves no rosto da pessoa (ponta do nariz e olhos) e alinhar o rosto da pessoa
* Usa uma CNN para extrair uma assinatura digital do rosto da pessoa sendo um vetor de 128 dimensoes. 
* Comparar os vetores de 128 dimensoes entre as imagens de duas pessoas e gera um valor de similaridade.
* Para valores de similaridade menores que 0.6 foi alcançada uma acurácia de 99.38% no dataset LFW.
* Usamos 0.5 de threshold para combater a diferença de domínios e garantir que é a pessoa correta