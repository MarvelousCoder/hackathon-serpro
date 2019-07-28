from flask import Flask, render_template, request, jsonify, Response
import base64
import re
from src.face_rec.eye_closed import Eye_Closed
from src.face_rec.compare_cnh import face_compare
from src.face_rec.extract_cnh import recognize_card
from src.face_rec.face_angle import Face_angle
from src.smile_detection.smile_detection import smile_detection_image
import io
import os
import cv2
from tensorflow.keras.models import model_from_json

model = model_from_json(open('notebooks/model.json').read())
model.load_weights('notebooks/weights.h5')
detector = cv2.dnn.readNetFromCaffe('face_detection_model/deploy.prototxt', 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
embedder = cv2.dnn.readNetFromTorch('face_detection_model/openface_nn4.small2.v1.t7')

app = Flask(__name__)             # create an app instance

@app.route("/")                   # at the end point /
def hello():                      # call method hello
    # return "Hello World!"         # which returns "hello world"
    return render_template('main.html')

@app.route("/cnh", methods = ['GET'])
def cnh():
    return render_template('cnh.html')

@app.route("/selfie", methods = ['GET'])
def selfie():
    return render_template('selfie.html')

@app.route("/api", methods = ['POST'])                   # 	at the end point /
def processa():                      # call method hello
    # return "Hello World!"         # which returns "hello world"

    seq = 0
    acertos = 0
    resultado = []

    for item in request.json:

        action = item['action'].replace(' ', '-')
        img = item['img']['myimage']

        seq += 1
        with open("{}-{}.png".format(action, seq), "wb") as img_file:
            match = re.search(',', img)
            i, j = match.span()
            imgdata = base64.b64decode(img[j:])
            img_file.write(imgdata)

        path = "{}-{}.png".format(action, seq)

        acerto = 0

        if action == 'rosto-reto':

            try:
                probability = face_compare(detector, confidence=0.45, base_img="cnh.jpg", test_img=path, embedder=embedder)

                acerto = probability

            except:
                pass


        elif action != 'sorrir':
            angle = Face_angle(base64image=path)

            try:

                if action == 'virar-rosto-para-direita':
                    if angle > 315 or angle < 45:
                        acerto = 1
                elif action == 'virar-rosto-para-cima':
                    if angle > 225 and angle < 315:
                        acerto = 1
                elif action == 'virar-rosto-para-baixo':
                    if angle > 45 and angle < 135:
                        acerto = 1
                elif action == 'virar-rosto-para-esquerda':
                    if angle > 135 and angle < 225:
                        acerto = 1

            except:
                pass

        elif smile_detection_image(detector, 0.5, model, path) > 0.5:
                acerto = 1

        acertos += acerto

        resultado.append({'action': action, 'resultado': acerto})

    if acertos >= 3.3:
        resultado.append({'action': 'resultado', 'resultado': "Login OK"})
    else:
        resultado.append({'action': 'resultado', 'resultado': "Login Falhou"})

    return jsonify(resultado)


@app.route("/read-cnh", methods = ['POST'])
def processa_actions():

    image_bytes = request.json['myimage']
    match = re.search(',', image_bytes)
    i, j = match.span()
    imgdata = base64.b64decode(image_bytes[j:])

    with open("cnh.jpg", "wb") as img_file:
        img_file.write(imgdata)

    info = read_chn_text("cnh.jpg")

    return jsonify({'response': info})


# @app.route("/read-cnh", methods = ['GET'])
def read_chn_text(img_file):
    path = os.path.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path) + "/gcloud.json"

    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    # for img_file in [x for x in os.listdir('.') if '.jpg' in x and 'tmp' not in x]:

    recognize_card(img_file, "tmp.jpg")

    # img = cv2.imread('tmp.jpg', cv2.IMREAD_UNCHANGED)
    # dim = (925, 672)
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # cv2.imwrite('tmp.jpg', resized)

    with open("tmp.jpg", "rb") as img:
        content = img.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    infos = {
        'nome': '',
        'cpf': '',
        # 'rg': '',
        'dt_nascimento': ''
        # 'nome_pai': '',
        # 'nome_mae': ''
    }

    # nome_mae = []
    # nome_pai = []
    nome = []
    # rg = []
    cpf = None
    dt_nascimento = None

    nome_pos = None

    for text in texts[1:]:
        print('\n"{}"'.format(text.description))

        if re.search('\d{3}\.\d{3}\.\d{3}\-\d{2}', text.description) and not cpf:
            cpf = re.search('\d{3}\.\d{3}\.\d{3}\-\d{2}', text.description).group(0)
            continue

        if re.search('\d{2}\/\d{2}\/\d{4}', text.description) and not dt_nascimento:
            dt_nascimento = re.search('\d{2}\/\d{2}\/\d{4}', text.description).group(0)
            continue

        # vertices = ([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices if vertex.y > 230 and vertex.y < 560])
        vertices = (
        [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])

        if not nome_pos and re.search("N?OME?", text.description):
            nome_pos = vertices
            continue

        # if vertices[0][0] > 400 and vertices[0][1] > 350 and vertices[0][1] < 420:
        #     nome_pai.append(text.description)
        #
        # if vertices[0][0] > 400 and vertices[0][1] > 420 and vertices[0][1] < 490:
        #     nome_mae.append(text.description)

        if nome_pos and vertices[0][1] > nome_pos[2][1] and vertices[2][1] < nome_pos[2][1] + 35:
            nome.append(text.description)

        # if vertices[0][0] > 400 and vertices[0][1] > 230 and vertices[0][1] < 290:
        #     rg.append(text.description)

        # if vertices[0][0] > 400 and vertices[0][1] > 290 and vertices[0][1] < 310 and vertices[1][0] < 690:
        #     cpf.append(text.description)
        #
        # if vertices[0][0] > 690 and vertices[0][1] > 290 and vertices[0][1] < 310 and vertices[1][0] < 850:
        #     dt_nascimento.append(text.description)

        print('bounds: ', vertices)

    infos['nome'] = ' '.join(nome)
    infos['cpf'] = cpf
    infos['dt_nascimento'] = dt_nascimento
    # infos['rg'] = ' '.join(rg)
    # infos['nome_mae'] = ' '.join(nome_mae)
    # infos['nome_pai'] = ' '.join(nome_pai)

    return infos

if __name__ == "__main__":
    # app.run(debug=True, ssl_context='adhoc')
    app.run(debug=True)
