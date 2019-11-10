import os, cv2, io, re
from src.face_rec.extract_cnh import recognize_card

def read_chn_text(img_file):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/raphasramos/dev/face_rec/gcloud.json"

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
        # 'nome': '',
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

    # infos['nome'] = ' '.join(nome)
    infos['cpf'] = cpf
    infos['dt_nascimento'] = dt_nascimento
    #infos['rg'] = ' '.join(rg)
    #infos['nome_mae'] = ' '.join(nome_mae)
    #infos['nome_pai'] = ' '.join(nome_pai)

    return infos